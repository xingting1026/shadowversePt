from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any, Sequence

import torch

from bridge import TrainingBridge
from model import CandidateActorCritic, EncodingMetadata, load_model_state_compatible, metadata_from_checkpoint, tensorize


ACTION_KINDS = ("mulligan", "choice", "play", "attack", "activate", "evolve", "end")


def load_policy(path: Path, expected_policy: str, device: torch.device) -> tuple[CandidateActorCritic, dict[str, Any], EncodingMetadata]:
    saved = torch.load(path, map_location=device, weights_only=False)
    if saved.get("policy") != expected_policy or not saved.get("self_play"):
        raise RuntimeError(f"{path} is not the {expected_policy} league policy")
    metadata = metadata_from_checkpoint(saved["metadata"])
    model = CandidateActorCritic(metadata).to(device)
    load_model_state_compatible(model, saved["model"])
    model.eval()
    return model, saved, metadata


def policy_actions(
    model: CandidateActorCritic,
    observations: Sequence[dict[str, Any]],
    metadata: EncodingMetadata,
    device: torch.device,
) -> tuple[list[int], list[float], list[list[float]]]:
    if not observations:
        return [], [], []
    with torch.no_grad():
        logits, values = model(tensorize(observations, metadata, device))
        probabilities = logits.softmax(dim=-1)
        selected = logits.argmax(dim=-1)
    return (
        selected.cpu().tolist(),
        values.cpu().tolist(),
        [probabilities[row, : len(observation["actions"])].cpu().tolist() for row, observation in enumerate(observations)],
    )


def empty_diagnostics() -> dict[str, Any]:
    return {
        "decisions": 0,
        "actionKindCounts": [0] * len(ACTION_KINDS),
        "selectedProbabilityTotal": 0.0,
        "mulligans": 0,
        "redraws": 0,
    }


def observe_diagnostics(
    diagnostics: dict[str, dict[str, Any]],
    observations: Sequence[dict[str, Any]],
    actions: Sequence[int],
    probabilities: Sequence[Sequence[float]],
) -> None:
    for observation, action_index, row_probabilities in zip(observations, actions, probabilities):
        actor = observation["actor"]
        actor_diagnostics = diagnostics[actor]
        action = observation["actions"][action_index]
        kind = int(action["kind"])
        actor_diagnostics["decisions"] += 1
        actor_diagnostics["actionKindCounts"][kind] += 1
        actor_diagnostics["selectedProbabilityTotal"] += row_probabilities[action_index]
        if kind == 0:
            actor_diagnostics["mulligans"] += 1
            actor_diagnostics["redraws"] += int(action["numbers"][12] > 0.5)


def finalize_diagnostics(diagnostics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for actor, values in diagnostics.items():
        decisions = values["decisions"]
        mulligans = values["mulligans"]
        output[actor] = {
            "decisionsObserved": decisions,
            "actionKindCounts": dict(zip(ACTION_KINDS, values["actionKindCounts"])),
            "meanSelectedProbability": values["selectedProbabilityTotal"] / max(1, decisions),
            "mulligansObserved": mulligans,
            "mulliganRedrawRate": values["redraws"] / max(1, mulligans),
        }
    return output


def evaluate_order(
    repo: Path,
    deck: str,
    player_model: CandidateActorCritic,
    destruction_model: CandidateActorCritic,
    metadata: EncodingMetadata,
    device: torch.device,
    games: int,
    envs: int,
    base_seed: int,
    player_first: bool,
    diagnostics: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    with TrainingBridge(repo) as bridge:
        initialized = bridge.request({
            "cmd": "init", "envs": min(envs, games), "deck": deck,
            "baseSeed": base_seed, "fixedFirst": player_first,
            "record": False, "selfPlay": True,
        })
        if initialized["engineVersion"] != 3:
            raise RuntimeError("league recorder requires engine version 3")
        observations = initialized["observations"]
        while len(results) < games:
            actions = [0] * len(observations)
            selected_probabilities: list[list[float]] = [[] for _ in observations]
            for actor, model in (("player", player_model), ("ai", destruction_model)):
                rows = [index for index, observation in enumerate(observations) if observation["actor"] == actor]
                selected, _, probabilities = policy_actions(model, [observations[index] for index in rows], metadata, device)
                for row, action, row_probabilities in zip(rows, selected, probabilities):
                    actions[row] = action
                    selected_probabilities[row] = row_probabilities
            observe_diagnostics(diagnostics, observations, actions, selected_probabilities)
            response = bridge.request({"cmd": "step", "actions": actions})
            for item in response["items"]:
                if item["done"] and len(results) < games:
                    if item.get("truncated"):
                        raise RuntimeError(f"evaluation game seed {item['result']['seed']} was truncated")
                    results.append(item["result"])
            observations = [item["observation"] for item in response["items"]]
    return results


def wilson_interval(wins: int, games: int, z: float = 1.959963984540054) -> list[float]:
    if games <= 0:
        return [0.0, 1.0]
    probability = wins / games
    denominator = 1 + z * z / games
    center = (probability + z * z / (2 * games)) / denominator
    radius = z * math.sqrt(probability * (1 - probability) / games + z * z / (4 * games * games)) / denominator
    return [center - radius, center + radius]


def choose_replays(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    wins = [result for result in results if result["winner"] == "player"]
    losses = [result for result in results if result["winner"] == "ai"]
    selected: dict[str, dict[str, Any]] = {}
    if wins:
        selected["showcaseWin"] = max(wins, key=lambda result: (result["decisions"], result["globalTurn"]))
        selected["closeWin"] = min(wins, key=lambda result: (result["playerHp"], -result["globalTurn"]))
        selected["cleanWin"] = max(wins, key=lambda result: (result["playerHp"], -result["globalTurn"]))
        first_wins = [result for result in wins if result["playerFirst"]]
        second_wins = [result for result in wins if not result["playerFirst"]]
        if first_wins:
            selected["firstWin"] = max(first_wins, key=lambda result: result["decisions"])
        if second_wins:
            selected["secondWin"] = max(second_wins, key=lambda result: result["decisions"])
    if losses:
        selected["typicalLoss"] = sorted(losses, key=lambda result: (result["globalTurn"], result["decisions"]))[len(losses) // 2]
    blind_rng = random.Random(20_260_816)
    for index, result in enumerate(blind_rng.sample(results, min(3, len(results))), start=1):
        selected[f"randomAudit{index}"] = result
    return selected


def record_replay(
    repo: Path,
    deck: str,
    player_model: CandidateActorCritic,
    destruction_model: CandidateActorCritic,
    metadata: EncodingMetadata,
    seed: int,
    player_first: bool,
    policy_note: str,
) -> dict[str, Any]:
    device = torch.device("cpu")
    player_model = player_model.to(device)
    destruction_model = destruction_model.to(device)
    with TrainingBridge(repo) as bridge:
        initialized = bridge.request({
            "cmd": "init", "envs": 1, "deck": deck, "baseSeed": seed,
            "fixedFirst": player_first, "record": True, "selfPlay": True,
        })
        observations = initialized["observations"]
        for _ in range(500):
            model = player_model if observations[0]["actor"] == "player" else destruction_model
            actions, values, probabilities = policy_actions(model, observations, metadata, device)
            response = bridge.request({
                "cmd": "step", "actions": actions,
                "audits": [{"value": values[0], "probabilities": probabilities[0], "note": policy_note}],
            })
            item = response["items"][0]
            if item["done"]:
                if item.get("truncated"):
                    raise RuntimeError(f"replay seed {seed} was truncated")
                return item["replay"]
            observations = [item["observation"]]
    raise RuntimeError(f"replay seed {seed} did not finish")


def main(args: argparse.Namespace) -> None:
    repo = Path(__file__).resolve().parents[1]
    league = Path(args.league)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    player_path = Path(args.player_checkpoint) if args.player_checkpoint else league / args.deck / "current.pt"
    destruction_path = Path(args.destruction_checkpoint) if args.destruction_checkpoint else league / "destruction" / "current.pt"
    player_model, player_saved, metadata = load_policy(player_path, args.deck, device)
    destruction_model, destruction_saved, destruction_metadata = load_policy(destruction_path, "destruction", device)
    if metadata != destruction_metadata or player_saved["card_ids"] != destruction_saved["card_ids"]:
        raise RuntimeError("player and destruction checkpoints use different encodings")

    first_games = (args.games + 1) // 2
    second_games = args.games // 2
    diagnostics = {"player": empty_diagnostics(), "ai": empty_diagnostics()}
    results = evaluate_order(
        repo, args.deck, player_model, destruction_model, metadata, device,
        first_games, args.envs, args.base_seed, True, diagnostics,
    )
    results += evaluate_order(
        repo, args.deck, player_model, destruction_model, metadata, device,
        second_games, args.envs, args.base_seed + 50_000_000, False, diagnostics,
    )
    wins = sum(result["winner"] == "player" for result in results)
    losses = sum(result["winner"] == "ai" for result in results)
    first_results = results[:first_games]
    second_results = results[first_games:]
    summary = {
        "deck": args.deck,
        "engineVersion": player_saved["engine_version"],
        "deterministic": True,
        "playerCycle": player_saved["cycle"],
        "destructionCycle": destruction_saved["cycle"],
        "games": len(results),
        "wins": wins,
        "losses": losses,
        "draws": len(results) - wins - losses,
        "winRate": wins / max(1, wins + losses),
        "wilson95": wilson_interval(wins, max(1, wins + losses)),
        "firstGames": len(first_results),
        "firstWinRate": sum(result["winner"] == "player" for result in first_results) / max(1, len(first_results)),
        "secondGames": len(second_results),
        "secondWinRate": sum(result["winner"] == "player" for result in second_results) / max(1, len(second_results)),
        "averageGlobalTurn": statistics.fmean(result["globalTurn"] for result in results),
        "averageDecisions": statistics.fmean(result["decisions"] for result in results),
        "truncated": sum(bool(result.get("truncated")) for result in results),
        "policyDiagnostics": finalize_diagnostics(diagnostics),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    if args.no_replays:
        return

    selected = choose_replays(results)
    policy_note = f"league-c{player_saved['cycle']}-{args.deck}-vs-c{destruction_saved['cycle']}-destruction"
    replays = {
        name: record_replay(
            repo, args.deck, player_model, destruction_model, metadata,
            result["seed"], bool(result["playerFirst"]), policy_note,
        )
        for name, result in selected.items()
    }
    output = Path(args.output) if args.output else repo / "match-logs" / f"{args.deck}-league-replays.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "format": "shadowverse-pt-training-replay-bundle",
        "replayVersion": 1,
        "policy": policy_note,
        "summary": summary,
        "selectedGames": selected,
        "replays": replays,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    evaluation_path = league / f"evaluation-{args.deck}-cycle-{player_saved['cycle']}.json"
    evaluation_path.write_text(json.dumps({"summary": summary, "selectedGames": selected}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"replays={output}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate two league policies with exact turn-order balance and export audited replays")
    parser.add_argument("--league", required=True)
    parser.add_argument("--deck", choices=("fairy", "levin"), required=True)
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--envs", type=int, default=32)
    parser.add_argument("--base-seed", type=int, default=900_000_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--player-checkpoint")
    parser.add_argument("--destruction-checkpoint")
    parser.add_argument("--no-replays", action="store_true")
    parser.add_argument("--output")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
