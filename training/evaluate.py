from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.distributions import Categorical

from bridge import TrainingBridge
from model import CandidateActorCritic, EncodingMetadata, load_model_state_compatible, metadata_from_checkpoint, tensorize


def policy_actions(
    model: CandidateActorCritic,
    observations: list[dict[str, Any]],
    metadata: EncodingMetadata,
    device: torch.device,
) -> tuple[list[int], list[float], list[list[float]]]:
    with torch.no_grad():
        logits, values = model(tensorize(observations, metadata, device))
        distribution = Categorical(logits=logits)
        probabilities = distribution.probs
        selected = torch.argmax(logits, dim=-1)
    action_indices = selected.cpu().tolist()
    value_list = values.cpu().tolist()
    probability_lists = [
        probabilities[row, : len(observation["actions"])].cpu().tolist()
        for row, observation in enumerate(observations)
    ]
    return action_indices, value_list, probability_lists


def evaluate_games(
    bridge: TrainingBridge,
    model: CandidateActorCritic,
    metadata: EncodingMetadata,
    device: torch.device,
    deck: str,
    games: int,
    envs: int,
    base_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    initialized = bridge.request({
        "cmd": "init",
        "envs": envs,
        "deck": deck,
        "baseSeed": base_seed,
        "record": False,
    })
    observations = initialized["observations"]
    target_seeds = set(range(base_seed, base_seed + games))
    results: dict[int, dict[str, Any]] = {}
    action_kinds = [0] * 6
    mulligans = 0
    mulligan_redraws = 0
    mulligan_redraw_probability = 0.0
    selected_probability = 0.0
    decisions = 0
    while len(results) < games:
        actions, _, probabilities = policy_actions(model, observations, metadata, device)
        for row, action_index in enumerate(actions):
            action = observations[row]["actions"][action_index]
            kind = int(action["kind"])
            action_kinds[kind] += 1
            decisions += 1
            selected_probability += probabilities[row][action_index]
            if kind == 0:
                mulligans += 1
                mulligan_redraws += int(action["numbers"][12] > 0.5)
                if len(probabilities[row]) >= 2:
                    mulligan_redraw_probability += probabilities[row][1]
        response = bridge.request({"cmd": "step", "actions": actions})
        for item in response["items"]:
            result = item.get("result")
            if result and result["seed"] in target_seeds:
                results[result["seed"]] = result
        observations = [item["observation"] for item in response["items"]]
    diagnostics = {
        "decisionsObserved": decisions,
        "actionKindCounts": {
            name: action_kinds[index]
            for index, name in enumerate(["mulligan", "choice", "play", "activate", "evolve", "end"])
        },
        "meanSelectedProbability": selected_probability / max(1, decisions),
        "mulligansObserved": mulligans,
        "mulliganRedrawRate": mulligan_redraws / max(1, mulligans),
        "meanMulliganRedrawProbability": mulligan_redraw_probability / max(1, mulligans),
    }
    return [results[seed] for seed in sorted(results)], diagnostics


def record_replay(
    repo: Path,
    model: CandidateActorCritic,
    metadata: EncodingMetadata,
    device: torch.device,
    deck: str,
    seed: int,
    player_first: bool,
    policy_name: str,
) -> dict[str, Any]:
    with TrainingBridge(repo) as bridge:
        initialized = bridge.request({
            "cmd": "init",
            "envs": 1,
            "deck": deck,
            "baseSeed": seed,
            "record": True,
            "fixedFirst": player_first,
        })
        observations = initialized["observations"]
        for _ in range(500):
            actions, values, probabilities = policy_actions(model, observations, metadata, device)
            response = bridge.request({
                "cmd": "step",
                "actions": actions,
                "audits": [{"value": values[0], "probabilities": probabilities[0], "note": policy_name}],
            })
            item = response["items"][0]
            if item["done"]:
                if item.get("truncated"):
                    raise RuntimeError(f"replay seed {seed} was truncated")
                return item["replay"]
            observations = [item["observation"]]
    raise RuntimeError(f"replay seed {seed} did not finish")


def choose_replay_seeds(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
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
        ordered = sorted(losses, key=lambda result: (result["globalTurn"], result["decisions"]))
        selected["typicalLoss"] = ordered[len(ordered) // 2]
    # 盲抽樣本使用固定 RNG 種子，在看勝負前即由完整評估集合等機率抽出。
    # 它們和精選局並存，避免回放只呈現事後挑出的漂亮操作。
    audit_rng = random.Random(20_260_816)
    for index, result in enumerate(audit_rng.sample(results, min(3, len(results))), start=1):
        selected[f"randomAudit{index}"] = result
    return selected


def summarize(
    results: list[dict[str, Any]],
    checkpoint: dict[str, Any],
    base_seed: int,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    wins = [result for result in results if result["winner"] == "player"]
    first = [result for result in results if result["playerFirst"]]
    second = [result for result in results if not result["playerFirst"]]
    return {
        "deck": checkpoint["deck"],
        "checkpointDecisions": checkpoint["decisions"],
        "games": len(results),
        "seedRange": [base_seed, base_seed + len(results) - 1],
        "wins": len(wins),
        "losses": sum(result["winner"] == "ai" for result in results),
        "draws": sum(result["winner"] == "draw" for result in results),
        "winRate": len(wins) / max(1, len(results)),
        "firstWinRate": sum(result["winner"] == "player" for result in first) / max(1, len(first)),
        "secondWinRate": sum(result["winner"] == "player" for result in second) / max(1, len(second)),
        "averageGlobalTurn": statistics.fmean(result["globalTurn"] for result in results),
        "averageDecisions": statistics.fmean(result["decisions"] for result in results),
        "truncated": sum(bool(result.get("truncated")) for result in results),
        "policyDiagnostics": diagnostics,
    }


def main(args: argparse.Namespace) -> None:
    repo = Path(__file__).resolve().parents[1]
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else repo / "training-output" / args.deck / "checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["deck"] != args.deck:
        raise RuntimeError(f"checkpoint is for {checkpoint['deck']}, not {args.deck}")
    metadata = metadata_from_checkpoint(checkpoint["metadata"])
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = CandidateActorCritic(metadata).to(device)
    load_model_state_compatible(model, checkpoint["model"])
    model.eval()
    with TrainingBridge(repo) as bridge:
        initialized = bridge.request({"cmd": "init", "envs": 1, "deck": args.deck, "baseSeed": args.base_seed})
        if initialized["metadata"]["cardIds"] != checkpoint["card_ids"]:
            raise RuntimeError("checkpoint card vocabulary does not match the current engine")
        if checkpoint.get("engine_version") != initialized["engineVersion"]:
            raise RuntimeError("checkpoint engine version does not match the current engine")
        results, diagnostics = evaluate_games(bridge, model, metadata, device, args.deck, args.games, args.envs, args.base_seed)
    summary = summarize(results, checkpoint, args.base_seed, diagnostics)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    evaluation_name = "evaluation.json" if checkpoint_path.name == "checkpoint.pt" else f"evaluation-{checkpoint_path.stem}.json"
    evaluation_path = checkpoint_path.parent / evaluation_name
    if args.no_replays:
        evaluation_path.write_text(json.dumps({"summary": summary}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return

    policy_name = f"masked-ppo-{args.deck}-{checkpoint['decisions']}"
    selected = choose_replay_seeds(results)
    # 單場回放每次只有一個 observation；小 MLP 在 CPU 上可避免數十次 CUDA kernel 啟動延遲。
    replay_device = torch.device("cpu")
    model = model.to(replay_device)
    replays = {
        key: record_replay(repo, model, metadata, replay_device, args.deck, result["seed"], result["playerFirst"], policy_name)
        for key, result in selected.items()
    }
    output = Path(args.output) if args.output else repo / "match-logs" / f"{args.deck}-ppo-replays.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "format": "shadowverse-pt-training-replay-bundle",
        "replayVersion": 1,
        "policy": policy_name,
        "summary": summary,
        "selectedGames": selected,
        "replays": replays,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    evaluation_path.write_text(json.dumps({"summary": summary, "selectedGames": selected}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"replays={output}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained policy on unseen seeds and export audited replays")
    parser.add_argument("--deck", choices=["fairy", "levin"], required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--envs", type=int, default=32)
    parser.add_argument("--base-seed", type=int, default=50_000_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output")
    parser.add_argument("--no-replays", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
