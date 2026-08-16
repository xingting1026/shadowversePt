from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from bridge import TrainingBridge
from model import CandidateActorCritic, EncodingMetadata, load_model_state_compatible, metadata_from_checkpoint, tensorize


def load_policy(path: Path, device: torch.device) -> tuple[CandidateActorCritic, dict[str, Any], EncodingMetadata]:
    saved = torch.load(path, map_location=device, weights_only=False)
    metadata = metadata_from_checkpoint(saved["metadata"])
    model = CandidateActorCritic(metadata).to(device)
    load_model_state_compatible(model, saved["model"])
    model.eval()
    return model, saved, metadata


def deterministic_actions(
    models: dict[str, CandidateActorCritic],
    observations: Sequence[dict[str, Any]],
    metadata: EncodingMetadata,
    device: torch.device,
) -> list[int]:
    actions = [0] * len(observations)
    for actor in ("player", "ai"):
        rows = [index for index, observation in enumerate(observations) if observation["actor"] == actor]
        if not rows:
            continue
        with torch.no_grad():
            logits, _ = models[actor](tensorize([observations[index] for index in rows], metadata, device))
        selected = logits.argmax(dim=-1).cpu().tolist()
        for row, action in zip(rows, selected):
            actions[row] = action
    return actions


def mulligan_index(observation: dict[str, Any], redraw: bool) -> int:
    for index, action in enumerate(observation["actions"]):
        if int(action["kind"]) == 0 and bool(action["numbers"][12] > 0.5) == redraw:
            return index
    raise RuntimeError(f"missing {'redraw' if redraw else 'keep'} action")


def terminal_score(result: dict[str, Any], target_role: str) -> float:
    target_won = result["winner"] == target_role
    opponent_won = result["winner"] in ("player", "ai") and not target_won
    outcome = 1.0 if target_won else (-1.0 if opponent_won else 0.0)
    own_hp = result["playerHp"] if target_role == "player" else result["aiHp"]
    opponent_hp = result["aiHp"] if target_role == "player" else result["playerHp"]
    tempo = -result["globalTurn"] if target_won else result["globalTurn"]
    return outcome * 100.0 + (own_hp - opponent_hp) * 0.1 + tempo * 0.001


def collect_order(
    repo: Path,
    deck: str,
    target_role: str,
    models: dict[str, CandidateActorCritic],
    metadata: EncodingMetadata,
    device: torch.device,
    games: int,
    envs: int,
    base_seed: int,
    player_first: bool,
) -> tuple[list[dict[str, Any]], list[int], dict[str, Any]]:
    env_count = min(envs, games)
    target_seeds = {base_seed + index for index in range(env_count)}
    # Each slot is followed only through its initial episode. Additional reset episodes
    # may be stepped while a slower paired branch finishes, but are excluded by seed.
    keep_results: dict[int, dict[str, Any]] = {}
    redraw_results: dict[int, dict[str, Any]] = {}
    samples: dict[int, dict[str, Any]] = {}
    keep_forced = [False] * env_count
    redraw_forced = [False] * env_count

    with TrainingBridge(repo) as keep_bridge, TrainingBridge(repo) as redraw_bridge:
        request = {
            "cmd": "init", "envs": env_count, "deck": deck, "baseSeed": base_seed,
            "fixedFirst": player_first, "record": False, "selfPlay": True,
        }
        keep_observations = keep_bridge.request(request)["observations"]
        redraw_observations = redraw_bridge.request(request)["observations"]
        for _ in range(500):
            if len(keep_results) < games:
                actions = deterministic_actions(models, keep_observations, metadata, device)
                for row, observation in enumerate(keep_observations):
                    if not keep_forced[row] and observation["actor"] == target_role and any(int(action["kind"]) == 0 for action in observation["actions"]):
                        samples[row] = observation
                        actions[row] = mulligan_index(observation, False)
                        keep_forced[row] = True
                response = keep_bridge.request({"cmd": "step", "actions": actions})
                for item in response["items"]:
                    result = item.get("result")
                    if result and result["seed"] in target_seeds:
                        if item.get("truncated"):
                            raise RuntimeError(f"keep branch seed {result['seed']} was truncated")
                        keep_results[result["seed"]] = result
                keep_observations = [item["observation"] for item in response["items"]]

            if len(redraw_results) < games:
                actions = deterministic_actions(models, redraw_observations, metadata, device)
                for row, observation in enumerate(redraw_observations):
                    if not redraw_forced[row] and observation["actor"] == target_role and any(int(action["kind"]) == 0 for action in observation["actions"]):
                        actions[row] = mulligan_index(observation, True)
                        redraw_forced[row] = True
                response = redraw_bridge.request({"cmd": "step", "actions": actions})
                for item in response["items"]:
                    result = item.get("result")
                    if result and result["seed"] in target_seeds:
                        if item.get("truncated"):
                            raise RuntimeError(f"redraw branch seed {result['seed']} was truncated")
                        redraw_results[result["seed"]] = result
                redraw_observations = [item["observation"] for item in response["items"]]

            if len(keep_results) == games and len(redraw_results) == games:
                break
        else:
            raise RuntimeError("paired mulligan rollouts did not finish")

    observations: list[dict[str, Any]] = []
    labels: list[int] = []
    outcome_flips = 0
    redraw_better = 0
    score_deltas: list[float] = []
    for slot, seed in enumerate(sorted(target_seeds)):
        keep_result = keep_results[seed]
        redraw_result = redraw_results[seed]
        keep_score = terminal_score(keep_result, target_role)
        redraw_score = terminal_score(redraw_result, target_role)
        redraw = redraw_score > keep_score
        observations.append(samples[slot])
        labels.append(mulligan_index(samples[slot], redraw))
        redraw_better += int(redraw)
        outcome_flips += int(keep_result["winner"] != redraw_result["winner"])
        score_deltas.append(redraw_score - keep_score)
    return observations, labels, {
        "deck": deck,
        "targetRole": target_role,
        "playerFirst": player_first,
        "games": games,
        "redrawBetterRate": redraw_better / games,
        "winnerFlipRate": outcome_flips / games,
        "meanRedrawScoreDelta": float(np.mean(score_deltas)),
    }


def tune_policy(
    model: CandidateActorCritic,
    observations: list[dict[str, Any]],
    labels: list[int],
    metadata: EncodingMetadata,
    device: torch.device,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    order = np.arange(len(observations))
    rng = np.random.default_rng(seed)
    rng.shuffle(order)
    validation_size = max(1, len(order) // 5)
    validation = order[:validation_size]
    training = order[validation_size:]
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.mulligan_head.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.mulligan_head.parameters(), lr=learning_rate, eps=1e-5, weight_decay=1e-5)
    losses: list[float] = []
    oracle_redraw_rate = sum(
        observations[index]["actions"][labels[index]]["numbers"][12] > 0.5
        for index in order
    ) / len(order)
    best_state = copy.deepcopy(model.mulligan_head.state_dict())
    best_score = (-1.0, float("-inf"))
    best_metrics = {"epoch": 0, "accuracy": 0.0, "predictedRedrawRate": 0.0}

    def validation_metrics() -> tuple[float, float]:
        model.eval()
        with torch.no_grad():
            logits, _ = model(tensorize([observations[index] for index in validation], metadata, device))
        predicted = logits.argmax(dim=-1).cpu().tolist()
        expected = [labels[index] for index in validation]
        accuracy = sum(left == right for left, right in zip(predicted, expected)) / len(expected)
        redraw_rate = sum(
            observations[index]["actions"][prediction]["numbers"][12] > 0.5
            for index, prediction in zip(validation, predicted)
        ) / len(predicted)
        return accuracy, redraw_rate

    initial_accuracy, initial_redraw_rate = validation_metrics()
    best_score = (initial_accuracy, -abs(initial_redraw_rate - oracle_redraw_rate))
    best_metrics = {"epoch": 0, "accuracy": initial_accuracy, "predictedRedrawRate": initial_redraw_rate}
    model.train()
    for epoch in range(1, epochs + 1):
        rng.shuffle(training)
        for start in range(0, len(training), batch_size):
            indices = training[start:start + batch_size]
            batch = tensorize([observations[index] for index in indices], metadata, device)
            target = torch.as_tensor([labels[index] for index in indices], device=device)
            logits, _ = model(batch)
            loss = torch.nn.functional.cross_entropy(logits, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        accuracy, redraw_rate = validation_metrics()
        score = (accuracy, -abs(redraw_rate - oracle_redraw_rate))
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.mulligan_head.state_dict())
            best_metrics = {"epoch": epoch, "accuracy": accuracy, "predictedRedrawRate": redraw_rate}
        model.train()
    model.mulligan_head.load_state_dict(best_state)
    model.eval()
    return {
        "samples": len(observations),
        "trainingSamples": len(training),
        "validationSamples": len(validation),
        "bestEpoch": best_metrics["epoch"],
        "validationAccuracy": best_metrics["accuracy"],
        "validationPredictedRedrawRate": best_metrics["predictedRedrawRate"],
        "finalMeanLoss": float(np.mean(losses[-max(1, len(training) // batch_size):])),
        "optimizer": optimizer,
    }


def save_policy(path: Path, saved: dict[str, Any], model: CandidateActorCritic, tuning: dict[str, Any]) -> None:
    output = copy.copy(saved)
    output["model"] = model.state_dict()
    # The architecture gained a residual parameter group, so a resumed PPO run
    # must start fresh optimizer moments while retaining all policy weights.
    output["optimizer"] = None
    output["cycle"] = 13
    output["mulligan_tuning"] = tuning
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(output, temporary)
    temporary.replace(path)


def main(args: argparse.Namespace) -> None:
    repo = Path(__file__).resolve().parents[1]
    league = Path(args.league)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    models: dict[str, CandidateActorCritic] = {}
    saved: dict[str, dict[str, Any]] = {}
    metadata: EncodingMetadata | None = None
    for policy in ("fairy", "levin", "destruction"):
        model, checkpoint, policy_metadata = load_policy(league / policy / "cycle-012.pt", device)
        if int(checkpoint.get("cycle", -1)) != 12:
            raise RuntimeError(f"expected cycle 12 {policy} checkpoint, got {checkpoint.get('cycle')}")
        if metadata is not None and metadata != policy_metadata:
            raise RuntimeError("league policies use different encodings")
        metadata = policy_metadata
        models[policy] = model
        saved[policy] = checkpoint
    assert metadata is not None

    samples: dict[str, list[dict[str, Any]]] = {policy: [] for policy in models}
    labels: dict[str, list[int]] = {policy: [] for policy in models}
    rollout_reports: list[dict[str, Any]] = []
    jobs = [
        ("fairy", "fairy", "player", args.base_seed),
        ("levin", "levin", "player", args.base_seed + 10_000_000),
        ("destruction", "fairy", "ai", args.base_seed + 20_000_000),
        ("destruction", "levin", "ai", args.base_seed + 30_000_000),
    ]
    selected_policies = set(args.policies)
    for policy, deck, target_role, seed in jobs:
        if policy not in selected_policies:
            continue
        job_games = args.games_per_order if policy != "destruction" else max(1, args.games_per_order // 2)
        actor_models = {"player": models[deck], "ai": models["destruction"]}
        for order_index, player_first in enumerate((True, False)):
            chunk_reports: list[dict[str, Any]] = []
            for start in range(0, job_games, args.envs):
                chunk_games = min(args.envs, job_games - start)
                observations, action_labels, chunk_report = collect_order(
                    repo, deck, target_role, actor_models, metadata, device,
                    chunk_games, chunk_games, seed + order_index * 5_000_000 + start, player_first,
                )
                samples[policy].extend(observations)
                labels[policy].extend(action_labels)
                chunk_reports.append(chunk_report)
            report = {
                "deck": deck,
                "targetRole": target_role,
                "playerFirst": player_first,
                "games": job_games,
                "redrawBetterRate": sum(item["redrawBetterRate"] * item["games"] for item in chunk_reports) / job_games,
                "winnerFlipRate": sum(item["winnerFlipRate"] * item["games"] for item in chunk_reports) / job_games,
                "meanRedrawScoreDelta": sum(item["meanRedrawScoreDelta"] * item["games"] for item in chunk_reports) / job_games,
            }
            rollout_reports.append(report)
            print(json.dumps({"stage": "rollout", **report}), flush=True)

    if args.collect_only:
        print(json.dumps({"stage": "collect-only", "samples": {policy: len(values) for policy, values in samples.items()}}), flush=True)
        return

    tuning_reports: dict[str, Any] = {}
    for index, policy in enumerate(("fairy", "levin", "destruction")):
        if policy not in selected_policies:
            continue
        report = tune_policy(
            models[policy], samples[policy], labels[policy], metadata, device,
            args.learning_rate, args.epochs, args.batch_size, args.seed + index,
        )
        report.pop("optimizer")
        report["pairedOracleRedrawRate"] = sum(
            observation["actions"][label]["numbers"][12] > 0.5
            for observation, label in zip(samples[policy], labels[policy])
        ) / len(labels[policy])
        tuning_reports[policy] = report
        tuning = {"method": "paired-counterfactual-rollout", "rollouts": rollout_reports, "report": report}
        cycle_path = league / policy / "cycle-013.pt"
        current_path = league / policy / "current.pt"
        save_policy(cycle_path, saved[policy], models[policy], tuning)
        save_policy(current_path, saved[policy], models[policy], tuning)
        print(json.dumps({"stage": "tuned", "policy": policy, **report}), flush=True)
    report_suffix = "-" + "-".join(args.policies) if len(args.policies) < 3 else ""
    (league / f"mulligan-tuning{report_suffix}.json").write_text(json.dumps({
        "format": "shadowverse-pt-paired-mulligan-tuning",
        "sourceCycle": 12,
        "outputCycle": 13,
        "rollouts": rollout_reports,
        "policies": tuning_reports,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune all three league mulligan policies with paired keep/redraw rollouts")
    parser.add_argument("--league", required=True)
    parser.add_argument("--games-per-order", type=int, default=128)
    parser.add_argument("--envs", type=int, default=32)
    parser.add_argument("--base-seed", type=int, default=920_000_000)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=20_260_816)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--policies", nargs="+", choices=("fairy", "levin", "destruction"), default=["fairy", "levin", "destruction"])
    parser.add_argument("--collect-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
