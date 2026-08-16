"""從零訓練一對「玩家牌組 vs 玩家牌組」的 pair league（例如雷維翁 vs 妖精）。

兩側都是神經網路 policy，交替 best-response：一方更新時對手凍結，
每局 50% 對最新對手、50% 均勻抽歷史 checkpoint。收斂後用未見過的
seed 做決定論盲測（先後攻各半），輸出 Wilson CI 勝率。

用法：
  python training/train_pair.py --player-deck levin --ai-deck fairy \
      --cycles 10 --phase-decisions 20000 --envs 32 \
      --output training-output/pair-levin-fairy
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from bridge import TrainingBridge
from model import CandidateActorCritic, EncodingMetadata, checkpoint_metadata, tensorize
from train_league import (
    Transition,
    act_batch,
    behavior_clone,
    clone_frozen_model,
    cpu_state_dict,
    heuristic_action_index,
    ppo_update,
    sample_pool_index,
)

ROLES = ("player", "ai")


def init_request(args: argparse.Namespace, envs: int, base_seed: int, fixed_first: bool | None = None) -> dict[str, Any]:
    request: dict[str, Any] = {
        "cmd": "init",
        "envs": envs,
        "deck": args.player_deck,
        "aiDeck": args.ai_deck,
        "baseSeed": base_seed,
        "record": False,
        "selfPlay": True,
    }
    if fixed_first is not None:
        request["fixedFirst"] = fixed_first
    return request


def collect_pair_behavior(
    repo: Path,
    args: argparse.Namespace,
    role: str,
    decisions: int,
    base_seed: int,
) -> tuple[list[dict[str, Any]], list[int], dict[str, Any]]:
    observations_out: list[dict[str, Any]] = []
    labels: list[int] = []
    with TrainingBridge(repo) as bridge:
        initialized = bridge.request(init_request(args, args.envs, base_seed))
        observations = initialized["observations"]
        while len(labels) < decisions:
            actions = [heuristic_action_index(observation) for observation in observations]
            for observation, action in zip(observations, actions):
                if observation["actor"] == role and len(labels) < decisions:
                    observations_out.append(observation)
                    labels.append(action)
            response = bridge.request({"cmd": "step", "actions": actions})
            observations = [item["observation"] for item in response["items"]]
    return observations_out, labels, initialized


def train_pair_phase(
    repo: Path,
    args: argparse.Namespace,
    target_role: str,
    learner: CandidateActorCritic,
    optimizer: torch.optim.Optimizer,
    opponent_pool: Sequence[dict[str, torch.Tensor]],
    metadata: EncodingMetadata,
    device: torch.device,
    base_seed: int,
) -> dict[str, Any]:
    frozen_opponents = [clone_frozen_model(state, metadata, device) for state in opponent_pool]
    completed_episodes: list[list[Transition]] = []
    live_episodes: list[list[Transition]] = [[] for _ in range(args.envs)]
    opponent_assignments = [sample_pool_index(len(frozen_opponents)) for _ in range(args.envs)]
    completed_transitions = 0
    wins = losses = draws = games = 0
    started = time.perf_counter()
    learner.eval()
    with TrainingBridge(repo) as bridge:
        initialized = bridge.request(init_request(args, args.envs, base_seed))
        observations = initialized["observations"]
        while completed_transitions < args.phase_decisions:
            action_indices = [0] * args.envs
            target_rows = [index for index, observation in enumerate(observations) if observation["actor"] == target_role]
            target_observations = [observations[index] for index in target_rows]
            actions, log_probabilities, values = act_batch(learner, target_observations, metadata, device)
            for row, action, log_probability, value in zip(target_rows, actions, log_probabilities, values):
                action_indices[row] = action
                live_episodes[row].append(Transition(observations[row], action, log_probability, value))

            opponent_rows = [index for index, observation in enumerate(observations) if observation["actor"] != target_role]
            grouped: dict[int, list[int]] = {}
            for row in opponent_rows:
                grouped.setdefault(opponent_assignments[row], []).append(row)
            for pool_index, rows in grouped.items():
                selected, _, _ = act_batch(frozen_opponents[pool_index], [observations[row] for row in rows], metadata, device)
                for row, action in zip(rows, selected):
                    action_indices[row] = action

            response = bridge.request({"cmd": "step", "actions": action_indices})
            for row, item in enumerate(response["items"]):
                if not item["done"]:
                    continue
                winner = item["result"].get("winner")
                won = winner == target_role
                lost = winner in ("player", "ai") and not won
                reward = 1.0 if won else -1.0 if lost else 0.0
                if live_episodes[row]:
                    live_episodes[row][-1].reward = reward
                    completed_episodes.append(live_episodes[row])
                    completed_transitions += len(live_episodes[row])
                live_episodes[row] = []
                wins += int(won)
                losses += int(lost)
                draws += int(not won and not lost)
                games += 1
                opponent_assignments[row] = sample_pool_index(len(frozen_opponents))
            observations = [item["observation"] for item in response["items"]]

    metrics = ppo_update(learner, optimizer, completed_episodes, metadata, device, args)
    elapsed = time.perf_counter() - started
    del frozen_opponents
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "targetRole": target_role,
        "games": games,
        "decisions": completed_transitions,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "winRate": wins / max(1, wins + losses),
        "opponentPoolSize": len(opponent_pool),
        "seconds": round(elapsed, 3),
        **metrics,
    }


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return (0.0, 1.0)
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = (z / denominator) * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def evaluate_pair(
    repo: Path,
    args: argparse.Namespace,
    models: dict[str, CandidateActorCritic],
    metadata: EncodingMetadata,
    device: torch.device,
) -> dict[str, Any]:
    per_side = args.eval_games // 2
    summary = {"playerWins": 0, "aiWins": 0, "draws": 0, "truncated": 0, "turns": [], "byFirst": {}}
    for fixed_first in (True, False):
        wins = losses = draws = truncated = games = 0
        envs = min(args.envs, per_side)
        base_seed = args.eval_seed + (0 if fixed_first else 50_000_000)
        with TrainingBridge(repo) as bridge:
            initialized = bridge.request(init_request(args, envs, base_seed, fixed_first))
            observations = initialized["observations"]
            while games < per_side:
                actions = [0] * envs
                for role in ROLES:
                    rows = [index for index, observation in enumerate(observations) if observation["actor"] == role]
                    if not rows:
                        continue
                    selected, _, _ = act_batch(models[role], [observations[index] for index in rows], metadata, device, deterministic=True)
                    for row, action in zip(rows, selected):
                        actions[row] = action
                response = bridge.request({"cmd": "step", "actions": actions})
                for item in response["items"]:
                    result = item.get("result")
                    if not result or games >= per_side:
                        continue
                    games += 1
                    summary["turns"].append(result["globalTurn"])
                    if item.get("truncated"):
                        truncated += 1
                    elif result.get("winner") == "player":
                        wins += 1
                    elif result.get("winner") == "ai":
                        losses += 1
                    else:
                        draws += 1
                observations = [item["observation"] for item in response["items"]]
        summary["playerWins"] += wins
        summary["aiWins"] += losses
        summary["draws"] += draws
        summary["truncated"] += truncated
        summary["byFirst"]["playerFirst" if fixed_first else "aiFirst"] = {
            "games": games, "playerWins": wins, "aiWins": losses, "draws": draws,
        }
    decided = summary["playerWins"] + summary["aiWins"]
    low, high = wilson(summary["playerWins"], max(1, decided))
    summary["playerWinRate"] = summary["playerWins"] / max(1, decided)
    summary["wilson95"] = [round(low, 4), round(high, 4)]
    summary["avgTurns"] = round(float(np.mean(summary["turns"])), 2) if summary["turns"] else 0
    del summary["turns"]
    return summary


def save_pair_policy(path: Path, role: str, cycle: int, model: CandidateActorCritic, optimizer: torch.optim.Optimizer,
                     metadata: EncodingMetadata, initialized: dict[str, Any], args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metadata": checkpoint_metadata(metadata),
        "card_ids": initialized["metadata"]["cardIds"],
        "engine_version": initialized["engineVersion"],
        "policy": f"{args.player_deck}-vs-{args.ai_deck}:{role}",
        "cycle": cycle,
        "self_play": True,
        "config": vars(args),
    }, temporary)
    temporary.replace(path)


def train(args: argparse.Namespace) -> None:
    repo = Path(__file__).resolve().parents[1]
    output = Path(args.output) if args.output else repo / "training-output" / f"pair-{args.player_deck}-{args.ai_deck}"
    output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    with TrainingBridge(repo) as bridge:
        initialized = bridge.request(init_request(args, 1, args.base_seed))
    metadata = EncodingMetadata.from_server(initialized["metadata"])
    models = {role: CandidateActorCritic(metadata).to(device) for role in ROLES}
    optimizers = {
        role: torch.optim.AdamW(models[role].parameters(), lr=args.learning_rate, eps=1e-5, weight_decay=1e-5)
        for role in ROLES
    }
    metrics_path = output / "pair-metrics.jsonl"
    started = time.perf_counter()

    for role, seed_offset in (("player", 1_000_000), ("ai", 2_000_000)):
        observations, labels, _ = collect_pair_behavior(repo, args, role, args.warmup_decisions, args.base_seed + seed_offset)
        loss = behavior_clone(models[role], optimizers[role], observations, labels, metadata, device, args.warmup_epochs, args.minibatch_size)
        print(json.dumps({"stage": "warmup", "role": role, "decisions": len(labels), "loss": round(loss, 4)}), flush=True)

    pools: dict[str, list[dict[str, torch.Tensor]]] = {role: [cpu_state_dict(models[role])] for role in ROLES}

    for cycle in range(1, args.cycles + 1):
        phase_seed = args.base_seed + cycle * 10_000_000
        phases = []
        phases.append(train_pair_phase(repo, args, "player", models["player"], optimizers["player"], pools["ai"], metadata, device, phase_seed + 100_000))
        pools["player"].append(cpu_state_dict(models["player"]))
        phases.append(train_pair_phase(repo, args, "ai", models["ai"], optimizers["ai"], pools["player"], metadata, device, phase_seed + 200_000))
        pools["ai"].append(cpu_state_dict(models["ai"]))
        for role in ROLES:
            save_pair_policy(output / role / f"cycle-{cycle:03d}.pt", role, cycle, models[role], optimizers[role], metadata, initialized, args)
            save_pair_policy(output / role / "current.pt", role, cycle, models[role], optimizers[role], metadata, initialized, args)
        record = {"cycle": cycle, "phases": phases, "elapsed": round(time.perf_counter() - started, 1)}
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)

    evaluation = evaluate_pair(repo, args, models, metadata, device)
    result = {
        "pair": f"{args.player_deck}(player) vs {args.ai_deck}(ai)",
        "cycles": args.cycles,
        "phaseDecisions": args.phase_decisions,
        "totalSeconds": round(time.perf_counter() - started, 1),
        "evaluation": evaluation,
    }
    (output / "pair-result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="From-scratch pair training: player deck vs ai deck, both neural policies")
    parser.add_argument("--player-deck", default="levin")
    parser.add_argument("--ai-deck", default="fairy")
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--phase-decisions", type=int, default=20_000)
    parser.add_argument("--warmup-decisions", type=int, default=8_000)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--envs", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--entropy-coefficient", type=float, default=0.02)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.7)
    parser.add_argument("--eval-games", type=int, default=1000)
    parser.add_argument("--eval-seed", type=int, default=990_000_000)
    parser.add_argument("--base-seed", type=int, default=260_816_000)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
