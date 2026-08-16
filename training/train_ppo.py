from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import deque
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from bridge import TrainingBridge
from model import CandidateActorCritic, EncodingMetadata, checkpoint_metadata, load_model_state_compatible, tensorize


def heuristic_action_index(observation: dict[str, Any]) -> int:
    actions = observation["actions"]
    choices = [(index, action) for index, action in enumerate(actions) if action["kind"] == 1]
    if choices:
        face = next((index for index, action in choices if 1 in action["selectedSpecials"]), None)
        if face is not None:
            return face
        return max(choices, key=lambda item: (item[1]["numbers"][6], -item[0]))[0]
    plays = [(index, action) for index, action in enumerate(actions) if action["kind"] == 2]
    if plays:
        return max(plays, key=lambda item: (item[1]["numbers"][0], -item[0]))[0]
    evolve = next((index for index, action in enumerate(actions) if action["kind"] == 4), None)
    if evolve is not None:
        return evolve
    activate = next((index for index, action in enumerate(actions) if action["kind"] == 3), None)
    if activate is not None:
        return activate
    keep = next((index for index, action in enumerate(actions) if action["kind"] == 0 and action["numbers"][12] == 0), None)
    if keep is not None:
        return keep
    return next(index for index, action in enumerate(actions) if action["kind"] == 5)


def behavior_clone(
    model: CandidateActorCritic,
    optimizer: torch.optim.Optimizer,
    bridge: TrainingBridge,
    observations: list[dict[str, Any]],
    metadata: EncodingMetadata,
    device: torch.device,
    total_decisions: int,
    chunk_steps: int = 16,
) -> list[dict[str, Any]]:
    completed = 0
    started = time.perf_counter()
    while completed < total_decisions:
        samples: list[dict[str, Any]] = []
        labels: list[int] = []
        steps = min(chunk_steps, math.ceil((total_decisions - completed) / len(observations)))
        for _ in range(steps):
            action_indices: list[int] = []
            for observation in observations:
                # 貪心教師沒有可靠的留牌知識。換牌在暖啟動期隨機探索，且不拿來做
                # imitation label，避免模型被硬教成永遠保留整手。
                if observation["actions"] and observation["actions"][0]["kind"] == 0:
                    action_indices.append(random.randrange(len(observation["actions"])))
                    continue
                action_index = heuristic_action_index(observation)
                action_indices.append(action_index)
                samples.append(observation)
                labels.append(action_index)
            response = bridge.request({"cmd": "step", "actions": action_indices})
            observations = [item["observation"] for item in response["items"]]
        completed += len(samples)
        order = np.arange(len(samples))
        for _ in range(2):
            np.random.shuffle(order)
            for start in range(0, len(samples), 512):
                indices = order[start : start + 512]
                batch = tensorize([samples[index] for index in indices], metadata, device)
                target = torch.as_tensor([labels[index] for index in indices], device=device)
                logits, _ = model(batch)
                loss = F.cross_entropy(logits, target)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        if completed % max(len(observations) * chunk_steps * 4, 1) == 0 or completed >= total_decisions:
            rate = completed / max(0.001, time.perf_counter() - started)
            print(f"warmup decisions={completed}/{total_decisions} rate={rate:.0f}/s loss={loss.item():.4f}", flush=True)
    return observations


def save_checkpoint(
    path: Path,
    model: CandidateActorCritic,
    optimizer: torch.optim.Optimizer,
    metadata: EncodingMetadata,
    deck: str,
    engine_version: int,
    decisions: int,
    update: int,
    card_ids: Sequence[str],
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metadata": checkpoint_metadata(metadata),
            "card_ids": list(card_ids),
            "deck": deck,
            "engine_version": engine_version,
            "decisions": decisions,
            "update": update,
            "config": config,
        },
        temporary,
    )
    os.replace(temporary, path)


def train(args: argparse.Namespace) -> None:
    repo = Path(__file__).resolve().parents[1]
    output = Path(args.output) if args.output else repo / "training-output" / args.deck
    checkpoint_path = output / "checkpoint.pt"
    metrics_path = output / "metrics.jsonl"
    output.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device={device} deck={args.deck} envs={args.envs}", flush=True)

    with TrainingBridge(repo) as bridge:
        initialized = bridge.request({
            "cmd": "init",
            "envs": args.envs,
            "deck": args.deck,
            "baseSeed": args.base_seed,
            "record": False,
        })
        metadata = EncodingMetadata.from_server(initialized["metadata"])
        observations = initialized["observations"]
        model = CandidateActorCritic(metadata).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, eps=1e-5, weight_decay=1e-5)
        decisions = 0
        first_update = 0
        if args.resume and checkpoint_path.exists():
            saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
            if (
                saved["deck"] != args.deck
                or saved["card_ids"] != initialized["metadata"]["cardIds"]
                or saved.get("engine_version") != initialized["engineVersion"]
            ):
                raise RuntimeError("checkpoint deck/card vocabulary/engine version does not match the current engine")
            load_model_state_compatible(model, saved["model"])
            try:
                optimizer.load_state_dict(saved["optimizer"])
            except ValueError:
                pass
            decisions = int(saved["decisions"])
            first_update = int(saved["update"]) + 1
            print(f"resumed decisions={decisions} update={first_update}", flush=True)
        elif args.warmup_decisions > 0:
            observations = behavior_clone(
                model,
                optimizer,
                bridge,
                observations,
                metadata,
                device,
                args.warmup_decisions,
            )

        episode_results: deque[dict[str, Any]] = deque(maxlen=1000)
        starting_decisions = decisions
        start_time = time.perf_counter()
        update = first_update
        config = vars(args).copy()
        snapshot_interval = max(0, args.snapshot_every_decisions)
        next_snapshot = (
            ((decisions // snapshot_interval) + 1) * snapshot_interval
            if snapshot_interval else 0
        )
        while decisions < args.total_decisions:
            rollout_observations: list[list[dict[str, Any]]] = []
            rollout_actions: list[np.ndarray] = []
            rollout_log_probs: list[np.ndarray] = []
            rollout_values: list[np.ndarray] = []
            rollout_rewards: list[np.ndarray] = []
            rollout_dones: list[np.ndarray] = []

            model.eval()
            for _ in range(args.rollout_steps):
                rollout_observations.append(observations)
                batch = tensorize(observations, metadata, device)
                with torch.no_grad():
                    logits, values = model(batch)
                    distribution = Categorical(logits=logits)
                    selected = distribution.sample()
                    log_probabilities = distribution.log_prob(selected)
                action_indices = selected.cpu().numpy()
                response = bridge.request({"cmd": "step", "actions": action_indices.tolist()})
                items = response["items"]
                next_observations = [item["observation"] for item in items]
                dones = np.asarray([item["done"] for item in items], dtype=np.float32)
                raw_rewards = np.asarray([item["rawReward"] for item in items], dtype=np.float32)
                previous_potential = np.asarray([observation["state"]["potential"] for observation in observations], dtype=np.float32)
                next_potential = np.asarray([
                    0.0 if item["done"] else item["observation"]["state"]["potential"] for item in items
                ], dtype=np.float32)
                rewards = raw_rewards + args.shaping_scale * (args.gamma * next_potential - previous_potential)
                for item in items:
                    if item.get("result"):
                        episode_results.append(item["result"])
                rollout_actions.append(action_indices)
                rollout_log_probs.append(log_probabilities.cpu().numpy())
                rollout_values.append(values.cpu().numpy())
                rollout_rewards.append(rewards)
                rollout_dones.append(dones)
                observations = next_observations

            with torch.no_grad():
                _, bootstrap_values_tensor = model(tensorize(observations, metadata, device))
            bootstrap_values = bootstrap_values_tensor.cpu().numpy()
            values_array = np.asarray(rollout_values, dtype=np.float32)
            rewards_array = np.asarray(rollout_rewards, dtype=np.float32)
            dones_array = np.asarray(rollout_dones, dtype=np.float32)
            advantages = np.zeros_like(rewards_array)
            next_advantage = np.zeros(args.envs, dtype=np.float32)
            next_values = bootstrap_values
            for step in reversed(range(args.rollout_steps)):
                not_done = 1.0 - dones_array[step]
                delta = rewards_array[step] + args.gamma * next_values * not_done - values_array[step]
                next_advantage = delta + args.gamma * args.gae_lambda * not_done * next_advantage
                advantages[step] = next_advantage
                next_values = values_array[step]
            returns = advantages + values_array

            flat_observations = [observation for step in rollout_observations for observation in step]
            flat_actions = np.asarray(rollout_actions, dtype=np.int64).reshape(-1)
            flat_old_log_probs = np.asarray(rollout_log_probs, dtype=np.float32).reshape(-1)
            flat_old_values = values_array.reshape(-1)
            flat_advantages = advantages.reshape(-1)
            flat_returns = returns.reshape(-1)
            flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)
            sample_count = len(flat_observations)
            order = np.arange(sample_count)

            model.train()
            policy_losses: list[float] = []
            value_losses: list[float] = []
            entropies: list[float] = []
            approximate_kls: list[float] = []
            for _ in range(args.epochs):
                np.random.shuffle(order)
                for start in range(0, sample_count, args.minibatch_size):
                    indices = order[start : start + args.minibatch_size]
                    batch = tensorize([flat_observations[index] for index in indices], metadata, device)
                    selected_actions = torch.as_tensor(flat_actions[indices], device=device)
                    old_log_probs = torch.as_tensor(flat_old_log_probs[indices], device=device)
                    old_values = torch.as_tensor(flat_old_values[indices], device=device)
                    target_advantages = torch.as_tensor(flat_advantages[indices], device=device)
                    target_returns = torch.as_tensor(flat_returns[indices], device=device)
                    logits, predicted_values = model(batch)
                    distribution = Categorical(logits=logits)
                    new_log_probs = distribution.log_prob(selected_actions)
                    entropy = distribution.entropy().mean()
                    log_ratio = new_log_probs - old_log_probs
                    ratio = log_ratio.exp()
                    unclipped = -target_advantages * ratio
                    clipped = -target_advantages * torch.clamp(ratio, 1 - args.clip, 1 + args.clip)
                    policy_loss = torch.maximum(unclipped, clipped).mean()
                    clipped_values = old_values + torch.clamp(predicted_values - old_values, -args.clip, args.clip)
                    value_loss = 0.5 * torch.maximum(
                        (predicted_values - target_returns) ** 2,
                        (clipped_values - target_returns) ** 2,
                    ).mean()
                    loss = policy_loss + args.value_coefficient * value_loss - args.entropy_coefficient * entropy
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    with torch.no_grad():
                        approximate_kl = ((ratio - 1) - log_ratio).mean()
                    policy_losses.append(float(policy_loss.item()))
                    value_losses.append(float(value_loss.item()))
                    entropies.append(float(entropy.item()))
                    approximate_kls.append(float(approximate_kl.item()))

            decisions += sample_count
            elapsed = time.perf_counter() - start_time
            recent = list(episode_results)
            wins = sum(result.get("winner") == "player" for result in recent)
            losses = sum(result.get("winner") == "ai" for result in recent)
            first_games = [result for result in recent if result.get("playerFirst")]
            second_games = [result for result in recent if not result.get("playerFirst")]
            metrics = {
                "deck": args.deck,
                "update": update,
                "decisions": decisions,
                "decisionsPerSecond": round((decisions - starting_decisions) / max(elapsed, 1e-6), 2),
                "episodesWindow": len(recent),
                "winRateWindow": wins / max(1, wins + losses),
                "firstWinRateWindow": sum(result.get("winner") == "player" for result in first_games) / max(1, len(first_games)),
                "secondWinRateWindow": sum(result.get("winner") == "player" for result in second_games) / max(1, len(second_games)),
                "policyLoss": float(np.mean(policy_losses)),
                "valueLoss": float(np.mean(value_losses)),
                "entropy": float(np.mean(entropies)),
                "approximateKl": float(np.mean(approximate_kls)),
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metrics, ensure_ascii=False) + "\n")
            print(json.dumps(metrics, ensure_ascii=False), flush=True)
            if update % args.checkpoint_every == 0 or decisions >= args.total_decisions:
                save_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    metadata,
                    args.deck,
                    initialized["engineVersion"],
                    decisions,
                    update,
                    initialized["metadata"]["cardIds"],
                    config,
                )
            if snapshot_interval and decisions >= next_snapshot:
                snapshot_path = output / f"checkpoint-{decisions}.pt"
                save_checkpoint(
                    snapshot_path,
                    model,
                    optimizer,
                    metadata,
                    args.deck,
                    initialized["engineVersion"],
                    decisions,
                    update,
                    initialized["metadata"]["cardIds"],
                    config,
                )
                print(f"snapshot {snapshot_path}", flush=True)
                while next_snapshot <= decisions:
                    next_snapshot += snapshot_interval
            update += 1

    print(f"saved {checkpoint_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a masked candidate actor-critic policy against the Destruction AI")
    parser.add_argument("--deck", choices=["fairy", "levin"], required=True)
    parser.add_argument("--total-decisions", type=int, default=200_000)
    parser.add_argument("--warmup-decisions", type=int, default=20_000)
    parser.add_argument("--envs", type=int, default=32)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--entropy-coefficient", type=float, default=0.015)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.7)
    parser.add_argument("--shaping-scale", type=float, default=1.0)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--snapshot-every-decisions", type=int, default=100_000)
    parser.add_argument("--base-seed", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
