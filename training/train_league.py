from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.distributions import Categorical

from bridge import TrainingBridge
from model import CandidateActorCritic, EncodingMetadata, checkpoint_metadata, load_model_state_compatible, tensorize


POLICIES = ("fairy", "levin", "destruction")


@dataclass
class Transition:
    observation: dict[str, Any]
    action: int
    log_probability: float
    value: float
    reward: float = 0.0


def heuristic_action_index(observation: dict[str, Any]) -> int:
    """A legal but intentionally simple teacher used only to avoid an all-end-turn cold start."""
    actions = observation["actions"]
    by_kind = lambda kind: [(index, action) for index, action in enumerate(actions) if action["kind"] == kind]
    choices = by_kind(1)
    if choices:
        non_pass = [item for item in choices if 5 not in item[1]["selectedSpecials"]]
        return max(non_pass or choices, key=lambda item: (item[1]["numbers"][6], -item[0]))[0]
    plays = by_kind(2)
    if plays:
        return max(plays, key=lambda item: (item[1]["numbers"][0], -item[0]))[0]
    evolves = by_kind(5)
    if evolves:
        return evolves[0][0]
    attacks = by_kind(3)
    if attacks:
        face = next((item for item in attacks if 1 in item[1]["selectedSpecials"] or 2 in item[1]["selectedSpecials"]), None)
        return (face or attacks[0])[0]
    activations = by_kind(4)
    if activations:
        face = next((item for item in activations if 1 in item[1]["selectedSpecials"] or 2 in item[1]["selectedSpecials"]), None)
        return (face or activations[0])[0]
    mulligans = by_kind(0)
    if mulligans:
        return mulligans[0][0]
    ends = by_kind(6)
    if ends:
        return ends[0][0]
    raise RuntimeError("observation has no legal action")


def act_batch(
    model: CandidateActorCritic,
    observations: Sequence[dict[str, Any]],
    metadata: EncodingMetadata,
    device: torch.device,
    deterministic: bool = False,
) -> tuple[list[int], list[float], list[float]]:
    if not observations:
        return [], [], []
    with torch.no_grad():
        logits, values = model(tensorize(observations, metadata, device))
        distribution = Categorical(logits=logits)
        selected = logits.argmax(dim=-1) if deterministic else distribution.sample()
        log_probabilities = distribution.log_prob(selected)
    return (
        selected.cpu().tolist(),
        log_probabilities.cpu().tolist(),
        values.cpu().tolist(),
    )


def clone_frozen_model(
    source_state: dict[str, torch.Tensor],
    metadata: EncodingMetadata,
    device: torch.device,
) -> CandidateActorCritic:
    model = CandidateActorCritic(metadata).to(device)
    load_model_state_compatible(model, source_state)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def cpu_state_dict(model: CandidateActorCritic) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def sample_pool_index(size: int) -> int:
    if size <= 1 or random.random() < 0.5:
        return size - 1
    return random.randrange(size - 1)


def collect_behavior_samples(
    repo: Path,
    deck: str,
    role: str,
    decisions: int,
    envs: int,
    base_seed: int,
) -> tuple[list[dict[str, Any]], list[int], dict[str, Any]]:
    observations_out: list[dict[str, Any]] = []
    labels: list[int] = []
    with TrainingBridge(repo) as bridge:
        initialized = bridge.request({
            "cmd": "init",
            "envs": envs,
            "deck": deck,
            "baseSeed": base_seed,
            "record": False,
            "selfPlay": True,
        })
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


def behavior_clone(
    model: CandidateActorCritic,
    optimizer: torch.optim.Optimizer,
    observations: list[dict[str, Any]],
    labels: list[int],
    metadata: EncodingMetadata,
    device: torch.device,
    epochs: int,
    minibatch_size: int,
) -> float:
    if not observations:
        return 0.0
    model.train()
    order = np.arange(len(observations))
    losses: list[float] = []
    for _ in range(epochs):
        np.random.shuffle(order)
        for start in range(0, len(order), minibatch_size):
            indices = order[start : start + minibatch_size]
            batch = tensorize([observations[index] for index in indices], metadata, device)
            target = torch.as_tensor([labels[index] for index in indices], device=device)
            logits, _ = model(batch)
            loss = torch.nn.functional.cross_entropy(logits, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
    return float(np.mean(losses))


def episode_advantages(
    episodes: Sequence[Sequence[Transition]],
    gamma: float,
    gae_lambda: float,
) -> tuple[list[Transition], np.ndarray, np.ndarray]:
    flattened: list[Transition] = []
    advantages: list[float] = []
    returns: list[float] = []
    for episode in episodes:
        episode_advantages_values = [0.0] * len(episode)
        episode_returns = [0.0] * len(episode)
        gae = 0.0
        next_value = 0.0
        for index in reversed(range(len(episode))):
            transition = episode[index]
            delta = transition.reward + gamma * next_value - transition.value
            gae = delta + gamma * gae_lambda * gae
            episode_advantages_values[index] = gae
            episode_returns[index] = gae + transition.value
            next_value = transition.value
        flattened.extend(episode)
        advantages.extend(episode_advantages_values)
        returns.extend(episode_returns)
    advantage_array = np.asarray(advantages, dtype=np.float32)
    advantage_array = (advantage_array - advantage_array.mean()) / (advantage_array.std() + 1e-8)
    return flattened, advantage_array, np.asarray(returns, dtype=np.float32)


def ppo_update(
    model: CandidateActorCritic,
    optimizer: torch.optim.Optimizer,
    episodes: Sequence[Sequence[Transition]],
    metadata: EncodingMetadata,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    samples, advantages, returns = episode_advantages(episodes, args.gamma, args.gae_lambda)
    old_log_probabilities = np.asarray([sample.log_probability for sample in samples], dtype=np.float32)
    old_values = np.asarray([sample.value for sample in samples], dtype=np.float32)
    selected_actions = np.asarray([sample.action for sample in samples], dtype=np.int64)
    order = np.arange(len(samples))
    policy_losses: list[float] = []
    value_losses: list[float] = []
    entropies: list[float] = []
    approximate_kls: list[float] = []
    model.train()
    for _ in range(args.epochs):
        np.random.shuffle(order)
        for start in range(0, len(order), args.minibatch_size):
            indices = order[start : start + args.minibatch_size]
            batch = tensorize([samples[index].observation for index in indices], metadata, device)
            action_tensor = torch.as_tensor(selected_actions[indices], device=device)
            old_log_tensor = torch.as_tensor(old_log_probabilities[indices], device=device)
            old_value_tensor = torch.as_tensor(old_values[indices], device=device)
            advantage_tensor = torch.as_tensor(advantages[indices], device=device)
            return_tensor = torch.as_tensor(returns[indices], device=device)
            logits, predicted_values = model(batch)
            distribution = Categorical(logits=logits)
            new_log_probabilities = distribution.log_prob(action_tensor)
            log_ratio = new_log_probabilities - old_log_tensor
            ratio = log_ratio.exp()
            unclipped = -advantage_tensor * ratio
            clipped = -advantage_tensor * torch.clamp(ratio, 1 - args.clip, 1 + args.clip)
            policy_loss = torch.maximum(unclipped, clipped).mean()
            clipped_values = old_value_tensor + torch.clamp(predicted_values - old_value_tensor, -args.clip, args.clip)
            value_loss = 0.5 * torch.maximum(
                (predicted_values - return_tensor) ** 2,
                (clipped_values - return_tensor) ** 2,
            ).mean()
            entropy = distribution.entropy().mean()
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
    return {
        "policyLoss": float(np.mean(policy_losses)),
        "valueLoss": float(np.mean(value_losses)),
        "entropy": float(np.mean(entropies)),
        "approximateKl": float(np.mean(approximate_kls)),
    }


def train_phase(
    repo: Path,
    deck: str,
    target_role: str,
    learner: CandidateActorCritic,
    optimizer: torch.optim.Optimizer,
    opponent_pool: Sequence[dict[str, torch.Tensor]],
    metadata: EncodingMetadata,
    device: torch.device,
    args: argparse.Namespace,
    base_seed: int,
) -> dict[str, Any]:
    frozen_opponents = [clone_frozen_model(state, metadata, device) for state in opponent_pool]
    completed_episodes: list[list[Transition]] = []
    live_episodes: list[list[Transition]] = [[] for _ in range(args.envs)]
    opponent_assignments = [sample_pool_index(len(frozen_opponents)) for _ in range(args.envs)]
    completed_transitions = 0
    target_wins = 0
    target_losses = 0
    draws = 0
    games = 0
    started = time.perf_counter()
    learner.eval()
    with TrainingBridge(repo) as bridge:
        initialized = bridge.request({
            "cmd": "init",
            "envs": args.envs,
            "deck": deck,
            "baseSeed": base_seed,
            "record": False,
            "selfPlay": True,
        })
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
                opponent_observations = [observations[row] for row in rows]
                selected, _, _ = act_batch(frozen_opponents[pool_index], opponent_observations, metadata, device)
                for row, action in zip(rows, selected):
                    action_indices[row] = action

            response = bridge.request({"cmd": "step", "actions": action_indices})
            for row, item in enumerate(response["items"]):
                if not item["done"]:
                    continue
                result = item["result"]
                winner = result.get("winner")
                won = winner == target_role
                lost = winner in ("player", "ai") and not won
                reward = 1.0 if won else -1.0 if lost else 0.0
                if live_episodes[row]:
                    live_episodes[row][-1].reward = reward
                    completed_episodes.append(live_episodes[row])
                    completed_transitions += len(live_episodes[row])
                live_episodes[row] = []
                target_wins += int(won)
                target_losses += int(lost)
                draws += int(not won and not lost)
                games += 1
                opponent_assignments[row] = sample_pool_index(len(frozen_opponents))
            observations = [item["observation"] for item in response["items"]]

    losses = ppo_update(learner, optimizer, completed_episodes, metadata, device, args)
    elapsed = time.perf_counter() - started
    del frozen_opponents
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "deck": deck,
        "targetRole": target_role,
        "games": games,
        "decisions": completed_transitions,
        "wins": target_wins,
        "losses": target_losses,
        "draws": draws,
        "winRate": target_wins / max(1, target_wins + target_losses),
        "opponentPoolSize": len(opponent_pool),
        "seconds": round(elapsed, 3),
        **losses,
    }


def save_policy(
    path: Path,
    policy: str,
    cycle: int,
    model: CandidateActorCritic,
    optimizer: torch.optim.Optimizer,
    metadata: EncodingMetadata,
    initialized: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metadata": checkpoint_metadata(metadata),
        "card_ids": initialized["metadata"]["cardIds"],
        "engine_version": initialized["engineVersion"],
        "policy": policy,
        "cycle": cycle,
        "self_play": True,
        "config": vars(args),
    }, temporary)
    temporary.replace(path)


def load_policy(path: Path, model: CandidateActorCritic, optimizer: torch.optim.Optimizer, initialized: dict[str, Any]) -> int:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("engine_version") != initialized["engineVersion"] or saved.get("card_ids") != initialized["metadata"]["cardIds"]:
        raise RuntimeError(f"incompatible league checkpoint: {path}")
    load_model_state_compatible(model, saved["model"])
    if saved.get("optimizer"):
        try:
            optimizer.load_state_dict(saved["optimizer"])
        except ValueError:
            # Compatible policy checkpoints may predate a newly isolated head.
            pass
    return int(saved.get("cycle", 0))


def load_pool(directory: Path) -> list[dict[str, torch.Tensor]]:
    states: list[dict[str, torch.Tensor]] = []
    for path in sorted(directory.glob("cycle-*.pt")):
        saved = torch.load(path, map_location="cpu", weights_only=False)
        states.append(saved["model"])
    return states


def train(args: argparse.Namespace) -> None:
    repo = Path(__file__).resolve().parents[1]
    output = Path(args.output) if args.output else repo / "training-output" / "league-v1"
    output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    with TrainingBridge(repo) as bridge:
        initialized = bridge.request({
            "cmd": "init", "envs": 1, "deck": "fairy", "baseSeed": args.base_seed,
            "record": False, "selfPlay": True,
        })
    metadata = EncodingMetadata.from_server(initialized["metadata"])
    models = {policy: CandidateActorCritic(metadata).to(device) for policy in POLICIES}
    optimizers = {
        policy: torch.optim.AdamW(models[policy].parameters(), lr=args.learning_rate, eps=1e-5, weight_decay=1e-5)
        for policy in POLICIES
    }
    start_cycle = 0
    if args.resume:
        cycles = []
        for policy in POLICIES:
            current = output / policy / "current.pt"
            if not current.exists():
                raise RuntimeError(f"missing resume checkpoint: {current}")
            cycles.append(load_policy(current, models[policy], optimizers[policy], initialized))
        if len(set(cycles)) != 1:
            raise RuntimeError(f"policy checkpoints are from different cycles: {cycles}")
        start_cycle = cycles[0]
    else:
        warmup_plan = [
            ("fairy", "player", args.warmup_decisions, args.base_seed + 1_000_000),
            ("levin", "player", args.warmup_decisions, args.base_seed + 2_000_000),
            ("destruction", "ai", args.warmup_decisions // 2, args.base_seed + 3_000_000),
            ("destruction", "ai", args.warmup_decisions - args.warmup_decisions // 2, args.base_seed + 4_000_000),
        ]
        warmup_decks = ["fairy", "levin", "fairy", "levin"]
        for (policy, role, count, seed), deck in zip(warmup_plan, warmup_decks):
            if count <= 0:
                continue
            observations, labels, _ = collect_behavior_samples(repo, deck, role, count, args.envs, seed)
            loss = behavior_clone(
                models[policy], optimizers[policy], observations, labels, metadata, device,
                args.warmup_epochs, args.minibatch_size,
            )
            print(json.dumps({"stage": "warmup", "policy": policy, "deck": deck, "decisions": count, "loss": loss}), flush=True)
        for policy in POLICIES:
            cycle_path = output / policy / "cycle-000.pt"
            save_policy(cycle_path, policy, 0, models[policy], optimizers[policy], metadata, initialized, args)
            save_policy(output / policy / "current.pt", policy, 0, models[policy], optimizers[policy], metadata, initialized, args)

    pools = {policy: load_pool(output / policy) for policy in POLICIES}
    if any(not pools[policy] for policy in POLICIES):
        raise RuntimeError("every policy needs at least one pool checkpoint")
    metrics_path = output / "league-metrics.jsonl"
    print(f"device={device} start_cycle={start_cycle} pools={ {key: len(value) for key, value in pools.items()} }", flush=True)

    for cycle in range(start_cycle + 1, args.cycles + 1):
        phase_seed = args.base_seed + cycle * 10_000_000
        phases: list[dict[str, Any]] = []
        phases.append(train_phase(
            repo, "fairy", "player", models["fairy"], optimizers["fairy"], pools["destruction"],
            metadata, device, args, phase_seed + 100_000,
        ))
        fairy_state = cpu_state_dict(models["fairy"])
        pools["fairy"].append(fairy_state)

        phases.append(train_phase(
            repo, "levin", "player", models["levin"], optimizers["levin"], pools["destruction"],
            metadata, device, args, phase_seed + 200_000,
        ))
        levin_state = cpu_state_dict(models["levin"])
        pools["levin"].append(levin_state)

        original_phase_decisions = args.phase_decisions
        args.phase_decisions = max(1, original_phase_decisions // 2)
        phases.append(train_phase(
            repo, "fairy", "ai", models["destruction"], optimizers["destruction"], pools["fairy"],
            metadata, device, args, phase_seed + 300_000,
        ))
        phases.append(train_phase(
            repo, "levin", "ai", models["destruction"], optimizers["destruction"], pools["levin"],
            metadata, device, args, phase_seed + 400_000,
        ))
        args.phase_decisions = original_phase_decisions
        destruction_state = cpu_state_dict(models["destruction"])
        pools["destruction"].append(destruction_state)

        for policy in POLICIES:
            cycle_path = output / policy / f"cycle-{cycle:03d}.pt"
            save_policy(cycle_path, policy, cycle, models[policy], optimizers[policy], metadata, initialized, args)
            save_policy(output / policy / "current.pt", policy, cycle, models[policy], optimizers[policy], metadata, initialized, args)
        record = {"cycle": cycle, "phases": phases, "poolSizes": {key: len(value) for key, value in pools.items()}}
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Alternating multi-agent self-play with historical opponent pools")
    parser.add_argument("--cycles", type=int, default=12)
    parser.add_argument("--phase-decisions", type=int, default=50_000)
    parser.add_argument("--warmup-decisions", type=int, default=10_000)
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
    parser.add_argument("--base-seed", type=int, default=130_000_000)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
