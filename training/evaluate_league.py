from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import torch

from bridge import TrainingBridge
from model import CandidateActorCritic, EncodingMetadata, load_model_state_compatible, metadata_from_checkpoint, tensorize


def load_checkpoint(path: Path, device: torch.device) -> tuple[CandidateActorCritic, dict[str, Any]]:
    saved = torch.load(path, map_location=device, weights_only=False)
    metadata = metadata_from_checkpoint(saved["metadata"])
    model = CandidateActorCritic(metadata).to(device)
    load_model_state_compatible(model, saved["model"])
    model.eval()
    return model, saved


def deterministic_actions(
    model: CandidateActorCritic,
    observations: Sequence[dict[str, Any]],
    metadata: EncodingMetadata,
    device: torch.device,
) -> list[int]:
    if not observations:
        return []
    with torch.no_grad():
        logits, _ = model(tensorize(observations, metadata, device))
    return logits.argmax(dim=-1).cpu().tolist()


def wilson_interval(wins: int, games: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if games <= 0:
        return 0.0, 1.0
    probability = wins / games
    denominator = 1 + z * z / games
    center = (probability + z * z / (2 * games)) / denominator
    radius = z * math.sqrt(probability * (1 - probability) / games + z * z / (4 * games * games)) / denominator
    return center - radius, center + radius


def evaluate_pair(
    repo: Path,
    deck: str,
    player_model: CandidateActorCritic,
    destruction_model: CandidateActorCritic,
    metadata: EncodingMetadata,
    device: torch.device,
    games: int,
    envs: int,
    base_seed: int,
) -> dict[str, Any]:
    player_wins = 0
    destruction_wins = 0
    draws = 0
    first_games = 0
    first_wins = 0
    second_games = 0
    second_wins = 0
    completed = 0

    def run_order(order_games: int, player_first: bool, order_seed: int) -> None:
        nonlocal player_wins, destruction_wins, draws, first_games, first_wins
        nonlocal second_games, second_wins, completed
        order_completed = 0
        with TrainingBridge(repo) as bridge:
            initialized = bridge.request({
                "cmd": "init", "envs": min(envs, order_games), "deck": deck,
                "baseSeed": order_seed, "record": False, "selfPlay": True,
                "fixedFirst": player_first,
            })
            if initialized["engineVersion"] != 3:
                raise RuntimeError("league evaluator requires the manual-AI engine")
            observations = initialized["observations"]
            while order_completed < order_games:
                actions = [0] * len(observations)
                player_rows = [index for index, observation in enumerate(observations) if observation["actor"] == "player"]
                ai_rows = [index for index, observation in enumerate(observations) if observation["actor"] == "ai"]
                for rows, model in ((player_rows, player_model), (ai_rows, destruction_model)):
                    selected = deterministic_actions(model, [observations[index] for index in rows], metadata, device)
                    for row, action in zip(rows, selected):
                        actions[row] = action
                response = bridge.request({"cmd": "step", "actions": actions})
                for item in response["items"]:
                    if not item["done"] or order_completed >= order_games:
                        continue
                    result = item["result"]
                    won = result.get("winner") == "player"
                    player_wins += int(won)
                    destruction_wins += int(result.get("winner") == "ai")
                    draws += int(result.get("winner") not in ("player", "ai"))
                    if player_first:
                        first_games += 1
                        first_wins += int(won)
                    else:
                        second_games += 1
                        second_wins += int(won)
                    order_completed += 1
                    completed += 1
                observations = [item["observation"] for item in response["items"]]

    first_target = (games + 1) // 2
    second_target = games // 2
    run_order(first_target, True, base_seed)
    run_order(second_target, False, base_seed + 50_000_000)
    lower, upper = wilson_interval(player_wins, max(1, player_wins + destruction_wins))
    return {
        "games": completed,
        "playerWins": player_wins,
        "destructionWins": destruction_wins,
        "draws": draws,
        "playerWinRate": player_wins / max(1, player_wins + destruction_wins),
        "wilson95": [lower, upper],
        "firstGames": first_games,
        "firstWinRate": first_wins / max(1, first_games),
        "secondGames": second_games,
        "secondWinRate": second_wins / max(1, second_games),
    }


def cycle_number(path: Path) -> int:
    return int(path.stem.split("-")[-1])


def choose_checkpoints(directory: Path, stride: int) -> list[Path]:
    paths = sorted(directory.glob("cycle-*.pt"), key=cycle_number)
    if stride <= 1 or len(paths) <= 2:
        return paths
    selected = paths[::stride]
    if paths[-1] not in selected:
        selected.append(paths[-1])
    return selected


def evaluate(args: argparse.Namespace) -> None:
    repo = Path(__file__).resolve().parents[1]
    league = Path(args.league)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    destruction_paths = choose_checkpoints(league / "destruction", args.stride)
    output: dict[str, Any] = {
        "format": "shadowverse-pt-league-matrix",
        "engineVersion": 3,
        "gamesPerCell": args.games,
        "stride": args.stride,
        "decks": {},
    }
    for deck_index, deck in enumerate(("fairy", "levin")):
        player_paths = choose_checkpoints(league / deck, args.stride)
        matrix: list[list[dict[str, Any]]] = []
        for player_path in player_paths:
            row: list[dict[str, Any]] = []
            player_model, player_saved = load_checkpoint(player_path, device)
            for destruction_path in destruction_paths:
                destruction_model, destruction_saved = load_checkpoint(destruction_path, device)
                if player_saved["metadata"] != destruction_saved["metadata"]:
                    raise RuntimeError("league checkpoints use different encodings")
                metadata = metadata_from_checkpoint(player_saved["metadata"])
                pair_seed = args.base_seed + deck_index * 100_000_000
                result = evaluate_pair(
                    repo, deck, player_model, destruction_model, metadata, device,
                    args.games, args.envs, pair_seed,
                )
                result["playerCycle"] = cycle_number(player_path)
                result["destructionCycle"] = cycle_number(destruction_path)
                row.append(result)
                print(json.dumps({"deck": deck, **result}), flush=True)
                del destruction_model
            matrix.append(row)
            del player_model
        current_row = matrix[-1]
        current_column = [row[-1] for row in matrix]
        current = matrix[-1][-1]["playerWinRate"]
        historical_player_best = max(cell["playerWinRate"] for cell in current_column)
        historical_destruction_best = min(cell["playerWinRate"] for cell in current_row)
        output["decks"][deck] = {
            "playerCycles": [cycle_number(path) for path in player_paths],
            "destructionCycles": [cycle_number(path) for path in destruction_paths],
            "matrix": matrix,
            "currentPairPlayerWinRate": current,
            "historicalPlayerExploitGap": historical_player_best - current,
            "historicalDestructionExploitGap": current - historical_destruction_best,
            "maxHistoricalExploitGap": max(historical_player_best - current, current - historical_destruction_best),
        }
    output_path = Path(args.output) if args.output else league / "league-matrix.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"saved {output_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate every selected player/destruction checkpoint pairing")
    parser.add_argument("--league", required=True)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--envs", type=int, default=32)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=700_000_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
