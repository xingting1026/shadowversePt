from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from evaluate_league import choose_checkpoints, cycle_number, evaluate_pair, load_checkpoint
from model import metadata_from_checkpoint


def main(args: argparse.Namespace) -> None:
    repo = Path(__file__).resolve().parents[1]
    league = Path(args.league)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    destruction_paths = choose_checkpoints(league / "destruction", args.stride)
    current_destruction_path = destruction_paths[-1]
    current_destruction, current_destruction_saved = load_checkpoint(current_destruction_path, device)
    output: dict[str, Any] = {
        "format": "shadowverse-pt-league-frontier",
        "engineVersion": 3,
        "gamesPerPair": args.games,
        "stride": args.stride,
        "decks": {},
    }
    for deck_index, deck in enumerate(("fairy", "levin")):
        player_paths = choose_checkpoints(league / deck, args.stride)
        current_player_path = player_paths[-1]
        current_player, current_player_saved = load_checkpoint(current_player_path, device)
        metadata = metadata_from_checkpoint(current_player_saved["metadata"])
        if current_player_saved["metadata"] != current_destruction_saved["metadata"]:
            raise RuntimeError("league checkpoints use different encodings")
        seed = args.base_seed + deck_index * 100_000_000

        current_player_row: list[dict[str, Any]] = []
        current_pair: dict[str, Any] | None = None
        for destruction_path in destruction_paths:
            if destruction_path == current_destruction_path:
                destruction_model = current_destruction
            else:
                destruction_model, _ = load_checkpoint(destruction_path, device)
            result = evaluate_pair(
                repo, deck, current_player, destruction_model, metadata, device,
                args.games, args.envs, seed,
            )
            result["playerCycle"] = cycle_number(current_player_path)
            result["destructionCycle"] = cycle_number(destruction_path)
            current_player_row.append(result)
            if destruction_path == current_destruction_path:
                current_pair = result
            print(json.dumps({"deck": deck, "frontier": "current-player", **result}), flush=True)
            if destruction_path != current_destruction_path:
                del destruction_model

        historical_player_column: list[dict[str, Any]] = []
        for player_path in player_paths:
            if player_path == current_player_path:
                assert current_pair is not None
                historical_player_column.append(current_pair)
                continue
            player_model, player_saved = load_checkpoint(player_path, device)
            if player_saved["metadata"] != current_destruction_saved["metadata"]:
                raise RuntimeError("league checkpoints use different encodings")
            result = evaluate_pair(
                repo, deck, player_model, current_destruction, metadata, device,
                args.games, args.envs, seed,
            )
            result["playerCycle"] = cycle_number(player_path)
            result["destructionCycle"] = cycle_number(current_destruction_path)
            historical_player_column.append(result)
            print(json.dumps({"deck": deck, "frontier": "current-destruction", **result}), flush=True)
            del player_model

        assert current_pair is not None
        current = current_pair["playerWinRate"]
        historical_player_best = max(cell["playerWinRate"] for cell in historical_player_column)
        historical_destruction_best = min(cell["playerWinRate"] for cell in current_player_row)
        output["decks"][deck] = {
            "currentPair": current_pair,
            "currentPlayerRow": current_player_row,
            "historicalPlayerColumn": historical_player_column,
            "historicalPlayerExploitGap": historical_player_best - current,
            "historicalDestructionExploitGap": current - historical_destruction_best,
            "maxHistoricalExploitGap": max(historical_player_best - current, current - historical_destruction_best),
        }
        del current_player

    output_path = Path(args.output) if args.output else league / "league-frontier.json"
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"saved {output_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the latest league row and column against sampled history")
    parser.add_argument("--league", required=True)
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--envs", type=int, default=32)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=980_000_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
