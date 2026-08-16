from __future__ import annotations

import argparse
import json
import sys

import torch

from model import CandidateActorCritic, load_model_state_compatible, metadata_from_checkpoint, tensorize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    observations = json.load(sys.stdin)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    metadata = metadata_from_checkpoint(checkpoint["metadata"])
    model = CandidateActorCritic(metadata)
    load_model_state_compatible(model, checkpoint["model"])
    model.eval()
    results = []
    with torch.inference_mode():
        for observation in observations:
            logits, value = model(tensorize([observation], metadata, torch.device("cpu")))
            results.append({"logits": logits[0].tolist(), "value": float(value[0])})
    json.dump(results, sys.stdout)


if __name__ == "__main__":
    main()
