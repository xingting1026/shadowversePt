from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

from model import (
    ActionBatch,
    CandidateActorCritic,
    PolicyBatch,
    StateBatch,
    load_model_state_compatible,
    metadata_from_checkpoint,
)


class BrowserPolicy(nn.Module):
    """Flat-tensor ONNX boundary used by onnxruntime-web."""

    def __init__(self, model: CandidateActorCritic):
        super().__init__()
        self.model = model

    def forward(
        self,
        scalars: torch.Tensor,
        zone_counts: torch.Tensor,
        field_cards: torch.Tensor,
        field_numbers: torch.Tensor,
        event_cards: torch.Tensor,
        event_numbers: torch.Tensor,
        kinds: torch.Tensor,
        cards: torch.Tensor,
        abilities: torch.Tensor,
        zones: torch.Tensor,
        selected_cards: torch.Tensor,
        selected_specials: torch.Tensor,
        numbers: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(
            PolicyBatch(
                state=StateBatch(
                    scalars=scalars,
                    zone_counts=zone_counts,
                    field_cards=field_cards,
                    field_numbers=field_numbers,
                    event_cards=event_cards,
                    event_numbers=event_numbers,
                ),
                actions=ActionBatch(
                    kinds=kinds,
                    cards=cards,
                    abilities=abilities,
                    zones=zones,
                    selected_cards=selected_cards,
                    selected_specials=selected_specials,
                    numbers=numbers,
                    mask=mask,
                ),
            )
        )


INPUT_NAMES = [
    "scalars",
    "zone_counts",
    "field_cards",
    "field_numbers",
    "event_cards",
    "event_numbers",
    "kinds",
    "cards",
    "abilities",
    "zones",
    "selected_cards",
    "selected_specials",
    "numbers",
    "mask",
]


def sample_inputs(metadata, batch: int = 2, actions: int = 7) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(20260816)
    floats = lambda *shape: torch.randn(*shape, generator=generator, dtype=torch.float32)
    integers = lambda high, *shape: torch.randint(0, high, shape, generator=generator, dtype=torch.int64)
    mask = torch.ones((batch, actions), dtype=torch.bool)
    if actions > 1:
        mask[-1, -1] = False
    return (
        floats(batch, metadata.scalar_size),
        floats(batch, metadata.zone_count, metadata.card_vocabulary_size),
        integers(metadata.card_vocabulary_size, batch, metadata.field_slots),
        floats(batch, metadata.field_slots, metadata.field_number_size),
        integers(metadata.card_vocabulary_size, batch, metadata.recent_event_slots),
        floats(batch, metadata.recent_event_slots, metadata.recent_event_number_size),
        integers(metadata.action_kind_count, batch, actions),
        integers(metadata.card_vocabulary_size, batch, actions),
        integers(metadata.ability_bucket_count, batch, actions),
        integers(4, batch, actions),
        integers(metadata.card_vocabulary_size, batch, actions, metadata.selection_slots),
        integers(6, batch, actions, metadata.selection_slots),
        floats(batch, actions, metadata.action_number_size),
        mask,
    )


def numpy_inputs(inputs: tuple[torch.Tensor, ...]) -> dict[str, np.ndarray]:
    return {name: tensor.detach().cpu().numpy() for name, tensor in zip(INPUT_NAMES, inputs, strict=True)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a trained destruction policy for the browser")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    metadata = metadata_from_checkpoint(checkpoint["metadata"])
    model = CandidateActorCritic(metadata)
    load_model_state_compatible(model, checkpoint["model"])
    wrapper = BrowserPolicy(model.eval()).eval()
    inputs = sample_inputs(metadata)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    dynamic_axes = {name: {0: "batch"} for name in INPUT_NAMES}
    for name in ["kinds", "cards", "abilities", "zones", "selected_cards", "selected_specials", "numbers", "mask"]:
        dynamic_axes[name][1] = "actions"
    dynamic_axes.update({"logits": {0: "batch", 1: "actions"}, "value": {0: "batch"}})

    torch.onnx.export(
        wrapper,
        inputs,
        args.output,
        input_names=INPUT_NAMES,
        output_names=["logits", "value"],
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
    )
    onnx.checker.check_model(onnx.load(args.output))

    with torch.inference_mode():
        expected_logits, expected_value = wrapper(*inputs)
    session = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    actual_logits, actual_value = session.run(None, numpy_inputs(inputs))
    logits_error = float(np.max(np.abs(actual_logits - expected_logits.numpy())))
    value_error = float(np.max(np.abs(actual_value - expected_value.numpy())))
    if logits_error > 2e-4 or value_error > 2e-4:
        raise RuntimeError(f"ONNX parity failed: logits={logits_error}, value={value_error}")

    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    manifest = {
        "format": "shadowverse-pt-browser-policy",
        "formatVersion": 1,
        "policy": checkpoint["policy"],
        "cycle": int(checkpoint["cycle"]),
        "engineVersion": int(checkpoint["engine_version"]),
        "selfPlay": bool(checkpoint["self_play"]),
        "model": args.output.name,
        "sha256": digest,
        "bytes": args.output.stat().st_size,
        "cardIds": checkpoint["card_ids"],
        "metadata": {
            "cardVocabularySize": metadata.card_vocabulary_size,
            "zoneCount": metadata.zone_count,
            "scalarSize": metadata.scalar_size,
            "fieldSlots": metadata.field_slots,
            "fieldNumberSize": metadata.field_number_size,
            "recentEventSlots": metadata.recent_event_slots,
            "recentEventNumberSize": metadata.recent_event_number_size,
            "actionKindCount": metadata.action_kind_count,
            "abilityBucketCount": metadata.ability_bucket_count,
            "selectionSlots": metadata.selection_slots,
            "actionNumberSize": metadata.action_number_size,
        },
        "onnxParity": {
            "sampleBatch": 2,
            "sampleActions": 7,
            "maxLogitError": logits_error,
            "maxValueError": value_error,
        },
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "sha256": digest, "bytes": args.output.stat().st_size, "parity": manifest["onnxParity"]}))


if __name__ == "__main__":
    main()
