from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class EncodingMetadata:
    card_vocabulary_size: int
    zone_count: int
    scalar_size: int
    field_slots: int
    field_number_size: int
    recent_event_slots: int
    recent_event_number_size: int
    action_kind_count: int
    ability_bucket_count: int
    selection_slots: int
    action_number_size: int

    @classmethod
    def from_server(cls, value: dict[str, Any]) -> "EncodingMetadata":
        return cls(
            card_vocabulary_size=int(value["cardVocabularySize"]),
            zone_count=len(value["zoneNames"]),
            scalar_size=int(value["scalarSize"]),
            field_slots=int(value["fieldSlots"]),
            field_number_size=int(value["fieldNumberSize"]),
            recent_event_slots=int(value["recentEventSlots"]),
            recent_event_number_size=int(value["recentEventNumberSize"]),
            action_kind_count=int(value["actionKindCount"]),
            ability_bucket_count=int(value["abilityBucketCount"]),
            selection_slots=int(value["actionSelectionSlots"]),
            action_number_size=int(value["actionNumberSize"]),
        )


@dataclass
class StateBatch:
    scalars: torch.Tensor
    zone_counts: torch.Tensor
    field_cards: torch.Tensor
    field_numbers: torch.Tensor
    event_cards: torch.Tensor
    event_numbers: torch.Tensor


@dataclass
class ActionBatch:
    kinds: torch.Tensor
    cards: torch.Tensor
    abilities: torch.Tensor
    zones: torch.Tensor
    selected_cards: torch.Tensor
    selected_specials: torch.Tensor
    numbers: torch.Tensor
    mask: torch.Tensor


@dataclass
class PolicyBatch:
    state: StateBatch
    actions: ActionBatch


def _state_arrays(observations: Sequence[dict[str, Any]], metadata: EncodingMetadata) -> tuple[np.ndarray, ...]:
    batch = len(observations)
    scalars = np.zeros((batch, metadata.scalar_size), dtype=np.float32)
    zone_counts = np.zeros((batch, metadata.zone_count, metadata.card_vocabulary_size), dtype=np.float32)
    field_cards = np.zeros((batch, metadata.field_slots), dtype=np.int64)
    field_numbers = np.zeros((batch, metadata.field_slots, metadata.field_number_size), dtype=np.float32)
    event_cards = np.zeros((batch, metadata.recent_event_slots), dtype=np.int64)
    event_numbers = np.zeros((batch, metadata.recent_event_slots, metadata.recent_event_number_size), dtype=np.float32)

    for row, observation in enumerate(observations):
        state = observation["state"]
        raw_scalars = state["scalars"]
        if len(raw_scalars) != metadata.scalar_size:
            raise ValueError(f"scalar size changed: expected {metadata.scalar_size}, received {len(raw_scalars)}")
        scalars[row] = raw_scalars
        for zone_index, cards in enumerate(state["zones"]):
            for card in cards:
                if 0 <= card < metadata.card_vocabulary_size:
                    zone_counts[row, zone_index, card] += 1.0 / 3.0
        for slot, card in enumerate(state["field"][: metadata.field_slots]):
            field_cards[row, slot] = card["card"]
            field_numbers[row, slot] = card["numbers"]
        recent = state["recentEvents"][-metadata.recent_event_slots :]
        start = metadata.recent_event_slots - len(recent)
        for slot, event in enumerate(recent, start=start):
            event_cards[row, slot] = event["card"]
            event_numbers[row, slot] = event["numbers"]
    return scalars, zone_counts, field_cards, field_numbers, event_cards, event_numbers


def tensorize(observations: Sequence[dict[str, Any]], metadata: EncodingMetadata, device: torch.device) -> PolicyBatch:
    if not observations:
        raise ValueError("cannot tensorize an empty observation batch")
    state_arrays = _state_arrays(observations, metadata)
    batch = len(observations)
    max_actions = max(len(observation["actions"]) for observation in observations)
    kinds = np.zeros((batch, max_actions), dtype=np.int64)
    cards = np.zeros((batch, max_actions), dtype=np.int64)
    abilities = np.zeros((batch, max_actions), dtype=np.int64)
    zones = np.zeros((batch, max_actions), dtype=np.int64)
    selected_cards = np.zeros((batch, max_actions, metadata.selection_slots), dtype=np.int64)
    selected_specials = np.zeros((batch, max_actions, metadata.selection_slots), dtype=np.int64)
    numbers = np.zeros((batch, max_actions, metadata.action_number_size), dtype=np.float32)
    mask = np.zeros((batch, max_actions), dtype=np.bool_)

    for row, observation in enumerate(observations):
        for column, action in enumerate(observation["actions"]):
            mask[row, column] = True
            kinds[row, column] = action["kind"]
            cards[row, column] = action["card"]
            abilities[row, column] = action["ability"]
            zones[row, column] = action["zone"]
            selection_count = min(metadata.selection_slots, len(action["selectedCards"]))
            selected_cards[row, column, :selection_count] = action["selectedCards"][:selection_count]
            special_count = min(metadata.selection_slots, len(action["selectedSpecials"]))
            selected_specials[row, column, :special_count] = action["selectedSpecials"][:special_count]
            numbers[row, column] = action["numbers"]

    state = StateBatch(
        scalars=torch.as_tensor(state_arrays[0], device=device),
        zone_counts=torch.as_tensor(state_arrays[1], device=device),
        field_cards=torch.as_tensor(state_arrays[2], device=device),
        field_numbers=torch.as_tensor(state_arrays[3], device=device),
        event_cards=torch.as_tensor(state_arrays[4], device=device),
        event_numbers=torch.as_tensor(state_arrays[5], device=device),
    )
    actions = ActionBatch(
        kinds=torch.as_tensor(kinds, device=device),
        cards=torch.as_tensor(cards, device=device),
        abilities=torch.as_tensor(abilities, device=device),
        zones=torch.as_tensor(zones, device=device),
        selected_cards=torch.as_tensor(selected_cards, device=device),
        selected_specials=torch.as_tensor(selected_specials, device=device),
        numbers=torch.as_tensor(numbers, device=device),
        mask=torch.as_tensor(mask, device=device),
    )
    return PolicyBatch(state=state, actions=actions)


class CandidateActorCritic(nn.Module):
    def __init__(self, metadata: EncodingMetadata, card_embedding: int = 24, hidden: int = 256, action_hidden: int = 128):
        super().__init__()
        self.metadata = metadata
        self.card_embedding = nn.Embedding(metadata.card_vocabulary_size, card_embedding, padding_idx=0)
        self.kind_embedding = nn.Embedding(metadata.action_kind_count, 8)
        self.ability_embedding = nn.Embedding(metadata.ability_bucket_count, 12)
        self.zone_embedding = nn.Embedding(4, 4)
        self.special_embedding = nn.Embedding(6, 4, padding_idx=0)

        state_size = (
            metadata.scalar_size
            + metadata.zone_count * metadata.card_vocabulary_size
            + metadata.field_slots * (card_embedding + metadata.field_number_size)
            + metadata.recent_event_slots * (card_embedding + metadata.recent_event_number_size)
        )
        self.state_encoder = nn.Sequential(
            nn.LayerNorm(state_size),
            nn.Linear(state_size, 512),
            nn.SiLU(),
            nn.Linear(512, hidden),
            nn.SiLU(),
        )
        action_size = (
            8 + card_embedding + 12 + 4
            + metadata.selection_slots * card_embedding
            + metadata.selection_slots * 4
            + metadata.action_number_size
        )
        self.action_encoder = nn.Sequential(
            nn.LayerNorm(action_size),
            nn.Linear(action_size, 256),
            nn.SiLU(),
            nn.Linear(256, action_hidden),
            nn.SiLU(),
        )
        self.state_policy = nn.Linear(hidden, action_hidden)
        self.action_bias = nn.Linear(action_hidden, 1)
        # A zero-initialized residual used only for the all-or-none mulligan pair.
        # Keeping it separate lets paired rollout tuning correct rare mulligan
        # decisions without changing play, attack, activation, or choice logits.
        self.mulligan_head = nn.Sequential(nn.Linear(hidden, 64), nn.SiLU(), nn.Linear(64, 1))
        nn.init.zeros_(self.mulligan_head[-1].weight)
        nn.init.zeros_(self.mulligan_head[-1].bias)
        self.value_head = nn.Sequential(nn.Linear(hidden, 128), nn.SiLU(), nn.Linear(128, 1))

    def encode_state(self, state: StateBatch) -> torch.Tensor:
        batch = state.scalars.shape[0]
        field_embeddings = self.card_embedding(state.field_cards)
        event_embeddings = self.card_embedding(state.event_cards)
        flattened = torch.cat(
            [
                state.scalars,
                state.zone_counts.flatten(1),
                torch.cat([field_embeddings, state.field_numbers], dim=-1).reshape(batch, -1),
                torch.cat([event_embeddings, state.event_numbers], dim=-1).reshape(batch, -1),
            ],
            dim=-1,
        )
        return self.state_encoder(flattened)

    def encode_actions(self, actions: ActionBatch) -> torch.Tensor:
        batch, count = actions.kinds.shape
        selected_card_embeddings = self.card_embedding(actions.selected_cards).reshape(batch, count, -1)
        selected_special_embeddings = self.special_embedding(actions.selected_specials).reshape(batch, count, -1)
        flattened = torch.cat(
            [
                self.kind_embedding(actions.kinds),
                self.card_embedding(actions.cards),
                self.ability_embedding(actions.abilities),
                self.zone_embedding(actions.zones),
                selected_card_embeddings,
                selected_special_embeddings,
                actions.numbers,
            ],
            dim=-1,
        )
        return self.action_encoder(flattened)

    def forward(self, batch: PolicyBatch) -> tuple[torch.Tensor, torch.Tensor]:
        state_hidden = self.encode_state(batch.state)
        action_hidden = self.encode_actions(batch.actions)
        query = self.state_policy(state_hidden).unsqueeze(1)
        logits = (query * action_hidden).sum(dim=-1) / (action_hidden.shape[-1] ** 0.5)
        logits = logits + self.action_bias(action_hidden).squeeze(-1)
        mulligan_mask = batch.actions.kinds.eq(0)
        redraw_direction = batch.actions.numbers[..., 12].gt(0.5).to(logits.dtype) * 2.0 - 1.0
        mulligan_residual = self.mulligan_head(state_hidden).squeeze(-1).unsqueeze(1)
        logits = logits + mulligan_mask.to(logits.dtype) * redraw_direction * mulligan_residual
        logits = logits.masked_fill(~batch.actions.mask, torch.finfo(logits.dtype).min)
        value = self.value_head(state_hidden).squeeze(-1)
        return logits, value


def load_model_state_compatible(model: CandidateActorCritic, state: dict[str, torch.Tensor]) -> None:
    """Load pre-mulligan-head checkpoints while rejecting unrelated schema drift."""
    incompatible = model.load_state_dict(state, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    missing = [key for key in incompatible.missing_keys if not key.startswith("mulligan_head.")]
    if unexpected or missing:
        raise RuntimeError(f"incompatible model state: missing={missing}, unexpected={unexpected}")


def checkpoint_metadata(metadata: EncodingMetadata) -> dict[str, int]:
    return {
        "card_vocabulary_size": metadata.card_vocabulary_size,
        "zone_count": metadata.zone_count,
        "scalar_size": metadata.scalar_size,
        "field_slots": metadata.field_slots,
        "field_number_size": metadata.field_number_size,
        "recent_event_slots": metadata.recent_event_slots,
        "recent_event_number_size": metadata.recent_event_number_size,
        "action_kind_count": metadata.action_kind_count,
        "ability_bucket_count": metadata.ability_bucket_count,
        "selection_slots": metadata.selection_slots,
        "action_number_size": metadata.action_number_size,
    }


def metadata_from_checkpoint(value: dict[str, Any]) -> EncodingMetadata:
    return EncodingMetadata(**value)
