import {
  TRAINING_ENCODING_METADATA,
  encodeTrainingAction,
  encodeTrainingState,
} from "./training-encoding";
import type { GameState } from "./engine";
import type { TrainingAction } from "./training";

export type PolicyTensorData = Float32Array | BigInt64Array | Uint8Array;

export type PolicyTensorInput = {
  type: "float32" | "int64" | "bool";
  data: PolicyTensorData;
  dims: number[];
};

export type BrowserPolicyInputs = Record<
  | "scalars"
  | "zone_counts"
  | "field_cards"
  | "field_numbers"
  | "event_cards"
  | "event_numbers"
  | "kinds"
  | "cards"
  | "abilities"
  | "zones"
  | "selected_cards"
  | "selected_specials"
  | "numbers"
  | "mask",
  PolicyTensorInput
>;

function floats(values: number[], dims: number[]): PolicyTensorInput {
  return { type: "float32", data: Float32Array.from(values), dims };
}

function integers(values: number[], dims: number[]): PolicyTensorInput {
  return { type: "int64", data: BigInt64Array.from(values, (value) => BigInt(Math.trunc(value))), dims };
}

function booleanMask(length: number): PolicyTensorInput {
  return { type: "bool", data: new Uint8Array(length).fill(1), dims: [1, length] };
}

function pad(values: number[], size: number): number[] {
  const result = new Array<number>(size).fill(0);
  for (let index = 0; index < Math.min(size, values.length); index += 1) result[index] = values[index];
  return result;
}

/** Reproduces training/model.py:tensorize exactly for a single live decision. */
export function buildBrowserPolicyInputs(state: GameState, actions: TrainingAction[]): BrowserPolicyInputs {
  if (!actions.length) throw new Error("模型推論需要至少一個合法動作");
  const metadata = TRAINING_ENCODING_METADATA;
  const encodedState = encodeTrainingState(state, "ai");
  const encodedActions = actions.map((action) => encodeTrainingAction(state, action));
  if (encodedState.scalars.length !== metadata.scalarSize) {
    throw new Error(`狀態編碼長度不符：${encodedState.scalars.length}/${metadata.scalarSize}`);
  }

  const zoneCounts = new Array<number>(metadata.zoneNames.length * metadata.cardVocabularySize).fill(0);
  encodedState.zones.forEach((cards, zoneIndex) => {
    for (const card of cards) {
      if (card >= 0 && card < metadata.cardVocabularySize) {
        zoneCounts[zoneIndex * metadata.cardVocabularySize + card] += 1 / 3;
      }
    }
  });

  const fieldCards = new Array<number>(metadata.fieldSlots).fill(0);
  const fieldNumbers = new Array<number>(metadata.fieldSlots * metadata.fieldNumberSize).fill(0);
  encodedState.field.slice(0, metadata.fieldSlots).forEach((card, slot) => {
    fieldCards[slot] = card.card;
    fieldNumbers.splice(slot * metadata.fieldNumberSize, metadata.fieldNumberSize, ...pad(card.numbers, metadata.fieldNumberSize));
  });

  const eventCards = new Array<number>(metadata.recentEventSlots).fill(0);
  const eventNumbers = new Array<number>(metadata.recentEventSlots * metadata.recentEventNumberSize).fill(0);
  const recentEvents = encodedState.recentEvents.slice(-metadata.recentEventSlots);
  const eventStart = metadata.recentEventSlots - recentEvents.length;
  recentEvents.forEach((event, offset) => {
    const slot = eventStart + offset;
    eventCards[slot] = event.card;
    eventNumbers.splice(slot * metadata.recentEventNumberSize, metadata.recentEventNumberSize, ...pad(event.numbers, metadata.recentEventNumberSize));
  });

  const actionCount = encodedActions.length;
  const selectedCards = encodedActions.flatMap((action) => pad(action.selectedCards, metadata.actionSelectionSlots));
  const selectedSpecials = encodedActions.flatMap((action) => pad(action.selectedSpecials, metadata.actionSelectionSlots));
  const actionNumbers = encodedActions.flatMap((action) => pad(action.numbers, metadata.actionNumberSize));

  return {
    scalars: floats(encodedState.scalars, [1, metadata.scalarSize]),
    zone_counts: floats(zoneCounts, [1, metadata.zoneNames.length, metadata.cardVocabularySize]),
    field_cards: integers(fieldCards, [1, metadata.fieldSlots]),
    field_numbers: floats(fieldNumbers, [1, metadata.fieldSlots, metadata.fieldNumberSize]),
    event_cards: integers(eventCards, [1, metadata.recentEventSlots]),
    event_numbers: floats(eventNumbers, [1, metadata.recentEventSlots, metadata.recentEventNumberSize]),
    kinds: integers(encodedActions.map((action) => action.kind), [1, actionCount]),
    cards: integers(encodedActions.map((action) => action.card), [1, actionCount]),
    abilities: integers(encodedActions.map((action) => action.ability), [1, actionCount]),
    zones: integers(encodedActions.map((action) => action.zone), [1, actionCount]),
    selected_cards: integers(selectedCards, [1, actionCount, metadata.actionSelectionSlots]),
    selected_specials: integers(selectedSpecials, [1, actionCount, metadata.actionSelectionSlots]),
    numbers: floats(actionNumbers, [1, actionCount, metadata.actionNumberSize]),
    mask: booleanMask(actionCount),
  };
}
