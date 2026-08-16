import { CARDS, type Side } from "./cards";
import {
  attackOf,
  definition,
  hasKeyword,
  maxHealthOf,
  remainingHealthOf,
  type CardInstance,
  type GameState,
  type MatchEvent,
  type PlayerState,
  type Zone,
} from "./engine";
import type { TrainingAction } from "./training";

/**
 * 已部署模型（destruction cycle 13 等）訓練時的卡牌字典順序，永久凍結。
 * 之後加入的新卡一律 append 在這個前綴之後，確保舊卡索引永不改變，
 * 舊 manifest 的模型才能在擴充後的卡池上繼續運作。不要重排、不要插入。
 */
export const FROZEN_CARD_ORDER: readonly string[] = [
  "annihilationSong", "antiAir", "axia", "axiaEvo", "blackArtifact",
  "bouquetFairy", "breathFairyDancer", "brutalGeno", "destructionAffirmerEvo",
  "destructionFanatic", "destructionHermit", "destructionJoy", "destructionPrayer",
  "destructionServant", "destructionServantEvo", "destructionWilderness",
  "destructiveLishenna", "destructiveLishennaEvo", "dissonanceWorshipper",
  "fairy", "fairyArcher", "fairyBladeAmatsu", "fairyDragon", "fairyWisp",
  "fairyland", "forestFairy", "forestFairyEvo", "gawain", "greatZelgenea",
  "levinAlbert", "levinArcher", "levinAxeGeno", "levinDuke", "levinJustice",
  "levinMaim", "levinMaimEvo", "levinMeim", "levinMiim", "levinRunes",
  "levinRunesEvo", "levinSisters", "levinTranscend", "manifestedLishenna",
  "miasmaAria", "miasmaAriaEvo", "naturalAria", "naturalAriaEvo", "newBlack",
  "newWhite", "originalLishenna", "pureWaterFairy", "queenCynthia",
  "returningDissonance", "reverseAmatsu", "reverseAmatsuEvo", "riotousGarden",
  "solo", "tailwindFairy", "tentacleBite", "vistaElf", "whiteArtifact",
  "whiteBlackChapter", "wingQueen", "wonderTree", "zelgenea",
];

const FROZEN_CARD_SET = new Set(FROZEN_CARD_ORDER);
export const TRAINING_CARD_IDS = [
  ...FROZEN_CARD_ORDER,
  ...Object.keys(CARDS).filter((cardId) => !FROZEN_CARD_SET.has(cardId)).sort(),
];
const CARD_INDEX = new Map(TRAINING_CARD_IDS.map((cardId, index) => [cardId, index + 1]));

export const TRAINING_ZONE_NAMES = [
  "selfDeck",
  "selfHand",
  "selfField",
  "selfEx",
  "selfGrave",
  "selfBanished",
  "opponentField",
  "opponentEx",
  "opponentGrave",
  "opponentBanished",
] as const;

const PHASES: GameState["phase"][] = ["setup", "main", "end", "ai"];
const PENDING_KINDS = ["single", "multi", "order", "yesNo", "triggerOrder"] as const;
const ACTION_KINDS: TrainingAction["kind"][] = ["mulligan", "choice", "play", "attack", "activate", "evolve", "end"];
const EVENT_TYPES = ["gameStart", "mulligan", "turnStart", "play", "evolve", "superEvolve", "attack", "quick", "activate", "eggCycle", "gameover"];

export type EncodedFieldCard = {
  card: number;
  numbers: number[];
};

export type EncodedRecentEvent = {
  card: number;
  numbers: number[];
};

export type EncodedTrainingState = {
  scalars: number[];
  zones: number[][];
  field: EncodedFieldCard[];
  recentEvents: EncodedRecentEvent[];
  potential: number;
};

export type EncodedTrainingAction = {
  kind: number;
  card: number;
  ability: number;
  zone: number;
  selectedCards: number[];
  selectedSpecials: number[];
  numbers: number[];
};

export type TrainingEncodingMetadata = {
  cardIds: string[];
  cardVocabularySize: number;
  zoneNames: readonly string[];
  scalarSize: number;
  fieldSlots: number;
  fieldNumberSize: number;
  recentEventSlots: number;
  recentEventNumberSize: number;
  actionKindCount: number;
  abilityBucketCount: number;
  actionSelectionSlots: number;
  actionNumberSize: number;
};

const ABILITY_BUCKETS = 128;
const FIELD_SLOTS = 10;
const EVENT_SLOTS = 8;
const SELECTION_SLOTS = 5;

function cardIndex(cardId?: string): number {
  return cardId ? CARD_INDEX.get(cardId) ?? 0 : 0;
}

function oneHot<T>(values: readonly T[], selected: T | undefined): number[] {
  return values.map((value) => Number(value === selected));
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function hashBucket(text: string): number {
  let hash = 2166136261;
  for (let index = 0; index < text.length; index += 1) {
    hash ^= text.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0) % ABILITY_BUCKETS;
}

function fieldValue(state: GameState, card: CardInstance): number {
  const def = definition(card);
  if (def.kind !== "follower") return 1.2 + def.cost * 0.25 + (card.tapped ? -0.1 : 0.15);
  return attackOf(card) + remainingHealthOf(card) * 0.75
    + (hasKeyword(state, card, "ward") ? 1.1 : 0)
    + (hasKeyword(state, card, "storm") ? 0.65 : 0)
    + (card.tapped ? -0.2 : 0.2);
}

/**
 * 只使用公開資訊與自己的資訊。用於 potential-based reward shaping；終局固定回到 0，
 * 因此 gamma*Phi(s')-Phi(s) 不改變「最大化最終勝率」的最優策略。
 */
export function trainingPotential(state: GameState, actor: Side = "player"): number {
  if (state.status === "gameover") return 0;
  const board = state.player.field.reduce((sum, card) => sum + fieldValue(state, card), 0)
    - state.ai.field.reduce((sum, card) => sum + fieldValue(state, card), 0);
  const hp = state.player.hp - state.ai.hp;
  const cards = (state.player.hand.length + state.player.ex.length * 0.7)
    - (state.ai.hand.length + state.ai.ex.length * 0.7);
  const playerValue = clamp(hp / 60 + board / 90 + cards / 45, -0.75, 0.75);
  return actor === "player" ? playerValue : -playerValue;
}

function fieldEncoding(state: GameState, card: CardInstance, actor: Side): EncodedFieldCard {
  const def = definition(card);
  return {
    card: cardIndex(card.cardId),
    numbers: [
      card.owner === actor ? 1 : 0,
      def.kind === "follower" ? 1 : 0,
      card.tapped ? 1 : 0,
      attackOf(card) / 12,
      remainingHealthOf(card) / 12,
      maxHealthOf(card) / 12,
      card.enteredAt === state.globalTurn ? 1 : 0,
      card.evolvedThisTurn ? 1 : 0,
      hasKeyword(state, card, "ward") ? 1 : 0,
      hasKeyword(state, card, "storm") ? 1 : 0,
      hasKeyword(state, card, "rush") ? 1 : 0,
      hasKeyword(state, card, "designated") ? 1 : 0,
    ],
  };
}

function eventEncoding(state: GameState, event: MatchEvent, actor: Side): EncodedRecentEvent {
  return {
    card: cardIndex(event.cardId),
    numbers: [
      event.side === actor ? 1 : event.side === "system" ? 0 : -1,
      Math.max(0, EVENT_TYPES.indexOf(event.type)) / Math.max(1, EVENT_TYPES.length - 1),
      clamp((state.globalTurn - event.turn) / 20, 0, 1),
      event.hpPlayer / 20,
      event.hpAi / 20,
      (event.pp ?? 0) / 10,
    ],
  };
}

function zoneIds(cards: CardInstance[]): number[] {
  return cards.map((card) => cardIndex(card.cardId));
}

/** 不包含對手手牌內容或雙方牌庫順序；自己的牌庫只以無順序 multiset 表示。 */
export function encodeTrainingState(state: GameState, actor: Side = "player"): EncodedTrainingState {
  const pending = state.pending;
  const self = state[actor];
  const opponentSide: Side = actor === "player" ? "ai" : "player";
  const opponent = state[opponentSide];
  const actorFirst = actor === "player" ? state.playerFirst : !state.playerFirst;
  const scalars = [
    actorFirst ? 1 : 0,
    state.globalTurn / 24,
    self.ownTurn / 12,
    opponent.ownTurn / 12,
    state.turnSide === actor ? 1 : 0,
    ...oneHot(PHASES, state.phase),
    state.status === "mulligan" ? 1 : 0,
    state.status === "playing" ? 1 : 0,
    self.hp / 20,
    opponent.hp / 20,
    self.maxPP / 10,
    self.pp / 10,
    opponent.maxPP / 10,
    opponent.pp / 10,
    self.ep / 3,
    opponent.ep / 3,
    self.sep,
    opponent.sep,
    self.deck.length / 40,
    opponent.deck.length / 40,
    self.hand.length / 7,
    opponent.hand.length / 7,
    self.field.length / 5,
    opponent.field.length / 5,
    self.ex.length / 5,
    opponent.ex.length / 5,
    self.grave.length / 40,
    opponent.grave.length / 40,
    self.banished.length / 40,
    opponent.banished.length / 40,
    state.playedThisTurn / 10,
    state.evolvedThisTurn ? 1 : 0,
    state.playerDeck === "fairy" ? 1 : 0,
    state.playerDeck === "levin" ? 1 : 0,
    ...oneHot(PENDING_KINDS, pending?.kind),
    (pending?.min ?? 0) / 5,
    (pending?.max ?? 0) / 5,
    (pending?.options.length ?? 0) / 10,
    pending ? hashBucket(pending.effect) / (ABILITY_BUCKETS - 1) : 0,
  ];
  const zones = [
    zoneIds(self.deck).sort((left, right) => left - right),
    zoneIds(self.hand),
    zoneIds(self.field),
    zoneIds(self.ex),
    zoneIds(self.grave),
    zoneIds(self.banished),
    zoneIds(opponent.field),
    zoneIds(opponent.ex),
    zoneIds(opponent.grave),
    zoneIds(opponent.banished),
  ];
  const field = [...self.field, ...opponent.field].map((card) => fieldEncoding(state, card, actor));
  const recentEvents = state.events.slice(-EVENT_SLOTS).map((event) => eventEncoding(state, event, actor));
  return { scalars, zones, field, recentEvents, potential: trainingPotential(state, actor) };
}

function allCards(player: PlayerState): CardInstance[] {
  const zones: Zone[] = ["deck", "hand", "field", "ex", "grave", "banished"];
  return zones.flatMap((zone) => player[zone]);
}

function locateCard(state: GameState, uid: string): CardInstance | undefined {
  return [...allCards(state.player), ...allCards(state.ai)].find((card) => card.uid === uid);
}

function selectionSpecial(uid: string): number {
  if (uid === "ai-leader") return 1;
  if (uid === "player-leader") return 2;
  if (uid === "yes") return 3;
  if (uid === "no") return 4;
  if (uid === "pass") return 5;
  return 0;
}

function actionSource(state: GameState, action: TrainingAction): CardInstance | undefined {
  return "uid" in action ? locateCard(state, action.uid) : undefined;
}

export function encodeTrainingAction(state: GameState, action: TrainingAction): EncodedTrainingAction {
  const source = actionSource(state, action);
  const selected = action.kind === "choice" ? action.selected
    : action.kind === "attack" ? [action.targetUid]
      : action.kind === "activate" ? action.selected ?? []
        : [];
  const selectedCards = selected.map((uid) => {
    const option = state.pending?.options.find((item) => item.uid === uid);
    return cardIndex(option?.cardId ?? locateCard(state, uid)?.cardId);
  }).slice(0, SELECTION_SLOTS);
  const selectedSpecials = selected.map(selectionSpecial).slice(0, SELECTION_SLOTS);
  const selectedInstances = selected.map((uid) => locateCard(state, uid)).filter(Boolean) as CardInstance[];
  const abilityText = action.kind === "choice" ? state.pending?.effect ?? "choice"
    : action.kind === "activate" ? action.abilityId
      : action.kind === "attack" ? "attack"
      : action.kind === "evolve" ? `evolve:${action.superEvolve ? "super" : "normal"}:${action.payment}`
        : action.kind === "play" ? `play:${action.zone}`
          : action.kind === "mulligan" ? `mulligan:${action.redraw ? "redraw" : "keep"}`
            : action.kind;
  const def = source ? definition(source) : undefined;
  const zone = action.kind === "play" ? (action.zone === "hand" ? 1 : 2) : source?.zone === "field" ? 3 : 0;
  return {
    kind: ACTION_KINDS.indexOf(action.kind),
    card: cardIndex("cardId" in action ? action.cardId : source?.cardId),
    ability: hashBucket(abilityText),
    zone,
    selectedCards,
    selectedSpecials,
    numbers: [
      (def?.cost ?? 0) / 10,
      source ? attackOf(source) / 12 : 0,
      source ? remainingHealthOf(source) / 12 : 0,
      source ? maxHealthOf(source) / 12 : 0,
      source?.tapped ? 1 : 0,
      source?.enteredAt === state.globalTurn ? 1 : 0,
      selected.length / SELECTION_SLOTS,
      selected.length === 0 ? 1 : 0,
      selectedInstances.reduce((sum, card) => sum + attackOf(card), 0) / 20,
      selectedInstances.reduce((sum, card) => sum + remainingHealthOf(card), 0) / 20,
      selectedSpecials.includes(1) ? 1 : 0,
      selectedSpecials.includes(2) ? 1 : 0,
      action.kind === "mulligan" && action.redraw ? 1 : 0,
      action.kind === "evolve" && action.superEvolve ? 1 : 0,
      action.kind === "end" ? 1 : 0,
    ],
  };
}

const SAMPLE_STATE = {
  scalarSize: 46,
  fieldNumberSize: 12,
  recentEventNumberSize: 6,
  actionNumberSize: 15,
};

export const TRAINING_ENCODING_METADATA: TrainingEncodingMetadata = {
  cardIds: TRAINING_CARD_IDS,
  cardVocabularySize: TRAINING_CARD_IDS.length + 1,
  zoneNames: TRAINING_ZONE_NAMES,
  scalarSize: SAMPLE_STATE.scalarSize,
  fieldSlots: FIELD_SLOTS,
  fieldNumberSize: SAMPLE_STATE.fieldNumberSize,
  recentEventSlots: EVENT_SLOTS,
  recentEventNumberSize: SAMPLE_STATE.recentEventNumberSize,
  actionKindCount: ACTION_KINDS.length,
  abilityBucketCount: ABILITY_BUCKETS,
  actionSelectionSlots: SELECTION_SLOTS,
  actionNumberSize: SAMPLE_STATE.actionNumberSize,
};
