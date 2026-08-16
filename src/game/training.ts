import {
  activateFieldCard,
  activateManualAi,
  attackCardForSide,
  attackOf,
  attackTargets,
  cardActions,
  createGame,
  definition,
  endTurn,
  endTurnForSide,
  evolveCard,
  evolveCardForSide,
  finishMulligan,
  finishManualMulligan,
  manualAiActivations,
  playCard,
  playCardForSide,
  remainingHealthOf,
  resolveChoice,
  type CardInstance,
  type GameState,
  type PendingChoice,
  type PlayerState,
  type Zone,
} from "./engine";
import { AI_DECK, PLAYER_DECKS, cardName, type PlayerDeckId, type Side } from "./cards";

export type TrainingAction =
  | { key: string; kind: "mulligan"; redraw: boolean; label: string }
  | { key: string; kind: "choice"; selected: string[]; label: string }
  | { key: string; kind: "play"; uid: string; zone: "hand" | "ex"; cardId: string; label: string }
  | { key: string; kind: "attack"; uid: string; targetUid: string; cardId: string; label: string }
  | { key: string; kind: "activate"; uid: string; abilityId: string; activationKey?: string; selected?: string[]; cardId: string; label: string }
  | { key: string; kind: "evolve"; uid: string; payment: "pp" | "ep"; superEvolve: boolean; cardId: string; label: string }
  | { key: string; kind: "end"; label: string };

export type TrainingCardView = {
  cardId: string;
  baseCardId?: string;
  tapped: boolean;
  attack: number;
  health: number;
  damage: number;
  enteredAt: number;
  evolvedThisTurn: boolean;
};

export type TrainingSideView = {
  hp: number;
  maxPP: number;
  pp: number;
  ep: number;
  sep: number;
  ownTurn: number;
  deckCount: number;
  handCount: number;
  hand?: string[];
  field: TrainingCardView[];
  ex: string[];
  grave: string[];
  banished: string[];
  evolveRemaining: number;
};

export type TrainingObservation = {
  status: GameState["status"];
  winner?: GameState["winner"];
  playerFirst: boolean;
  playerDeck: PlayerDeckId;
  aiControl: GameState["aiControl"];
  actor: Side;
  globalTurn: number;
  turnSide: GameState["turnSide"];
  phase: GameState["phase"];
  playedThisTurn: number;
  evolvedThisTurn: boolean;
  ownDeckList: Record<string, number>;
  opponentDeckList: Record<string, number>;
  self: TrainingSideView;
  opponent: TrainingSideView;
  pending?: Pick<PendingChoice, "kind" | "effect" | "title" | "prompt" | "min" | "max" | "side">;
};

export type TrainingPolicyCandidate = {
  actionKey: string;
  probability: number;
  value?: number;
};

export type TrainingDecisionAudit = {
  /** 模型在落子前對局面的勝率／價值估計。 */
  value?: number;
  /** 可保留完整策略分布，讓回放不只看到落子，也能看到 AI 當時考慮了什麼。 */
  policy?: TrainingPolicyCandidate[];
  note?: string;
};

export type TrainingReplayDecision = {
  index: number;
  actor: "player" | "ai";
  globalTurn: number;
  ownTurn: number;
  phase: GameState["phase"];
  actionKey: string;
  actionLabel: string;
  legalActionKeys: string[];
  audit?: TrainingDecisionAudit;
};

export type TrainingReplayResult = {
  status: GameState["status"];
  winner?: GameState["winner"];
  reward: number;
  globalTurn: number;
  playerHp: number;
  aiHp: number;
  eventCount: number;
};

export type TrainingReplay = {
  format: "shadowverse-pt-training-replay";
  replayVersion: 1;
  engineVersion: GameState["version"];
  seed: number;
  playerFirst: boolean;
  playerDeck: PlayerDeckId;
  aiControl?: GameState["aiControl"];
  decisions: TrainingReplayDecision[];
  result?: TrainingReplayResult;
};

function combinations<T>(items: T[], count: number, start = 0, prefix: T[] = [], output: T[][] = []): T[][] {
  if (prefix.length === count) {
    output.push([...prefix]);
    return output;
  }
  for (let index = start; index <= items.length - (count - prefix.length); index += 1) {
    prefix.push(items[index]);
    combinations(items, count, index + 1, prefix, output);
    prefix.pop();
  }
  return output;
}

function orderedSelections<T>(items: T[], count: number, prefix: T[] = [], output: T[][] = []): T[][] {
  if (prefix.length === count) {
    output.push([...prefix]);
    return output;
  }
  for (const item of items) {
    if (prefix.includes(item)) continue;
    prefix.push(item);
    orderedSelections(items, count, prefix, output);
    prefix.pop();
  }
  return output;
}

function selectableOptions(pending: PendingChoice) {
  return pending.options.filter((option) => !option.description?.includes("不符合"));
}

function choiceLabel(pending: PendingChoice, selected: string[]): string {
  if (!selected.length) return `${pending.title}：不選擇`;
  const names = selected.map((uid) => {
    const option = pending.options.find((item) => item.uid === uid);
    return option?.label ?? (option?.cardId ? cardName(option.cardId) : uid);
  });
  return `${pending.title}：${names.join(" → ")}`;
}

function pendingActions(pending: PendingChoice): TrainingAction[] {
  const options = selectableOptions(pending);
  const uids = options.map((option) => option.uid);
  let selections: string[][] = [];
  if (pending.kind === "order") {
    if (pending.max <= uids.length) selections = orderedSelections(uids, pending.max);
  } else if (pending.kind === "single" || pending.kind === "yesNo" || pending.kind === "triggerOrder") {
    if (pending.min === 0) selections.push([]);
    selections.push(...uids.map((uid) => [uid]));
  } else {
    const max = Math.min(pending.max, uids.length);
    for (let count = pending.min; count <= max; count += 1) selections.push(...combinations(uids, count));
  }
  return selections.map((selected) => ({
    key: `choice:${selected.length ? selected.join(",") : "none"}`,
    kind: "choice",
    selected,
    label: choiceLabel(pending, selected),
  }));
}

function fieldCardView(card: CardInstance): TrainingCardView {
  return {
    cardId: card.cardId,
    baseCardId: card.baseCardId,
    tapped: card.tapped,
    attack: attackOf(card),
    health: remainingHealthOf(card),
    damage: card.damage,
    enteredAt: card.enteredAt,
    evolvedThisTurn: card.evolvedThisTurn,
  };
}

function sideView(player: PlayerState, revealHand: boolean): TrainingSideView {
  return {
    hp: player.hp,
    maxPP: player.maxPP,
    pp: player.pp,
    ep: player.ep,
    sep: player.sep,
    ownTurn: player.ownTurn,
    deckCount: player.deck.length,
    handCount: player.hand.length,
    hand: revealHand ? player.hand.map((card) => card.cardId) : undefined,
    field: player.field.map(fieldCardView),
    ex: player.ex.map((card) => card.cardId),
    grave: player.grave.map((card) => card.cardId),
    banished: player.banished.map((card) => card.cardId),
    evolveRemaining: Object.values(player.evolveRemaining).reduce((total, count) => total + count, 0),
  };
}

function deckCounts(list: [string, number][]): Record<string, number> {
  return Object.fromEntries(list);
}

/** 目前必須作答的策略。pending.side 未標註的既有互動都屬於 player。 */
export function trainingActor(state: GameState): Side | undefined {
  if (state.status === "gameover") return undefined;
  if (state.status === "mulligan") {
    if (state.aiControl === "scripted") return "player";
    return state.mulliganDone.player ? "ai" : "player";
  }
  if (state.pending) return state.pending.side ?? "player";
  if (state.phase === "main") return state.turnSide;
  return undefined;
}

/** 玩家視角的 observation：包含已知牌表與自己手牌，但不包含對手手牌或任何牌庫順序。 */
export function trainingObservation(state: GameState, actor = trainingActor(state) ?? "player"): TrainingObservation {
  const playerDeck = PLAYER_DECKS[state.playerDeck];
  const self = actor === "player" ? state.player : state.ai;
  const opponent = actor === "player" ? state.ai : state.player;
  return {
    status: state.status,
    winner: state.winner,
    playerFirst: state.playerFirst,
    playerDeck: state.playerDeck,
    aiControl: state.aiControl,
    actor,
    globalTurn: state.globalTurn,
    turnSide: state.turnSide,
    phase: state.phase,
    playedThisTurn: state.playedThisTurn,
    evolvedThisTurn: state.evolvedThisTurn,
    ownDeckList: deckCounts(actor === "player" ? playerDeck.main : AI_DECK),
    opponentDeckList: deckCounts(actor === "player" ? AI_DECK : playerDeck.main),
    self: sideView(self, true),
    opponent: sideView(opponent, false),
    pending: state.pending ? {
      kind: state.pending.kind,
      effect: state.pending.effect,
      title: state.pending.title,
      prompt: state.pending.prompt,
      min: state.pending.min,
      max: state.pending.max,
      side: state.pending.side,
    } : undefined,
  };
}

/** 列出目前由妖精玩家決定的完整合法動作；AI 回合若沒有玩家反應時會由引擎自動推進。 */
export function trainingLegalActions(state: GameState): TrainingAction[] {
  if (state.status === "gameover") return [];
  const actor = trainingActor(state);
  if (!actor) return [];
  if (state.status === "mulligan") {
    return [
      { key: `mulligan:${actor}:keep`, kind: "mulligan", redraw: false, label: `${actor === "player" ? "玩家" : "破壞方"}保留起手` },
      { key: `mulligan:${actor}:redraw`, kind: "mulligan", redraw: true, label: `${actor === "player" ? "玩家" : "破壞方"}全部重抽` },
    ];
  }
  if (state.pending) return pendingActions(state.pending);
  if (state.turnSide !== actor || state.phase !== "main") return [];

  const actions: TrainingAction[] = [];
  const actorState = state[actor];
  const playableZones: ("hand" | "ex")[] = ["hand", "ex"];
  for (const zone of playableZones) {
    for (const card of actorState[zone]) {
      const play = cardActions(state, card.uid, zone, actor).find((action) => action.id === "play" && action.enabled);
      if (play) actions.push({
        key: `play:${actor}:${zone}:${card.uid}`,
        kind: "play",
        uid: card.uid,
        zone,
        cardId: card.cardId,
        label: `使用${cardName(card.cardId)}`,
      });
    }
  }

  for (const card of actorState.field) {
    for (const action of cardActions(state, card.uid, "field", actor).filter((item) => item.enabled)) {
      if (action.id.startsWith("evolve-") || action.id.startsWith("super-")) {
        const superEvolve = action.id.startsWith("super-");
        const payment = action.id.endsWith("-ep") ? "ep" : "pp";
        actions.push({
          key: `evolve:${actor}:${superEvolve ? "super" : "normal"}:${payment}:${card.uid}`,
          kind: "evolve",
          uid: card.uid,
          payment,
          superEvolve,
          cardId: card.cardId,
          label: `${action.label}：${cardName(card.cardId)}`,
        });
      } else if (action.id === "attack") {
        for (const target of attackTargets(state, card)) actions.push({
          key: `attack:${actor}:${card.uid}:${target.uid}`,
          kind: "attack",
          uid: card.uid,
          targetUid: target.uid,
          cardId: card.cardId,
          label: `${cardName(card.cardId)}攻擊${target.cardId ? cardName(target.cardId) : "主戰者"}`,
        });
      } else {
        actions.push({
          key: `activate:${actor}:${action.id}:${card.uid}`,
          kind: "activate",
          uid: card.uid,
          abilityId: action.id,
          cardId: card.cardId,
          label: `${action.label}：${cardName(card.cardId)}`,
        });
      }
    }
  }
  if (actor === "ai") {
    for (const activation of manualAiActivations(state)) actions.push({
      key: `activate:ai:${activation.key}`,
      kind: "activate",
      uid: activation.sourceUid,
      abilityId: activation.abilityId,
      activationKey: activation.key,
      selected: activation.selected,
      cardId: state.ai.field.find((item) => item.uid === activation.sourceUid)?.cardId ?? "zelgenea",
      label: activation.label,
    });
  }
  actions.push({ key: `end:${actor}`, kind: "end", label: `${actor === "player" ? "玩家" : "破壞方"}結束回合` });
  return actions;
}

/** 執行一個由 trainingLegalActions 產生的動作；傳入過期或非法動作會直接報錯。 */
export function applyTrainingAction(input: GameState, action: TrainingAction): GameState {
  const legal = trainingLegalActions(input).find((candidate) => candidate.key === action.key);
  if (!legal) throw new Error(`Illegal or stale training action: ${action.key}`);
  const actor = trainingActor(input);
  if (!actor) throw new Error("No actor is waiting for a training action");
  switch (legal.kind) {
    case "mulligan":
      return input.aiControl === "manual"
        ? finishManualMulligan(input, actor, legal.redraw)
        : finishMulligan(input, legal.redraw);
    case "choice":
      return resolveChoice(input, legal.selected);
    case "play":
      return playCardForSide(input, actor, legal.uid, legal.zone as Zone);
    case "attack":
      return attackCardForSide(input, actor, legal.uid, legal.targetUid);
    case "activate":
      if (actor === "ai" && legal.activationKey) {
        const activation = manualAiActivations(input).find((candidate) => candidate.key === legal.activationKey);
        if (!activation) throw new Error(`Missing manual AI activation: ${legal.activationKey}`);
        return activateManualAi(input, activation);
      }
      return activateFieldCard(input, legal.uid, legal.abilityId);
    case "evolve":
      return evolveCardForSide(input, actor, legal.uid, legal.payment, legal.superEvolve);
    case "end":
      return endTurnForSide(input, actor);
  }
}

export function trainingReward(state: GameState): number {
  if (state.status !== "gameover") return 0;
  if (state.winner === "player") return 1;
  if (state.winner === "ai") return -1;
  return 0;
}

export function trainingRewardForSide(state: GameState, side: Side): number {
  const reward = trainingReward(state);
  return side === "player" ? reward : -reward;
}

function assertReplayMatchesState(replay: TrainingReplay, state: GameState): void {
  if (replay.engineVersion !== state.version) throw new Error("Replay engine version does not match game state");
  if (replay.seed !== state.seed || replay.playerFirst !== state.playerFirst || replay.playerDeck !== state.playerDeck) {
    throw new Error("Replay metadata does not match game state");
  }
  if ((replay.aiControl ?? "scripted") !== state.aiControl) throw new Error("Replay control mode does not match game state");
}

/** 從剛建立、尚未換牌的局面開始錄影。回放只存動作，不存隱藏資訊。 */
export function createTrainingReplay(state: GameState): TrainingReplay {
  if (state.status !== "mulligan" || state.events.length !== 1) {
    throw new Error("A training replay must start from a fresh mulligan state");
  }
  return {
    format: "shadowverse-pt-training-replay",
    replayVersion: 1,
    engineVersion: state.version,
    seed: state.seed,
    playerFirst: state.playerFirst,
    playerDeck: state.playerDeck,
    aiControl: state.aiControl,
    decisions: [],
  };
}

/**
 * 記錄落子前的合法操作集合與模型判斷。函式回傳新物件，方便批次訓練平行寫入。
 * actor 由當下待決策方決定；雙策略模式會同時記錄 player 與 ai 的顯式決策。
 */
export function recordTrainingDecision(
  replay: TrainingReplay,
  state: GameState,
  action: TrainingAction,
  audit?: TrainingDecisionAudit,
): TrainingReplay {
  assertReplayMatchesState(replay, state);
  if (replay.result) throw new Error("Cannot append to a finalized replay");
  const legal = trainingLegalActions(state);
  const selected = legal.find((candidate) => candidate.key === action.key);
  if (!selected) throw new Error(`Cannot record illegal or stale action: ${action.key}`);
  const actor = trainingActor(state);
  if (!actor) throw new Error("Cannot record a decision without an actor");
  if (audit?.policy) {
    const legalKeys = new Set(legal.map((candidate) => candidate.key));
    for (const candidate of audit.policy) {
      if (!legalKeys.has(candidate.actionKey)) throw new Error(`Policy contains illegal action: ${candidate.actionKey}`);
      if (!Number.isFinite(candidate.probability) || candidate.probability < 0 || candidate.probability > 1) {
        throw new Error(`Invalid policy probability for ${candidate.actionKey}`);
      }
    }
  }
  return {
    ...replay,
    decisions: [...replay.decisions, {
      index: replay.decisions.length,
      actor,
      globalTurn: state.globalTurn,
      ownTurn: state[actor].ownTurn,
      phase: state.phase,
      actionKey: selected.key,
      actionLabel: selected.label,
      legalActionKeys: legal.map((candidate) => candidate.key),
      audit: audit ? structuredClone(audit) : undefined,
    }],
  };
}

export function finalizeTrainingReplay(replay: TrainingReplay, state: GameState): TrainingReplay {
  assertReplayMatchesState(replay, state);
  return {
    ...replay,
    result: {
      status: state.status,
      winner: state.winner,
      reward: trainingReward(state),
      globalTurn: state.globalTurn,
      playerHp: state.player.hp,
      aiHp: state.ai.hp,
      eventCount: state.events.length,
    },
  };
}

/**
 * 從 seed 重建每個 decision 前後的完整 GameState。若合法操作集合或結果不同，立即拒絕舊回放。
 * states[0] 是起始換牌畫面；最後一格是重播完成後的局面。
 */
export function replayTrainingGame(replay: TrainingReplay): { states: GameState[]; finalState: GameState } {
  if (replay.format !== "shadowverse-pt-training-replay" || replay.replayVersion !== 1) {
    throw new Error("Unsupported training replay format");
  }
  let state = createGame(replay.playerFirst, replay.seed, replay.playerDeck, { aiControl: replay.aiControl ?? "scripted" });
  if (state.version !== replay.engineVersion) throw new Error("Replay was recorded with a different engine version");
  const states = [state];
  for (const [index, decision] of replay.decisions.entries()) {
    if (decision.index !== index) throw new Error(`Replay decision index mismatch at ${index}`);
    const legal = trainingLegalActions(state);
    const legalKeys = legal.map((candidate) => candidate.key);
    if (JSON.stringify(legalKeys) !== JSON.stringify(decision.legalActionKeys)) {
      throw new Error(`Replay legal actions diverged at decision ${index}`);
    }
    const action = legal.find((candidate) => candidate.key === decision.actionKey);
    if (!action) throw new Error(`Replay action is no longer legal at decision ${index}: ${decision.actionKey}`);
    state = applyTrainingAction(state, action);
    states.push(state);
  }
  if (replay.result) {
    const actual: TrainingReplayResult = {
      status: state.status,
      winner: state.winner,
      reward: trainingReward(state),
      globalTurn: state.globalTurn,
      playerHp: state.player.hp,
      aiHp: state.ai.hp,
      eventCount: state.events.length,
    };
    if (JSON.stringify(actual) !== JSON.stringify(replay.result)) throw new Error("Replay final result diverged");
  }
  return { states, finalState: state };
}
