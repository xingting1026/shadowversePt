import {
  AI_DECK,
  AI_EVOLVE,
  CARDS,
  PLAYER_DECKS,
  cardName,
  isAlbert,
  isFairyCard,
  isIdolCard,
  isLevinCard,
  isLishenna,
  isRoyalCard,
  type CardDef,
  type PlayerDeckId,
  type Side,
} from "./cards";

export type Zone = "deck" | "hand" | "field" | "ex" | "grave" | "banished";

export type CardInstance = {
  uid: string;
  cardId: string;
  baseCardId?: string;
  owner: Side;
  zone: Zone;
  tapped: boolean;
  damage: number;
  attackBuff: number;
  healthBuff: number;
  enteredAt: number;
  evolvedThisTurn: boolean;
  tempStorm: boolean;
  tempRush: boolean;
  tempDesignated: boolean;
  flags: Record<string, boolean | number>;
};

export type PlayerState = {
  hp: number;
  maxPP: number;
  pp: number;
  ep: number;
  sep: number;
  ownTurn: number;
  deck: CardInstance[];
  hand: CardInstance[];
  field: CardInstance[];
  ex: CardInstance[];
  grave: CardInstance[];
  banished: CardInstance[];
  evolveRemaining: Record<string, number>;
  evolveUsed: Record<string, number>;
};

export type Task = {
  type: string;
  side?: Side;
  sourceUid?: string;
  cardId?: string;
  amount?: number;
  cardIds?: string[];
  data?: Record<string, unknown>;
  label?: string;
};

export type ChoiceOption = {
  uid: string;
  cardId?: string;
  label?: string;
  description?: string;
};

export type PendingChoice = {
  kind: "single" | "multi" | "order" | "yesNo" | "triggerOrder";
  effect: string;
  title: string;
  prompt: string;
  options: ChoiceOption[];
  min: number;
  max: number;
  allowCancel?: boolean;
  /** 需要回答這個選擇的玩家；舊的互動選擇預設為 player。 */
  side?: Side;
  data?: Record<string, unknown>;
};

export type AiControl = "scripted" | "manual";
export const GAME_ENGINE_VERSION = 3 as const;

export type GameOptions = {
  aiControl?: AiControl;
};

export type MatchEvent = {
  seq: number;
  turn: number;
  ownTurn: number;
  side: Side | "system";
  type: string;
  cardId?: string;
  detail?: string;
  hpPlayer: number;
  hpAi: number;
  pp?: number;
};

export type GameState = {
  version: 3;
  seed: number;
  gameId: string;
  rng: number;
  uidCounter: number;
  status: "mulligan" | "playing" | "gameover";
  winner?: Side | "draw";
  playerFirst: boolean;
  playerDeck: PlayerDeckId;
  aiControl: AiControl;
  mulliganDone: Record<Side, boolean>;
  turnSide: Side;
  globalTurn: number;
  phase: "setup" | "main" | "end" | "ai";
  player: PlayerState;
  ai: PlayerState;
  playedThisTurn: number;
  evolvedThisTurn: boolean;
  tasks: Task[];
  pending?: PendingChoice;
  log: string[];
  lastAction?: string;
  events: MatchEvent[];
};

export type CardAction = {
  id: string;
  label: string;
  enabled: boolean;
  reason?: string;
  payment?: "pp" | "ep";
  superEvolve?: boolean;
};

const otherSide = (side: Side): Side => (side === "player" ? "ai" : "player");
const ps = (state: GameState, side: Side): PlayerState => state[side];
const clone = (state: GameState): GameState => structuredClone(state);

/**
 * 把對外可讀、可重播的 32-bit seed 擴散成 PRNG 初始狀態。
 * xorshift32 不適合直接接收連續整數；先做 avalanche 可避免鄰近 seed 產生高度相關的洗牌。
 */
function mixedRngSeed(seed: number): number {
  let value = seed >>> 0;
  value ^= value >>> 16;
  value = Math.imul(value, 0x7feb352d);
  value ^= value >>> 15;
  value = Math.imul(value, 0x846ca68b);
  value ^= value >>> 16;
  return (value || 0x9e3779b9) >>> 0;
}

function nextRandom(state: GameState): number {
  let x = state.rng | 0;
  x ^= x << 13;
  x ^= x >>> 17;
  x ^= x << 5;
  state.rng = x >>> 0;
  return state.rng / 0x100000000;
}

function makeInstance(state: GameState, cardId: string, owner: Side, zone: Zone): CardInstance {
  state.uidCounter += 1;
  return {
    uid: `${owner}-${state.uidCounter}`,
    cardId,
    owner,
    zone,
    tapped: false,
    damage: 0,
    attackBuff: 0,
    healthBuff: 0,
    enteredAt: -1,
    evolvedThisTurn: false,
    tempStorm: false,
    tempRush: false,
    tempDesignated: false,
    flags: {},
  };
}

function expandDeck(state: GameState, side: Side, list: [string, number][]): CardInstance[] {
  return list.flatMap(([cardId, count]) =>
    Array.from({ length: count }, () => makeInstance(state, cardId, side, "deck")),
  );
}

function shuffle<T>(state: GameState, items: T[]): void {
  for (let index = items.length - 1; index > 0; index -= 1) {
    const picked = Math.floor(nextRandom(state) * (index + 1));
    [items[index], items[picked]] = [items[picked], items[index]];
  }
}

function evolveMap(list: [string, number][]): Record<string, number> {
  return Object.fromEntries(list);
}

function emptyPlayer(evolve: [string, number][]): PlayerState {
  return {
    hp: 20,
    maxPP: 0,
    pp: 0,
    ep: 0,
    sep: 1,
    ownTurn: 0,
    deck: [],
    hand: [],
    field: [],
    ex: [],
    grave: [],
    banished: [],
    evolveRemaining: evolveMap(evolve),
    evolveUsed: {},
  };
}

function addLog(state: GameState, text: string): void {
  state.log.unshift(text);
  state.log = state.log.slice(0, 120);
  state.lastAction = text;
}

function addEvent(
  state: GameState,
  side: Side | "system",
  type: string,
  extra: { cardId?: string; detail?: string; pp?: number } = {},
): void {
  state.events.push({
    seq: state.events.length,
    turn: state.globalTurn,
    ownTurn: side === "system" ? state.globalTurn : ps(state, side).ownTurn,
    side,
    type,
    hpPlayer: state.player.hp,
    hpAi: state.ai.hp,
    ...extra,
  });
}

function drawOne(state: GameState, side: Side): CardInstance | undefined {
  const player = ps(state, side);
  const top = player.deck.pop();
  if (!top) {
    state.status = "gameover";
    state.winner = otherSide(side);
    addLog(state, `${side === "player" ? "你" : "破壞巫"}無牌可抽，敗北。`);
    addEvent(state, "system", "gameover", { detail: `winner=${state.winner} deckout=${side}` });
    return undefined;
  }
  top.zone = "hand";
  player.hand.push(top);
  return top;
}

function drawCards(state: GameState, side: Side, count: number): void {
  for (let i = 0; i < count && state.status !== "gameover"; i += 1) drawOne(state, side);
}

function findInField(state: GameState, uid: string): CardInstance | undefined {
  return [...state.player.field, ...state.ai.field].find((item) => item.uid === uid);
}

function removeFromZone(player: PlayerState, zone: Zone, uid: string): CardInstance | undefined {
  const items = player[zone] as CardInstance[];
  const index = items.findIndex((item) => item.uid === uid);
  if (index < 0) return undefined;
  return items.splice(index, 1)[0];
}

function findOwned(state: GameState, side: Side, uid: string): { card: CardInstance; zone: Zone } | undefined {
  const player = ps(state, side);
  const zones: Zone[] = ["hand", "field", "ex", "grave", "banished", "deck"];
  for (const zone of zones) {
    const found = player[zone].find((item) => item.uid === uid);
    if (found) return { card: found, zone };
  }
  return undefined;
}

export function definition(card: CardInstance | string): CardDef {
  return CARDS[typeof card === "string" ? card : card.cardId];
}

export function attackOf(card: CardInstance): number {
  return Math.max(0, (definition(card).attack ?? 0) + card.attackBuff);
}

export function maxHealthOf(card: CardInstance): number {
  return Math.max(0, (definition(card).health ?? 0) + card.healthBuff);
}

export function remainingHealthOf(card: CardInstance): number {
  return maxHealthOf(card) - card.damage;
}

export function isFollower(card: CardInstance): boolean {
  return definition(card).kind === "follower";
}

export function isAmulet(card: CardInstance): boolean {
  return definition(card).kind === "amulet";
}

export function hasKeyword(state: GameState, card: CardInstance, keyword: "ward" | "storm" | "rush" | "designated"): boolean {
  if (definition(card).keywords?.includes(keyword)) return true;
  if (keyword === "storm" && card.tempStorm) return true;
  if (keyword === "rush" && card.tempRush) return true;
  if (keyword === "designated" && card.tempDesignated) return true;
  if (keyword === "storm" && card.flags.permStorm) return true;
  if (card.owner === "player" && keyword === "storm" && card.cardId === "levinAxeGeno" && graveLevin(state) >= 5) return true;
  if (card.owner === "player" && keyword === "designated" && card.cardId === "levinMeim" && graveLevin(state) >= 5) return true;
  if (card.owner === "player" && definition(card).token && isFairyCard(card.cardId)) {
    if (keyword === "rush" && miasmaActive(state)) return true;
    if (keyword === "designated" && state.player.field.some((item) => item.cardId === "fairyland")) return true;
    if (keyword === "storm" && state.player.field.some((item) => item.cardId === "queenCynthia")) return true;
    if (keyword === "designated" && state.player.field.some((item) => item.cardId === "queenCynthia")) return true;
  }
  return false;
}

function graveLevin(state: GameState): number {
  return state.player.grave.filter((item) => isLevinCard(item.cardId)).length;
}

function graveRoyalFollowers(state: GameState): number {
  return state.player.grave.filter((item) => isRoyalCard(item.cardId) && isFollower(item)).length;
}

function discardPlayerCard(state: GameState, uid: string, reason = "效果"): CardInstance | undefined {
  const card = removeFromZone(state.player, "hand", uid);
  if (!card) return undefined;
  card.zone = "grave";
  if (!cardIsToken(card)) state.player.grave.push(card);
  addLog(state, `你捨棄${cardName(card.cardId)}（${reason}）。`);
  if (card.cardId === "levinAxeGeno") queueFront(state, { type: "genoDig", side: "player", label: "レヴィオンアックス・ジェノ的捨棄能力" });
  return card;
}

function miasmaActive(state: GameState): boolean {
  return [...state.player.field, ...state.player.ex].some((item) =>
    item.cardId === "miasmaAria" || item.cardId === "miasmaAriaEvo",
  );
}

function fieldFairyTokens(state: GameState, side: Side = "player"): CardInstance[] {
  return ps(state, side).field.filter((item) => definition(item).token && isFairyCard(item.cardId) && isFollower(item));
}

function exFairyFollowers(state: GameState, side: Side = "player"): CardInstance[] {
  return ps(state, side).ex.filter((item) => isFairyCard(item.cardId) && isFollower(item));
}

function exFairyCards(state: GameState, side: Side = "player"): CardInstance[] {
  return ps(state, side).ex.filter((item) => isFairyCard(item.cardId));
}

function idolField(state: GameState, side: Side = "ai"): CardInstance[] {
  return ps(state, side).field.filter((item) => isIdolCard(item.cardId));
}

function eggField(state: GameState): CardInstance[] {
  return state.ai.field.filter((item) => item.cardId === "newWhite" || item.cardId === "newBlack");
}

function canFitField(state: GameState, side: Side): boolean {
  return ps(state, side).field.length < 5;
}

function cardIsToken(card: CardInstance): boolean {
  return Boolean(definition(card).token);
}

function baseCardId(card: CardInstance): string {
  return card.baseCardId ?? card.cardId;
}

function checkWinner(state: GameState): void {
  if (state.player.hp <= 0 && state.ai.hp <= 0) {
    state.status = "gameover";
    state.winner = "draw";
  } else if (state.player.hp <= 0) {
    state.status = "gameover";
    state.winner = "ai";
  } else if (state.ai.hp <= 0) {
    state.status = "gameover";
    state.winner = "player";
  }
  if (state.status === "gameover") addEvent(state, "system", "gameover", { detail: `winner=${state.winner}` });
}

function healLeader(state: GameState, side: Side, amount: number): void {
  ps(state, side).hp += amount;
  addLog(state, `${side === "player" ? "你" : "破壞巫"}回復${amount}點體力。`);
}

function damageLeader(state: GameState, side: Side, amount: number): void {
  ps(state, side).hp -= amount;
  addLog(state, `${side === "player" ? "你" : "破壞巫"}受到${amount}點傷害。`);
  checkWinner(state);
}

function cleanupEvolveCard(state: GameState, card: CardInstance): void {
  if (!card.baseCardId) return;
  card.cardId = card.baseCardId;
  delete card.baseCardId;
  card.evolvedThisTurn = false;
  card.attackBuff = 0;
  card.healthBuff = 0;
  card.damage = 0;
}

function moveCardToBanished(state: GameState, side: Side, card: CardInstance, from: Zone): void {
  removeFromZone(ps(state, side), from, card.uid);
  cleanupEvolveCard(state, card);
  card.zone = "banished";
  card.tapped = false;
  card.damage = 0;
  card.attackBuff = 0;
  card.healthBuff = 0;
  card.tempStorm = false;
  card.tempRush = false;
  card.tempDesignated = false;
  if (!cardIsToken(card)) ps(state, side).banished.push(card);
}

function queue(state: GameState, ...tasks: Task[]): void {
  state.tasks.push(...tasks);
}

function queueFront(state: GameState, ...tasks: Task[]): void {
  state.tasks.unshift(...tasks);
}

function groupNewPlayerTasks(state: GameState, taskCountBefore: number, label: string): void {
  const added = state.tasks.length - taskCountBefore;
  if (added <= 1) return;
  const simultaneous = state.tasks.splice(0, added);
  queueFront(state, { type: "triggerGroup", side: "player", data: { tasks: simultaneous }, label });
}

function moveFieldToGrave(state: GameState, card: CardInstance, reason = "效果"): void {
  const owner = card.owner;
  const player = ps(state, owner);
  if (!removeFromZone(player, "field", card.uid)) return;
  const leavingId = card.cardId;
  const mainId = baseCardId(card);
  const wasToken = cardIsToken({ ...card, cardId: mainId });
  cleanupEvolveCard(state, card);
  card.zone = "grave";
  card.tapped = false;
  card.damage = 0;
  card.attackBuff = 0;
  card.healthBuff = 0;
  card.tempStorm = false;
  card.tempRush = false;
  card.tempDesignated = false;
  if (!wasToken) player.grave.push(card);
  addLog(state, `${cardName(mainId)}因${reason}離場。`);

  if (mainId === "pureWaterFairy") queueFront(state, { type: "addEx", side: owner, cardIds: ["fairy"], label: "純粋なるウォーターフェアリー的謝幕曲" });
  if (mainId === "fairyDragon") queueFront(state, { type: "addEx", side: owner, cardIds: ["fairyWisp", "fairy"], label: "フェアリードラゴン的謝幕曲" });
  if (mainId === "miasmaAria") {
    queueFront(state, { type: "miasmaLastWord", side: owner, sourceUid: card.uid, cardId: mainId, label: "瘴気の妖精姫・アリア的謝幕曲" });
  }
  if (leavingId === "newWhite") {
    queueFront(state,
      { type: "spawn", side: owner, cardId: "newBlack", label: "新約・白の章的謝幕曲" },
      { type: "heal", side: owner, amount: 1 },
    );
  }
  if (leavingId === "newBlack") {
    queueFront(state,
      { type: "spawn", side: owner, cardId: "newWhite", label: "新約・黒の章的謝幕曲" },
      { type: "leaderDamage", side: otherSide(owner), amount: 1 },
    );
  }
}

function destroyFollower(state: GameState, card: CardInstance, reason = "破壞"): void {
  if (!findInField(state, card.uid)) return;
  moveFieldToGrave(state, card, reason);
}

function dealDamageToFollower(state: GameState, card: CardInstance, amount: number, source = "效果"): void {
  if (!findInField(state, card.uid)) return;
  card.damage += amount;
  addLog(state, `${cardName(card.cardId)}受到${amount}點傷害（${source}）。`);
  if (remainingHealthOf(card) <= 0) destroyFollower(state, card, "傷害");
}

function putExistingIntoField(state: GameState, card: CardInstance, side: Side, triggerFanfare = true): boolean {
  if (!canFitField(state, side)) {
    addLog(state, `場上已滿，${cardName(card.cardId)}無法進場。`);
    return false;
  }
  card.owner = side;
  card.zone = "field";
  card.tapped = false;
  card.enteredAt = state.globalTurn;
  card.damage = 0;
  card.evolvedThisTurn = false;
  card.tempStorm = false;
  card.tempRush = false;
  card.tempDesignated = false;
  ps(state, side).field.push(card);

  if (side === "player" && definition(card).token && isFairyCard(card.cardId)) {
    const fairylands = state.player.field.filter((item) => item.cardId === "fairyland").length;
    const cynthias = state.player.field.filter((item) => item.cardId === "queenCynthia").length;
    card.attackBuff += fairylands + cynthias * 2;
    card.healthBuff += fairylands;
  }
  if (side === "ai" && isAmulet(card) && isIdolCard(card.cardId)) {
    for (const axia of state.ai.field.filter((item) => item.cardId === "axiaEvo" && !item.flags.axiaTriggered)) {
      axia.flags.axiaTriggered = true;
      queue(state, { type: "aiDamageBestFollower", side: "ai", sourceUid: axia.uid, amount: 2, label: "アクシア的每回合一次傷害" });
    }
  }
  if (side === "player" && state.turnSide === "player" && isFollower(card) && isAlbert(card.cardId)) {
    for (const runes of state.player.field.filter((item) => item.cardId === "levinRunesEvo" && item.uid !== card.uid)) {
      queue(state, { type: "runesSnipe", side: "player", sourceUid: runes.uid, label: "ルネス（EVOLVE）的アルベール進場效果" });
    }
  }
  if (triggerFanfare) {
    const fanfare: Task = { type: "fanfare", side, sourceUid: card.uid, cardId: card.cardId, label: `${cardName(card.cardId)}的入場曲` };
    if (side === "player" && isFollower(card) && hasKeyword(state, card, "ward")) {
      queue(state, { type: "wardOnEntry", side, sourceUid: card.uid, data: { tasks: [fanfare] }, label: `${cardName(card.cardId)}的守護進場選擇` });
    } else queue(state, fanfare);
  }
  return true;
}

function spawnToken(state: GameState, side: Side, cardId: string): CardInstance | undefined {
  if (!canFitField(state, side)) {
    addLog(state, `場上已滿，${cardName(cardId)}無法進場。`);
    return undefined;
  }
  const card = makeInstance(state, cardId, side, "field");
  putExistingIntoField(state, card, side, false);
  addLog(state, `${side === "player" ? "你" : "破壞巫"}的場上出現${cardName(cardId)}。`);
  return card;
}

function addExDirect(state: GameState, side: Side, cardIds: string[]): void {
  const player = ps(state, side);
  const capacity = Math.max(0, 5 - player.ex.length);
  if (capacity <= 0) {
    addLog(state, `${side === "player" ? "你" : "破壞巫"}的EX區已滿，無法放入新卡。`);
    return;
  }
  const accepted = cardIds.slice(0, capacity);
  for (const cardId of accepted) {
    const instance = makeInstance(state, cardId, side, "ex");
    player.ex.push(instance);
    addLog(state, `${cardName(cardId)}放入${side === "player" ? "你的" : "破壞巫的"}EX區。`);
  }
}

function actionSource(state: GameState, uid?: string): CardInstance | undefined {
  if (!uid) return undefined;
  return findInField(state, uid) ?? state.player.grave.find((item) => item.uid === uid) ?? state.ai.grave.find((item) => item.uid === uid);
}

function isSuperEligible(state: GameState, side: Side): boolean {
  const turn = ps(state, side).ownTurn;
  const first = side === "player" ? state.playerFirst : !state.playerFirst;
  return turn >= (first ? 7 : 6) && ps(state, side).sep > 0;
}

function createSide(state: GameState, side: Side, list: [string, number][], evolve: [string, number][]): PlayerState {
  const player = emptyPlayer(evolve);
  state[side] = player;
  player.deck = expandDeck(state, side, list);
  shuffle(state, player.deck);
  drawCards(state, side, 4);
  return player;
}

export function createGame(
  playerFirst: boolean,
  seed = Date.now(),
  playerDeck: PlayerDeckId = "fairy",
  options: GameOptions = {},
): GameState {
  const deck = PLAYER_DECKS[playerDeck];
  const state: GameState = {
    version: GAME_ENGINE_VERSION,
    seed: seed >>> 0,
    gameId: `${(seed >>> 0).toString(36)}-${Date.now().toString(36)}`,
    rng: mixedRngSeed(seed),
    uidCounter: 0,
    status: "mulligan",
    playerFirst,
    playerDeck,
    aiControl: options.aiControl ?? "scripted",
    mulliganDone: { player: false, ai: false },
    turnSide: playerFirst ? "player" : "ai",
    globalTurn: 0,
    phase: "setup",
    player: emptyPlayer(deck.evolve),
    ai: emptyPlayer(AI_EVOLVE),
    playedThisTurn: 0,
    evolvedThisTurn: false,
    tasks: [],
    log: [],
    events: [],
  };
  createSide(state, "player", deck.main, deck.evolve);
  createSide(state, "ai", AI_DECK, AI_EVOLVE);
  state.player.ep = playerFirst ? 0 : 3;
  state.ai.ep = playerFirst ? 3 : 0;
  addLog(state, `遊戲種子：${state.seed}。你使用${deck.label}，選擇${playerFirst ? "先攻" : "後攻"}。`);
  addEvent(state, "system", "gameStart", { detail: `seed=${state.seed} playerFirst=${playerFirst} deck=${playerDeck}` });
  return state;
}

function aiWantsMulligan(state: GameState): boolean {
  const hand = state.ai.hand.map((item) => item.cardId);
  // 「低費」必須是能上場的東西（從者/護符）或能生蛋的殲滅の歌声；愉悦、章這種不算數，
  // 否則會留下第1、2回合只能空過的起手。
  const hasEarly = hand.some((id) => {
    const def = CARDS[id];
    if (id === "annihilationSong") return true;
    return def.cost <= 2 && (def.kind === "follower" || def.kind === "amulet") && id !== "returningDissonance";
  });
  const hasEngine = hand.some((id) => ["destructionWilderness", "manifestedLishenna", "dissonanceWorshipper", "annihilationSong"].includes(id));
  return !hasEarly || !hasEngine;
}

function redrawHand(state: GameState, side: Side): void {
  const player = ps(state, side);
  const old = player.hand.splice(0);
  shuffle(state, old);
  for (const card of old) {
    card.zone = "deck";
    player.deck.unshift(card);
  }
  drawCards(state, side, 4);
}

export function finishMulligan(input: GameState, redraw: boolean): GameState {
  const state = clone(input);
  if (state.status !== "mulligan" || state.aiControl === "manual") return state;
  if (redraw) {
    redrawHand(state, "player");
    addLog(state, "你將起手4張全部放到牌庫底並重抽。―");
  } else {
    addLog(state, "你保留起手。―");
  }
  addEvent(state, "player", "mulligan", { detail: redraw ? "redraw" : "keep" });
  if (aiWantsMulligan(state)) {
    redrawHand(state, "ai");
    addLog(state, "破壞巫選擇重抽起手。―");
    addEvent(state, "ai", "mulligan", { detail: "redraw" });
  } else {
    addLog(state, "破壞巫保留起手。―");
    addEvent(state, "ai", "mulligan", { detail: "keep" });
  }
  state.status = "playing";
  state.phase = "main";
  beginTurnMutable(state, state.turnSide);
  runTasks(state);
  if (state.turnSide === "ai" && !state.pending) runAiTurnMutable(state);
  return state;
}

function logMulliganChoice(state: GameState, side: Side, redraw: boolean): void {
  if (redraw) {
    redrawHand(state, side);
    addLog(state, `${side === "player" ? "你" : "破壞巫"}將起手4張全部放到牌庫底並重抽。―`);
  } else {
    addLog(state, `${side === "player" ? "你" : "破壞巫"}保留起手。―`);
  }
  addEvent(state, side, "mulligan", { detail: redraw ? "redraw" : "keep" });
  state.mulliganDone[side] = true;
}

/**
 * 雙策略對局使用的逐方換牌。player 先提交，ai 再提交；雙方都完成前不開始第1回合，
 * 因此第二位決策者仍看不到對方手牌內容。
 */
export function finishManualMulligan(input: GameState, side: Side, redraw: boolean): GameState {
  const state = clone(input);
  if (state.status !== "mulligan" || state.aiControl !== "manual") return state;
  const expected: Side = state.mulliganDone.player ? "ai" : "player";
  if (side !== expected || state.mulliganDone[side]) return state;
  logMulliganChoice(state, side, redraw);
  if (!state.mulliganDone.player || !state.mulliganDone.ai) return state;
  state.status = "playing";
  state.phase = "main";
  beginTurnMutable(state, state.turnSide);
  runTasks(state);
  return state;
}

function beginTurnMutable(state: GameState, side: Side): void {
  if (state.status !== "playing") return;
  state.globalTurn += 1;
  state.turnSide = side;
  state.phase = side === "player" || state.aiControl === "manual" ? "main" : "ai";
  state.playedThisTurn = 0;
  state.evolvedThisTurn = false;
  for (const card of [...state.player.field, ...state.ai.field]) {
    delete card.flags.axiaTriggered;
    delete card.flags.freeEvolve;
    delete card.flags.albertRestandUsed;
  }
  const player = ps(state, side);
  player.ownTurn += 1;
  player.maxPP = Math.min(10, player.maxPP + 1);
  player.pp = player.maxPP;
  for (const card of player.field) {
    card.tapped = false;
    card.evolvedThisTurn = false;
    card.tempStorm = false;
    card.tempRush = false;
    card.tempDesignated = false;
  }
  const isFirstPlayersFirstTurn = player.ownTurn === 1 && ((side === "player" && state.playerFirst) || (side === "ai" && !state.playerFirst));
  if (!isFirstPlayersFirstTurn) drawCards(state, side, 1);
  addLog(state, `${side === "player" ? "你的" : "破壞巫的"}第${player.ownTurn}回合開始（PP ${player.pp}/${player.maxPP}）。`);
  addEvent(state, side, "turnStart", {
    pp: player.pp,
    detail: `hand=${player.hand.length} field=${player.field.length}${side === "ai" ? ` cards=${player.hand.map((item) => item.cardId).join(",")}` : ""}`,
  });
  if (side === "ai" && state.aiControl === "scripted") {
    for (const card of player.field) {
      if (card.cardId === "whiteArtifact") queue(state, { type: "heal", side: "ai", amount: 2 });
      if (card.cardId === "blackArtifact") queue(state, { type: "leaderDamage", side: "player", amount: 2 });
    }
  }
}

function bestFollower(state: GameState, side: Side, mode: "kill" | "threat" = "threat"): CardInstance | undefined {
  // 【オーラ】不能成為對方效果的目標（此函式僅供破壞巫選擇玩家目標用）。
  const followers = ps(state, side).field.filter(isFollower).filter((item) => !item.flags.aura);
  return [...followers].sort((a, b) => {
    const av = attackOf(a) * 2 + remainingHealthOf(a) + (hasKeyword(state, a, "ward") ? 4 : 0) + (a.cardId === "queenCynthia" ? 8 : 0);
    const bv = attackOf(b) * 2 + remainingHealthOf(b) + (hasKeyword(state, b, "ward") ? 4 : 0) + (b.cardId === "queenCynthia" ? 8 : 0);
    if (mode === "kill") {
      const ak = remainingHealthOf(a) <= 2 ? 10 : 0;
      const bk = remainingHealthOf(b) <= 2 ? 10 : 0;
      return bv + bk - (av + ak);
    }
    return bv - av;
  })[0];
}

function taskLabel(task: Task): string {
  return task.label ?? (task.cardId ? `${cardName(task.cardId)}的效果` : "卡片效果");
}

function setTriggerOrder(state: GameState, tasks: Task[]): void {
  if (tasks.length <= 1) {
    queueFront(state, ...tasks);
    return;
  }
  state.pending = {
    kind: "triggerOrder",
    effect: "triggerOrder",
    title: "選擇效果順序",
    prompt: "這些效果同時觸發。請選擇下一個要先處理的效果。",
    options: tasks.map((task, index) => ({ uid: String(index), label: taskLabel(task) })),
    min: 1,
    max: 1,
    data: { tasks },
  };
}

function addExTask(state: GameState, side: Side, cardIds: string[], after?: Task[]): void {
  const capacity = Math.max(0, 5 - ps(state, side).ex.length);
  if (capacity <= 0) {
    addLog(state, `${side === "player" ? "你的" : "破壞巫的"}EX區已滿，無法放入新卡。`);
    if (after?.length) queueFront(state, ...after);
    return;
  }
  if (cardIds.length <= capacity) {
    addExDirect(state, side, cardIds);
    if (after?.length) queueFront(state, ...after);
    return;
  }
  if (side === "ai") {
    const value: Record<string, number> = { blackArtifact: 9, whiteArtifact: 8, solo: 7, fairyWisp: 7, fairy: 5 };
    const selected = [...cardIds].sort((a, b) => (value[b] ?? 3) - (value[a] ?? 3)).slice(0, capacity);
    addExDirect(state, side, selected);
    if (after?.length) queueFront(state, ...after);
    return;
  }
  state.pending = {
    kind: "multi",
    effect: "addExSubset",
    title: "EX區空間不足",
    prompt: `只能再放${capacity}張。選擇實際放入EX區的卡；其餘不會被生成。`,
    options: cardIds.map((cardId, index) => ({ uid: `new-${index}`, cardId })),
    min: capacity,
    max: capacity,
    side,
    data: { side, cardIds, after: after ?? [] },
  };
}

function topCards(state: GameState, side: Side, count: number): CardInstance[] {
  const cards: CardInstance[] = [];
  for (let i = 0; i < count; i += 1) {
    const top = ps(state, side).deck.pop();
    if (top) cards.push(top);
  }
  return cards;
}

function putBottom(state: GameState, side: Side, cards: CardInstance[]): void {
  for (const card of cards) card.zone = "deck";
  ps(state, side).deck.unshift(...cards);
}

function userTopSearch(state: GameState, cards: CardInstance[], eligible: CardInstance[], effect: string, title: string, max = 1): void {
  state.pending = {
    kind: "multi",
    effect,
    title,
    prompt: max === 1 ? "可以選擇1張加入手牌；也可以不選。之後你會排列其餘牌的牌庫底順序。" : `最多選擇${max}張。`,
    options: cards.map((item) => ({
      uid: item.uid,
      cardId: item.cardId,
      description: eligible.some((hit) => hit.uid === item.uid) ? undefined : "不符合條件",
    })),
    min: 0,
    max,
    data: { cards, eligible: eligible.map((item) => item.uid) },
  };
}

function aiTopSearchToHand(state: GameState, count: number, predicate: (card: CardInstance) => boolean): void {
  const cards = topCards(state, "ai", count);
  const candidates = cards.filter(predicate);
  const priority: Record<string, number> = {
    axia: 18, manifestedLishenna: 17, annihilationSong: 16,
    destructiveLishenna: 15, destructionWilderness: 14,
    dissonanceWorshipper: 13, destructionFanatic: 12,
    originalLishenna: 11, destructionPrayer: 10, whiteBlackChapter: 9,
  };
  const copiesAvailable = (cardId: string): number =>
    [...state.ai.hand, ...state.ai.field].filter((item) => baseCardId(item) === cardId).length;
  // 快被沖死時優先撈解場牌，不撈長線價值牌。
  const dangerBonus = (cardId: string): number => {
    if (!aiInDanger(state)) return 0;
    if (cardId === "destructionFanatic" || cardId === "destructionHermit" || cardId === "whiteBlackChapter" || cardId === "destructionServant") return 8;
    if (cardId === "originalLishenna") return -6;
    return 0;
  };
  const picked = [...candidates].sort((a, b) => {
    const aScore = (priority[a.cardId] ?? 5) - copiesAvailable(a.cardId) * 4 + dangerBonus(a.cardId);
    const bScore = (priority[b.cardId] ?? 5) - copiesAvailable(b.cardId) * 4 + dangerBonus(b.cardId);
    return bScore - aScore;
  })[0];
  if (picked) {
    picked.zone = "hand";
    state.ai.hand.push(picked);
    cards.splice(cards.findIndex((item) => item.uid === picked.uid), 1);
    addLog(state, `破壞巫以效果將${cardName(picked.cardId)}加入手牌。`);
  }
  putBottom(state, "ai", cards);
}

function searchDeckInstances(state: GameState, side: Side, predicate: (card: CardInstance) => boolean): CardInstance[] {
  return ps(state, side).deck.filter(predicate);
}

function removeDeckInstance(state: GameState, side: Side, uid: string): CardInstance | undefined {
  return removeFromZone(ps(state, side), "deck", uid);
}

function shuffleAfterSearch(state: GameState, side: Side, source: string): void {
  shuffle(state, ps(state, side).deck);
  addLog(state, `${source}檢索完成，${side === "player" ? "你" : "破壞巫"}洗牌。`);
}

function beginRunesAlbertSearch(state: GameState): void {
  const candidates = searchDeckInstances(state, "player", (item) => isAlbert(item.cardId));
  if (!candidates.length) {
    addLog(state, "牌庫中已沒有符合條件的『アルベール』[ロイヤル]從者（可以檢索失敗）。");
    shuffleAfterSearch(state, "player", "ルネス");
    return;
  }
  state.pending = {
    kind: "multi",
    effect: "runesAlbertPick",
    title: "レヴィオンの見習い・ルネス",
    prompt: "可以從牌庫選擇1張名稱含『アルベール』的[ロイヤル]從者加入手牌；也可以不選。之後洗牌。",
    options: candidates.map((item) => ({ uid: item.uid, cardId: item.cardId })),
    min: 0,
    max: 1,
  };
}

function beginJusticeSaberSearch(state: GameState): void {
  const candidates = searchDeckInstances(state, "player", (item) => cardName(item.cardId) === "レヴィオンセイバー・アルベール");
  if (!candidates.length || !canFitField(state, "player")) {
    addLog(state, !candidates.length ? "牌庫中沒有『レヴィオンセイバー・アルベール』，追加檢索落空。" : "場上已滿，可以讓レヴィオンセイバー・アルベール的追加檢索落空。");
    shuffleAfterSearch(state, "player", "レヴィオンの正義（追加檢索）");
    return;
  }
  state.pending = {
    kind: "multi",
    effect: "justiceSaberPick",
    title: "レヴィオンの正義（追加檢索）",
    prompt: "可以從牌庫選擇1張レヴィオンセイバー・アルベール放置到場上；也可以不選。之後洗牌。",
    options: candidates.map((item) => ({ uid: item.uid, cardId: item.cardId })),
    min: 0,
    max: 1,
  };
}

function finishJusticeDukeSearch(state: GameState, kicker: boolean, pickedUid?: string): void {
  if (pickedUid && canFitField(state, "player")) {
    const duke = removeDeckInstance(state, "player", pickedUid);
    if (duke?.cardId === "levinDuke") {
      putExistingIntoField(state, duke, "player", true);
      addLog(state, "レヴィオンの正義將レヴィオンデューク・ユリウス放置到場上。");
    }
  } else if (pickedUid) {
    addLog(state, "場上已滿，ユリウス沒有被找出。");
  } else {
    addLog(state, "你選擇讓ユリウス的檢索落空。");
  }
  shuffleAfterSearch(state, "player", "レヴィオンの正義");
  if (kicker) beginJusticeSaberSearch(state);
}

function beginJusticeDukeSearch(state: GameState, kicker: boolean): void {
  const candidates = searchDeckInstances(state, "player", (item) => item.cardId === "levinDuke");
  if (!candidates.length || !canFitField(state, "player")) {
    addLog(state, !candidates.length ? "牌庫中找不到レヴィオンデューク・ユリウス（可以檢索失敗）。" : "場上已滿，可以不找出ユリウス。");
    shuffleAfterSearch(state, "player", "レヴィオンの正義");
    if (kicker) beginJusticeSaberSearch(state);
    return;
  }
  state.pending = {
    kind: "multi",
    effect: "justiceDukePick",
    title: "レヴィオンの正義",
    prompt: "可以從牌庫選擇1張レヴィオンデューク・ユリウス放置到場上；也可以不選。之後洗牌。",
    options: candidates.map((item) => ({ uid: item.uid, cardId: item.cardId })),
    min: 0,
    max: 1,
    data: { kicker },
  };
}

function deploySistersSimultaneously(state: GameState, selectedUids: string[]): void {
  const entering = selectedUids
    .map((uid) => removeDeckInstance(state, "player", uid))
    .filter(Boolean) as CardInstance[];
  const entered: CardInstance[] = [];
  for (const card of entering) {
    if (putExistingIntoField(state, card, "player", false)) {
      entered.push(card);
      addLog(state, `レヴィオンシスターズ登場！將${cardName(card.cardId)}放置到場上。`);
    }
  }
  const triggers: Task[] = entered.map((card) => {
    const fanfare: Task = { type: "fanfare", side: "player", sourceUid: card.uid, cardId: card.cardId, label: `${cardName(card.cardId)}的入場曲` };
    return hasKeyword(state, card, "ward")
      ? { type: "wardOnEntry", side: "player", sourceUid: card.uid, data: { tasks: [fanfare] }, label: `${cardName(card.cardId)}的守護進場選擇` }
      : fanfare;
  });
  if (triggers.length) setTriggerOrder(state, triggers);
}

function resolvePlayerFanfare(state: GameState, source: CardInstance | undefined, cardId: string): void {
  switch (cardId) {
    case "naturalAria":
      addExTask(state, "player", ["fairy", "fairy"]);
      break;
    case "pureWaterFairy":
      addExTask(state, "player", ["fairy"]);
      break;
    case "vistaElf":
      addExTask(state, "player", ["fairyWisp"]);
      break;
    case "fairyDragon": {
      if (!source) break;
      const count = fieldFairyTokens(state).length + state.player.ex.filter((item) => definition(item).token && isFairyCard(item.cardId) && isFollower(item)).length;
      source.attackBuff += count;
      addLog(state, `フェアリードラゴン因${count}張妖精衍生卡獲得+${count}攻擊力。`);
      break;
    }
    case "fairyBladeAmatsu":
      addExTask(state, "player", ["fairy"], [{ type: "buffExFairies", side: "player", amount: 1 }]);
      break;
    case "breathFairyDancer":
      for (const card of state.player.field) {
        if (card.uid !== source?.uid && isFairyCard(card.cardId) && isFollower(card)) {
          card.attackBuff += 1;
          card.healthBuff += 1;
        }
      }
      for (const card of exFairyFollowers(state)) {
        card.attackBuff += 1;
        card.healthBuff += 1;
      }
      addLog(state, "ブレスフェアリーダンサー強化場上其他妖精與EX妖精。 ");
      break;
    case "fairyArcher": {
      spawnToken(state, "player", "fairy");
      const cards = topCards(state, "player", 3);
      userTopSearch(state, cards, cards.filter((item) => isFairyCard(item.cardId)), "fairyArcherPick", "妖精の弓使い：查看牌頂3張");
      break;
    }
    case "miasmaAria":
      addExTask(state, "player", ["fairy"]);
      break;
    case "riotousGarden":
      spawnToken(state, "player", "fairy");
      break;
    case "wonderTree":
      addExTask(state, "player", ["fairy"]);
      break;
    case "bouquetFairy":
      addExTask(state, "player", ["fairyWisp"]);
      break;
    case "wingQueen": {
      if (exFairyCards(state).length < 2) break;
      const choices = searchDeckInstances(state, "player", (item) => {
        const def = definition(item);
        return def.kind === "follower" && isFairyCard(item.cardId) && def.cost <= 3 && item.cardId !== "wingQueen";
      });
      if (!choices.length || !canFitField(state, "player")) break;
      state.pending = {
        kind: "single",
        effect: "wingQueenTutor",
        title: "翅の女王・ティターニア",
        prompt: "選擇牌庫中1張原本費用3以下、不同名的妖精從者放到場上。放進場仍會觸發入場曲。",
        options: choices.map((item) => ({ uid: item.uid, cardId: item.cardId })),
        min: 1,
        max: 1,
      };
      break;
    }
    case "queenCynthia":
      addExTask(state, "player", ["fairyWisp"], [{ type: "buffFieldFairyTokensAttack", side: "player", amount: 2 }]);
      break;
    case "fairyland":
      for (const token of fieldFairyTokens(state)) {
        token.attackBuff += 1;
        token.healthBuff += 1;
      }
      addLog(state, "ティターニアの妖精郷使現有妖精衍生卡+1/+1。 ");
      break;
    case "levinMiim":
    case "levinRunes": {
      const levinHand = state.player.hand.filter((item) => isLevinCard(item.cardId));
      if (!levinHand.length) break;
      state.pending = {
        kind: "multi",
        effect: cardId === "levinMiim" ? "miimDiscard" : "runesDiscard",
        title: cardName(cardId),
        prompt: cardId === "levinMiim"
          ? "可以捨棄1張雷維翁・卡牌：抽1張牌。也可以不捨（直接按確認）。"
          : "可以捨棄1張雷維翁・卡牌：從牌庫找出1張『アルベール』[ロイヤル]從者加入手牌。也可以不捨。",
        options: levinHand.map((item) => ({ uid: item.uid, cardId: item.cardId })),
        min: 0,
        max: 1,
      };
      break;
    }
    case "gawain": {
      if (!source) break;
      const royals = graveRoyalFollowers(state);
      if (royals >= 5) {
        drawCards(state, "player", 1);
        addLog(state, "ガウェイン：墓場[ロイヤル]從者5+，抽1張。 ");
      }
      if (royals >= 10) {
        source.flags.permStorm = true;
        healLeader(state, "player", 2);
        addLog(state, "ガウェイン：10+，獲得【疾走】。 ");
      }
      if (royals >= 15) {
        source.flags.intimidate = true;
        source.flags.aura = true;
        addLog(state, "ガウェイン：15+，獲得【威圧】【オーラ】。 ");
      }
      if (royals >= 20) {
        source.attackBuff += 7;
        source.healthBuff += 7;
        addLog(state, "ガウェイン：20+，+7/+7。 ");
      }
      break;
    }
    case "levinMaim": {
      const targets = state.ai.field.filter(isFollower);
      if (!targets.length) break;
      state.pending = {
        kind: "single",
        effect: "maimTarget",
        title: "レヴィオンの副団長・マイム",
        prompt: "選擇對方1體從者。效果結算時若墓場雷維翁卡牌為5張以上，對其造成3點傷害。",
        options: targets.map((item) => ({ uid: item.uid, cardId: item.cardId })),
        min: 1,
        max: 1,
      };
      break;
    }
    case "levinArcher": {
      const cards = topCards(state, "player", 2);
      if (!cards.length) break;
      state.pending = {
        kind: "multi",
        effect: "archerLevinPick",
        title: "レヴィオンの弓使い",
        prompt: "查看牌頂2張。可以選1張雷維翁・卡牌加入手牌；其餘置入墓場。",
        options: cards.map((item) => ({
          uid: item.uid,
          cardId: item.cardId,
          description: isLevinCard(item.cardId) ? undefined : "不符合條件",
        })),
        min: 0,
        max: 1,
        data: { cards },
      };
      break;
    }
    case "levinTranscend": {
      const levinHand = state.player.hand.filter((item) => isLevinCard(item.cardId));
      if (levinHand.length < 2) break;
      state.pending = {
        kind: "yesNo",
        effect: "transcendRevealChoice",
        title: "レヴィオンの超越者・ユリウス",
        prompt: "要公開手牌中2張雷維翁・卡牌，支付這個可選費用並發動效果嗎？",
        options: [{ uid: "yes", label: "公開2張並發動" }, { uid: "no", label: "不公開、不發動" }],
        min: 1,
        max: 1,
      };
      break;
    }
    case "brutalGeno": {
      const targets = state.ai.field.filter(isFollower);
      if (!targets.length) break;
      state.pending = {
        kind: "single",
        effect: "brutalGenoTarget",
        title: "暴威の武人・ジェノ",
        prompt: "選擇對方1體從者，對其造成4點傷害。",
        options: targets.map((item) => ({ uid: item.uid, cardId: item.cardId })),
        min: 1,
        max: 1,
      };
      break;
    }
    case "levinAlbert": {
      const others = state.player.field.filter((item) => item.uid !== source?.uid && isLevinCard(item.cardId) && isFollower(item));
      if (!others.length) break;
      state.pending = {
        kind: "single",
        effect: "albertBuff",
        title: "レヴィオンの迅雷・アルベール",
        prompt: "選擇自己場上其他1體雷維翁・從者，使其攻擊力+1。",
        options: others.map((item) => ({ uid: item.uid, cardId: item.cardId })),
        min: 1,
        max: 1,
      };
      break;
    }
    default:
      break;
  }
}

function aiSacrificeValue(state: GameState, card: CardInstance): number {
  const value: Record<string, number> = {
    newWhite: state.ai.hp <= 12 ? -2 : 1,
    newBlack: state.player.hp <= 8 ? -3 : 0,
    destructionWilderness: state.ai.field.length >= 5 ? 2 : 5,
    manifestedLishenna: 4,
    destructionHermit: 6,
    dissonanceWorshipper: 7,
    destructionServant: 8,
    destructionFanatic: 10,
    destructionPrayer: card.tapped ? 8 : 13,
    destructiveLishenna: 11,
    originalLishenna: 12,
    whiteArtifact: state.ai.hp <= 12 ? 17 : 12,
    blackArtifact: state.player.hp <= 10 ? 19 : 14,
    axiaEvo: 22,
    destructiveLishennaEvo: 20,
  };
  return value[card.cardId] ?? 9;
}

function aiSacrificeCandidate(state: GameState, excludeUid?: string): CardInstance | undefined {
  return state.ai.field
    .filter((item) => item.uid !== excludeUid && isIdolCard(item.cardId))
    .sort((a, b) => aiSacrificeValue(state, a) - aiSacrificeValue(state, b))[0];
}

function setManualAiTopSearch(
  state: GameState,
  count: number,
  predicate: (card: CardInstance) => boolean,
  title: string,
): void {
  const cards = topCards(state, "ai", count);
  if (!cards.length) return;
  state.pending = {
    kind: "multi",
    effect: "manualAiTopSearch",
    title,
    prompt: `查看牌頂${cards.length}張。可以選1張符合條件的牌加入手牌；其餘以任意順序置於牌庫底。`,
    options: cards.map((item) => ({
      uid: item.uid,
      cardId: item.cardId,
      description: predicate(item) ? undefined : "不符合條件",
    })),
    min: 0,
    max: 1,
    side: "ai",
    data: { cards, eligible: cards.filter(predicate).map((item) => item.uid) },
  };
}

function resolveAiFanfare(state: GameState, source: CardInstance | undefined, cardId: string): void {
  const target = bestFollower(state, "player");
  if (state.aiControl === "manual") {
    switch (cardId) {
      case "destructionHermit": {
        const targets = state.player.field.filter(isFollower);
        if (targets.length) state.pending = {
          kind: "single",
          effect: "manualHermitTarget",
          title: "破壊の隠者",
          prompt: "選擇對方場上1體從者；若偶像卡牌達3張，對其造成3點傷害。",
          options: targets.map((item) => ({ uid: item.uid, cardId: item.cardId })),
          min: 1,
          max: 1,
          side: "ai",
        };
        return;
      }
      case "destructionWilderness":
        setManualAiTopSearch(state, 2, (item) => isIdolCard(item.cardId), "破壊の荒野");
        return;
      case "destructionServant":
        if (source && idolField(state).length >= 3) source.flags.freeEvolve = true;
        return;
      case "dissonanceWorshipper":
        setManualAiTopSearch(state, 3, (item) => isIdolCard(item.cardId), "奏絶の崇拝者");
        return;
      case "manifestedLishenna": {
        const sacrifices = idolField(state).filter((item) => item.uid !== source?.uid);
        const options: ChoiceOption[] = [
          { uid: "white", label: "放置新約・白の章" },
          ...sacrifices.map((item) => ({ uid: item.uid, cardId: item.cardId, label: `破壞${cardName(item.cardId)}並取得独唱` })),
        ];
        state.pending = {
          kind: "single",
          effect: "manualManifestedMode",
          title: "奏絶の顕現・リーシェナ",
          prompt: "選擇放置白蛋，或破壞其他1張偶像卡牌並將独唱置入EX。",
          options,
          min: 1,
          max: 1,
          side: "ai",
          data: { sourceUid: source?.uid },
        };
        return;
      }
      case "destructionFanatic": {
        const targets = state.player.field.filter(isFollower);
        if (idolField(state).length >= 3 && targets.length) state.pending = {
          kind: "single",
          effect: "manualFanaticTarget",
          title: "破壊の狂信者",
          prompt: "選擇要破壞的對方從者。",
          options: targets.map((item) => ({ uid: item.uid, cardId: item.cardId })),
          min: 1,
          max: 1,
          side: "ai",
        };
        return;
      }
      case "destructiveLishenna":
        if (idolField(state).length >= 3) addExTask(state, "ai", ["solo"]);
        return;
      case "originalLishenna":
        addExTask(state, "ai", ["whiteArtifact", "blackArtifact"]);
        return;
      case "zelgenea": {
        const targets = state.player.field.filter(isFollower);
        if (targets.length) state.pending = {
          kind: "single",
          effect: "manualZelgeneaTarget",
          title: "《世界》・ゼルガネイア",
          prompt: "選擇對方場上1體從者；自己沒有其他從者時將其破壞並抽1張。",
          options: targets.map((item) => ({ uid: item.uid, cardId: item.cardId })),
          min: 1,
          max: 1,
          side: "ai",
        };
        if (state.ai.hp <= 10) healLeader(state, "ai", 5);
        return;
      }
      case "greatZelgenea":
        damageLeader(state, "player", 10);
        {
          const taskCountBefore = state.tasks.length;
          for (const follower of [...state.player.field.filter(isFollower)]) dealDamageToFollower(state, follower, 10, "大いなる《世界》");
          groupNewPlayerTasks(state, taskCountBefore, "大いなる《世界》造成的同時離場效果");
        }
        return;
      default:
        return;
    }
  }
  switch (cardId) {
    case "destructionHermit":
      if (target && idolField(state).length >= 3) dealDamageToFollower(state, target, 3, "破壊の隠者");
      break;
    case "destructionWilderness":
      aiTopSearchToHand(state, 2, (item) => isIdolCard(item.cardId));
      break;
    case "destructionServant":
      if (source && idolField(state).length >= 3) source.flags.freeEvolve = true;
      break;
    case "dissonanceWorshipper":
      aiTopSearchToHand(state, 3, (item) => isIdolCard(item.cardId));
      break;
    case "manifestedLishenna": {
      if (!eggField(state).length && canFitField(state, "ai")) {
        spawnToken(state, "ai", "newWhite");
        addLog(state, "破壞巫優先選擇產出『新約・白の章』（白蛋）。");
      } else {
        const sacrifice = aiSacrificeCandidate(state, source?.uid);
        if (!sacrifice || state.ai.ex.length >= 5) break;
        moveFieldToGrave(state, sacrifice, "奏絶の顕現・リーシェナ破壞");
        addExTask(state, "ai", ["solo"]);
      }
      break;
    }
    case "destructionFanatic":
      if (idolField(state).length >= 3 && target) {
        destroyFollower(state, target, "破壊の狂信者");
        damageLeader(state, "player", 2);
        healLeader(state, "ai", 2);
      }
      break;
    case "destructiveLishenna":
      if (idolField(state).length >= 3) addExTask(state, "ai", ["solo"]);
      break;
    case "originalLishenna":
      addExTask(state, "ai", ["whiteArtifact", "blackArtifact"]);
      break;
    case "zelgenea":
      if (target && state.ai.field.filter(isFollower).length === 1) {
        destroyFollower(state, target, "《世界》・ゼルガネイア");
        drawCards(state, "ai", 1);
      }
      if (state.ai.hp <= 10) healLeader(state, "ai", 5);
      break;
    case "greatZelgenea":
      damageLeader(state, "player", 10);
      {
        const taskCountBefore = state.tasks.length;
        for (const follower of [...state.player.field.filter(isFollower)]) dealDamageToFollower(state, follower, 10, "大いなる《世界》");
        groupNewPlayerTasks(state, taskCountBefore, "大いなる《世界》造成的同時離場效果");
      }
      break;
    default:
      break;
  }
}

function resolveFanfare(state: GameState, task: Task): void {
  const source = task.sourceUid ? findInField(state, task.sourceUid) : undefined;
  const cardId = task.cardId ?? source?.cardId;
  if (!cardId) return;
  if ((task.side ?? source?.owner) === "player") resolvePlayerFanfare(state, source, cardId);
  else resolveAiFanfare(state, source, cardId);
}

function resolveTask(state: GameState, task: Task): void {
  const side = task.side ?? "player";
  switch (task.type) {
    case "triggerGroup": {
      const tasks = (task.data?.tasks as Task[]) ?? [];
      if (side === "player") setTriggerOrder(state, tasks);
      else queueFront(state, ...tasks);
      break;
    }
    case "fanfare":
      resolveFanfare(state, task);
      break;
    case "wardOnEntry": {
      const source = task.sourceUid ? findInField(state, task.sourceUid) : undefined;
      const after = (task.data?.tasks as Task[]) ?? [];
      if (!source || source.tapped || !hasKeyword(state, source, "ward")) {
        if (after.length) setTriggerOrder(state, after);
        break;
      }
      state.pending = {
        kind: "yesNo",
        effect: "wardOnEntry",
        title: `${cardName(source.cardId)}：守護`,
        prompt: "守護從者進場時，可以改為以橫置狀態進場。要現在橫置它嗎？之後仍會處理入場曲。",
        options: [{ uid: "yes", label: "以橫置狀態進場" }, { uid: "no", label: "保持直立" }],
        min: 1,
        max: 1,
        data: { sourceUid: source.uid, tasks: after },
      };
      break;
    }
    case "spell":
      resolveSpell(state, task);
      break;
    case "aiAttackStep": {
      const remaining = [...((task.data?.remaining as string[]) ?? [])];
      const finish = Boolean(task.data?.finish);
      let attacker: CardInstance | undefined;
      let targetUid: string | undefined;
      while (remaining.length && !attacker) {
        const uid = remaining.shift() as string;
        const candidate = state.ai.field.find((item) => item.uid === uid);
        const chosenTarget = candidate ? chooseAiAttackTarget(state, candidate) : undefined;
        if (candidate && chosenTarget) {
          attacker = candidate;
          targetUid = chosenTarget;
        }
      }
      if (!attacker || !targetUid || !declareAiAttack(state, attacker, targetUid)) {
        queueAiAttackContinuation(state, remaining, finish);
        break;
      }
      queueFront(state, {
        type: "aiAttackQuickWindow",
        side: "player",
        data: { attackerUid: attacker.uid, targetUid, remaining, finish },
        label: `${cardName(attacker.cardId)}攻擊後的快速時機`,
      });
      break;
    }
    case "aiAttackQuickWindow": {
      const attackerUid = task.data?.attackerUid as string;
      const targetUid = task.data?.targetUid as string;
      const remaining = (task.data?.remaining as string[]) ?? [];
      const finish = Boolean(task.data?.finish);
      const dukes = standingDukes(state);
      const targets = state.ai.field.filter(isFollower);
      if (!dukes.length || !targets.length || !state.ai.field.some((item) => item.uid === attackerUid)) {
        resolveAiAttackCombat(state, attackerUid, targetUid);
        queueAiAttackContinuation(state, remaining, finish);
        break;
      }
      state.pending = {
        kind: "single",
        effect: "dukeQuickAttack",
        title: "攻擊後的【快速】時機",
        prompt: "可以起動1體直立的レヴィオンデューク・ユリウス；也可以略過，直接結算本次攻擊。",
        options: [
          ...dukes.map((item) => ({ uid: item.uid, cardId: item.cardId })),
          { uid: "pass", label: "不使用【快速】，結算攻擊" },
        ],
        min: 1,
        max: 1,
        data: { attackerUid, targetUid, remaining, finish },
      };
      break;
    }
    case "aiPostAttack": {
      let guard = 0;
      while (state.status === "playing" && state.turnSide === "ai" && !state.pending && guard < 8 && aiCycleChapter(state, false)) guard += 1;
      if (state.status === "playing" && state.turnSide === "ai" && !state.pending) aiEndPhase(state);
      break;
    }
    case "aiResumeMain":
      if (state.aiControl === "scripted" && state.status === "playing" && state.turnSide === "ai" && state.phase === "ai" && !state.pending) runAiTurnMutable(state);
      break;
    case "aiEndQuick": {
      const dukes = standingDukes(state);
      const targets = state.ai.field.filter(isFollower);
      if (!dukes.length || !targets.length) {
        finishAiEndPhase(state);
        break;
      }
      state.pending = {
        kind: "single",
        effect: "dukeQuickEnd",
        title: "對方結束階段的【快速】時機",
        prompt: "可以起動1體直立的レヴィオンデューク・ユリウス；也可以略過並結束對方回合。",
        options: [
          ...dukes.map((item) => ({ uid: item.uid, cardId: item.cardId })),
          { uid: "pass", label: "不使用【快速】，結束對方回合" },
        ],
        min: 1,
        max: 1,
      };
      break;
    }
    case "continuePlayerEnd":
      finishPlayerEndAfterQuickMutable(state);
      break;
    case "manualPlayerAttackCombat":
      resolvePlayerAttackCombat(state, task.data?.attackerUid as string, task.data?.targetUid as string);
      break;
    case "finishPlayerTurn":
      finishTurnSwitchMutable(state);
      break;
    case "addEx":
      addExTask(state, side, task.cardIds ?? [], (task.data?.after as Task[]) ?? []);
      break;
    case "spawn":
      if (task.cardId) spawnToken(state, side, task.cardId);
      break;
    case "draw":
      drawCards(state, side, task.amount ?? 1);
      break;
    case "heal":
      healLeader(state, side, task.amount ?? 0);
      break;
    case "leaderDamage":
      damageLeader(state, side, task.amount ?? 0);
      break;
    case "buffExFairies":
      for (const card of exFairyFollowers(state, side)) {
        card.attackBuff += task.amount ?? 1;
        card.healthBuff += task.amount ?? 1;
      }
      addLog(state, "EX區妖精從者獲得+1/+1。 ");
      break;
    case "buffFieldFairyTokensAttack":
      for (const card of fieldFairyTokens(state, side)) card.attackBuff += task.amount ?? 0;
      addLog(state, `場上妖精衍生卡獲得+${task.amount ?? 0}攻擊力。`);
      break;
    case "miasmaLastWord": {
      if (side === "player") {
        const exFull = state.player.ex.length >= 5;
        state.pending = {
          kind: "yesNo",
          effect: "miasmaLastWord",
          title: "瘴気の妖精姫・アリア",
          prompt: "要將墓場中的這張アリア放入EX區嗎？EX已滿時不能選擇。",
          options: [{ uid: "yes", label: "放入EX", description: exFull ? "不符合：EX區已滿" : undefined }, { uid: "no", label: "留在墓場" }],
          min: 1,
          max: 1,
          data: { sourceUid: task.sourceUid },
        };
      }
      break;
    }
    case "tailwindHeal":
      healLeader(state, side, 1);
      break;
    case "tailwindDraw":
      drawCards(state, side, 1);
      addLog(state, "追い風の妖精使你抽1張。 ");
      break;
    case "aiDamageBestFollower": {
      const targets = ps(state, otherSide(side)).field.filter(isFollower);
      if (side === "ai" && state.aiControl === "manual" && targets.length) state.pending = {
        kind: "single",
        effect: "manualAxiaPing",
        title: "アクシア：每回合1次",
        prompt: "選擇對方場上1體從者，對其造成2點傷害。",
        options: targets.map((item) => ({ uid: item.uid, cardId: item.cardId })),
        min: 1,
        max: 1,
        side: "ai",
        data: { amount: task.amount ?? 0 },
      };
      else {
        const target = bestFollower(state, otherSide(side), "kill");
        if (target) dealDamageToFollower(state, target, task.amount ?? 0, taskLabel(task));
      }
      break;
    }
    case "forestHealCheck":
      if (exFairyFollowers(state, side).length >= 3) healLeader(state, side, 2);
      break;
    case "reverseEvolve": {
      const targets = ps(state, otherSide(side)).field.filter(isFollower);
      if (!targets.length) break;
      if (side === "player") {
        state.pending = {
          kind: "single",
          effect: "reverseEvolveTarget",
          title: "リバースブレイダー・アマツ",
          prompt: "選擇對方1體從者。結算時會分別檢查EX卡3張、EX妖精卡3張。",
          options: targets.map((item) => ({ uid: item.uid, cardId: item.cardId })),
          min: 1,
          max: 1,
        };
      } else {
        const target = bestFollower(state, "player");
        if (target && ps(state, side).ex.length >= 3) destroyFollower(state, target, "リバースブレイダー・アマツ");
        if (exFairyCards(state, side).length >= 3) damageLeader(state, "player", 2);
      }
      break;
    }
    case "miasmaEvolve": {
      const choices = searchDeckInstances(state, side, (item) => isAmulet(item) && isFairyCard(item.cardId) && definition(item).cost <= 2);
      if (!choices.length || !canFitField(state, side)) break;
      if (side === "player") {
        state.pending = {
          kind: "single",
          effect: "miasmaAmuletTutor",
          title: "瘴気の妖精姫・アリア",
          prompt: "選擇原本費用2以下的妖精護符，從牌庫直接放到場上；會觸發其入場曲。",
          options: choices.map((item) => ({ uid: item.uid, cardId: item.cardId })),
          min: 1,
          max: 1,
        };
      }
      break;
    }
    case "naturalAriaSuper":
      for (const card of [...ps(state, side).field, ...ps(state, side).ex]) {
        if (definition(card).token && isFairyCard(card.cardId) && isFollower(card)) {
          card.attackBuff += 1;
          card.healthBuff += 1;
        }
      }
      addLog(state, "自然の妖精姫・アリア使場上與EX區的妖精衍生從者+1/+1。 ");
      break;
    case "servantEvolve": {
      const targets = ps(state, otherSide(side)).field.filter(isFollower);
      if (side === "ai" && state.aiControl === "manual" && targets.length) state.pending = {
        kind: "single",
        effect: "manualServantEvolveTarget",
        title: "破壊の従者（EVOLVE）",
        prompt: "選擇對方場上1體從者，對其與其主戰者各造成2點傷害。",
        options: targets.map((item) => ({ uid: item.uid, cardId: item.cardId })),
        min: 1,
        max: 1,
        side: "ai",
      };
      else {
        const target = bestFollower(state, otherSide(side), "kill");
        if (target) {
          dealDamageToFollower(state, target, 2, "破壊の従者進化時");
          damageLeader(state, otherSide(side), 2);
        }
      }
      break;
    }
    case "axiaEvolve": {
      const source = actionSource(state, task.sourceUid);
      if (side === "ai" && state.aiControl === "manual") {
        const sacrifices = idolField(state, side).filter((item) => item.uid !== source?.uid);
        if (sacrifices.length) state.pending = {
          kind: "multi",
          effect: "manualAxiaSacrifice",
          title: "アクシア（EVOLVE）",
          prompt: "可以將自己場上其他1張偶像卡牌置入墓場，以檢索1張リーシェナ；不選則不發動。",
          options: sacrifices.map((item) => ({ uid: item.uid, cardId: item.cardId })),
          min: 0,
          max: 1,
          side: "ai",
          data: { sourceUid: source?.uid, superEvolve: task.data?.superEvolve },
        };
        else if (task.data?.superEvolve) damageLeader(state, "player", idolField(state, side).length);
        break;
      }
      const sacrifice = aiSacrificeCandidate(state, source?.uid);
      if (sacrifice) {
        moveFieldToGrave(state, sacrifice, "アクシア進化費用");
        const choices = searchDeckInstances(state, side, (item) => isLishenna(item.cardId));
        const picked = choices.sort((a, b) => aiLishennaFetchPriority(state, b.cardId) - aiLishennaFetchPriority(state, a.cardId))[0];
        if (picked) {
          removeDeckInstance(state, side, picked.uid);
          picked.zone = "hand";
          ps(state, side).hand.push(picked);
          addLog(state, `アクシア將${cardName(picked.cardId)}加入手牌。`);
        }
      }
      if (task.data?.superEvolve) damageLeader(state, otherSide(side), idolField(state, side).length);
      break;
    }
    case "genoDig": {
      const top = state.player.deck[state.player.deck.length - 1];
      if (!top) break;
      state.pending = {
        kind: "multi",
        effect: "genoDigPick",
        title: "レヴィオンアックス・ジェノ",
        prompt: "查看牌庫最上面1張牌。若是雷維翁・卡牌，可以公開並加入手牌；也可以不拿，讓它留在牌庫頂。",
        options: [{ uid: top.uid, cardId: top.cardId, description: isLevinCard(top.cardId) ? undefined : "不是雷維翁・卡牌，不能加入手牌" }],
        min: 0,
        max: 1,
        data: { topUid: top.uid },
      };
      break;
    }
    case "runesSnipe": {
      const targets = ps(state, otherSide(side)).field.filter(isFollower);
      if (!targets.length || ps(state, side).pp < 1) break;
      state.pending = {
        kind: "multi",
        effect: "runesSnipe",
        title: "レヴィオンの見習い・ルネス（EVOLVE）",
        prompt: "アルベール進場：可以支付1PP，對對方1體從者造成3點傷害。不選則跳過。",
        options: targets.map((item) => ({ uid: item.uid, cardId: item.cardId })),
        min: 0,
        max: 1,
      };
      break;
    }
    case "runesEvolve": {
      const alberts = ps(state, side).hand.filter((item) => isAlbert(item.cardId) && definition(item).cost <= ps(state, side).maxPP);
      if (!alberts.length || !canFitField(state, side)) break;
      state.pending = {
        kind: "multi",
        effect: "runesPutAlbert",
        title: "レヴィオンの見習い・ルネス（EVOLVE）",
        prompt: "可以將手牌中1張原本費用不大於PP最大值的『アルベール』從者直接放置到場上。不選則跳過。",
        options: alberts.map((item) => ({ uid: item.uid, cardId: item.cardId })),
        min: 0,
        max: 1,
      };
      break;
    }
    case "maimEvolve": {
      const cards = topCards(state, side, 4);
      if (!cards.length) break;
      userTopSearch(state, cards, cards.filter((item) => isLevinCard(item.cardId)), "maimEvoPick", "マイム（EVOLVE）：查看牌頂4張");
      break;
    }
    case "transcendDestroy": {
      const targets = ps(state, otherSide(side)).field.filter(isFollower);
      if (!targets.length) break;
      state.pending = {
        kind: "single",
        effect: "transcendDestroy",
        title: "レヴィオンの超越者・ユリウス",
        prompt: "選擇要破壞的對方從者。",
        options: targets.map((item) => ({ uid: item.uid, cardId: item.cardId })),
        min: 1,
        max: 1,
      };
      break;
    }
    case "transcendDraw": {
      drawCards(state, side, 2);
      if (!ps(state, side).hand.length) break;
      state.pending = {
        kind: "single",
        effect: "transcendDiscard",
        title: "レヴィオンの超越者・ユリウス",
        prompt: "抽了2張。選擇1張手牌捨棄。",
        options: ps(state, side).hand.map((item) => ({ uid: item.uid, cardId: item.cardId })),
        min: 1,
        max: 1,
      };
      break;
    }
    case "lishennaEvolve": {
      const cards = topCards(state, side, 4);
      const capacity = Math.max(0, 5 - ps(state, side).ex.length);
      if (side === "ai" && state.aiControl === "manual") {
        state.pending = {
          kind: "multi",
          effect: "manualLishennaEvolvePick",
          title: "奏絶の破壊・リーシェナ（EVOLVE）",
          prompt: "查看牌頂4張。可以將最多2張偶像卡牌置入EX；其餘任意排序置於牌庫底。",
          options: cards.map((item) => ({
            uid: item.uid,
            cardId: item.cardId,
            description: isIdolCard(item.cardId) ? undefined : "不符合條件",
          })),
          min: 0,
          max: Math.min(2, capacity),
          side: "ai",
          data: { cards, eligible: cards.filter((item) => isIdolCard(item.cardId)).map((item) => item.uid) },
        };
        break;
      }
      const chosen = cards.filter((item) => isIdolCard(item.cardId)).slice(0, Math.min(2, capacity));
      for (const card of chosen) {
        card.zone = "ex";
        ps(state, side).ex.push(card);
        cards.splice(cards.findIndex((item) => item.uid === card.uid), 1);
        addLog(state, `${cardName(card.cardId)}由牌頂放入破壞巫EX區。`);
      }
      putBottom(state, side, cards);
      break;
    }
    default:
      break;
  }
}

function runTasks(state: GameState): void {
  let guard = 0;
  while (!state.pending && state.tasks.length && state.status === "playing" && guard < 200) {
    const task = state.tasks.shift();
    if (task) resolveTask(state, task);
    guard += 1;
  }
  if (guard >= 200) {
    addLog(state, "效果連鎖超過安全上限，已停止。 ");
    state.tasks = [];
  }
}

function playableReason(state: GameState, card: CardInstance, zone: Zone, side: Side): string | undefined {
  if (state.status !== "playing") return "遊戲尚未開始或已結束";
  if (state.turnSide !== side) return "不是這一方的回合";
  if (state.phase !== "main" && !(side === "ai" && state.aiControl === "scripted" && state.phase === "ai")) return "目前不是自己的主階段";
  const canReduceBrutal = card.cardId === "brutalGeno" && side === "player"
    && state.player.field.some((item) => isLevinCard(baseCardId(item)) && isFollower(item) && definition(baseCardId(item)).cost <= 3);
  const requiredPp = canReduceBrutal ? Math.min(1, definition(card).cost) : definition(card).cost;
  if (ps(state, side).pp < requiredPp) return "PP不足";
  if ((isFollower(card) || isAmulet(card)) && !canFitField(state, side) && !canReduceBrutal) return "場上5格已滿";
  if (card.cardId === "tentacleBite") {
    if (!state.player.field.some((item) => isLevinCard(item.cardId) && isFollower(item) && !item.tapped)) return "需要1體可橫置的雷維翁・從者作為追加費用";
    if (!state.ai.field.some(isFollower)) return "對方場上沒有從者";
  }
  if (card.cardId === "levinJustice" && !state.ai.field.some(isFollower)) return "對方場上沒有合法目標";
  if (zone === "ex" && card.cardId === "miasmaAria") return "這張アリア不能從EX區使用";
  if (card.cardId === "antiAir" && (!state.player.field.length || !state.ai.field.some(isFollower))) return "需要自己場上1張[エルフ]卡牌與對方1體從者";
  if (card.cardId === "returningDissonance") {
    if (!ps(state, side).grave.some((item) => isLishenna(item.cardId))) return "墓場沒有可作追加費用的リーシェナ從者";
    if (idolField(state, side).length < 2) return "場上偶像卡牌不足2張";
  }
  if (card.cardId === "whiteBlackChapter" && !ps(state, otherSide(side)).field.some(isFollower)) return "對方場上沒有合法目標";
  if (card.cardId === "solo" && !ps(state, otherSide(side)).field.some(isFollower)) return "對方場上沒有合法目標";
  return undefined;
}

function removePlayedCard(state: GameState, side: Side, zone: Zone, uid: string): CardInstance | undefined {
  return removeFromZone(ps(state, side), zone, uid);
}

function playCardMutable(state: GameState, side: Side, uid: string, zone: Zone, note?: string, opts?: { costOverride?: number; kicker?: boolean }): boolean {
  const owned = findOwned(state, side, uid);
  if (!owned || owned.zone !== zone) return false;
  const reason = playableReason(state, owned.card, zone, side);
  // costOverride（降費/追加費用）情境下改用覆寫後的費用檢查PP，其餘限制照舊。
  if (reason && !(opts?.costOverride !== undefined && reason === "PP不足")) return false;
  const cost = opts?.costOverride ?? definition(owned.card).cost;
  if (ps(state, side).pp < cost) return false;
  const card = removePlayedCard(state, side, zone, uid);
  if (!card) return false;
  ps(state, side).pp -= cost;
  state.playedThisTurn += 1;
  const primary: Task = definition(card).kind === "spell"
    ? { type: "spell", side, sourceUid: card.uid, cardId: card.cardId, label: `${cardName(card.cardId)}的效果`, data: { kicker: opts?.kicker } }
    : { type: "fanfare", side, sourceUid: card.uid, cardId: card.cardId, label: `${cardName(card.cardId)}的入場曲` };

  if (definition(card).kind === "spell") {
    card.zone = "grave";
    if (!cardIsToken(card)) ps(state, side).grave.push(card);
  } else {
    putExistingIntoField(state, card, side, false);
  }
  addLog(state, `${side === "player" ? "你" : "破壞巫"}使用${cardName(card.cardId)}（支付${cost}PP）。`);
  addEvent(state, side, "play", { cardId: card.cardId, pp: ps(state, side).pp, detail: `cost=${cost} zone=${zone}${note ? ` ${note}` : ""}` });

  const simultaneous: Task[] = [primary];
  if (side === "player" && (state.playedThisTurn === 3 || state.playedThisTurn === 5)) {
    for (const tailwind of state.player.field.filter((item) => item.cardId === "tailwindFairy")) {
      simultaneous.push({
        type: state.playedThisTurn === 3 ? "tailwindHeal" : "tailwindDraw",
        side: "player",
        sourceUid: tailwind.uid,
        label: `${cardName(tailwind.cardId)}的第${state.playedThisTurn}張觸發`,
      });
    }
  }
  if (side === "player") {
    if (isFollower(card) && hasKeyword(state, card, "ward")) {
      state.pending = {
        kind: "yesNo",
        effect: "wardOnEntry",
        title: `${cardName(card.cardId)}：守護`,
        prompt: "守護從者進場時，可以改為以橫置狀態進場。要現在橫置它嗎？之後仍會處理同時觸發的效果。",
        options: [{ uid: "yes", label: "以橫置狀態進場" }, { uid: "no", label: "保持直立" }],
        min: 1,
        max: 1,
        data: { sourceUid: card.uid, tasks: simultaneous },
      };
    } else setTriggerOrder(state, simultaneous);
  } else queueFront(state, ...simultaneous);
  runTasks(state);
  return true;
}

export function playCard(input: GameState, uid: string, zone: Zone): GameState {
  const state = clone(input);
  if (state.pending) return state;
  const owned = findOwned(state, "player", uid);
  if (owned && owned.zone === zone && !playableReason(state, owned.card, zone, "player")) {
    const id = owned.card.cardId;
    if (id === "levinSisters" && state.player.pp >= 5) {
      state.pending = {
        kind: "yesNo",
        effect: "sistersKicker",
        title: "レヴィオンシスターズ登場！",
        prompt: "要將費用+4嗎？+4時改為將マイム、ミイム、メイム各1張直接放置到場上；否則找1張加入手牌。",
        options: [{ uid: "yes", label: "費用+4（共5PP）：三姊妹直接進場" }, { uid: "no", label: "維持1PP：找1張加入手牌" }],
        min: 1,
        max: 1,
        data: { uid, zone },
      };
      return state;
    }
    if (id === "levinJustice" && state.player.pp >= 5) {
      state.pending = {
        kind: "yesNo",
        effect: "justiceKicker",
        title: "レヴィオンの正義",
        prompt: "要將費用+2嗎？+2時額外從牌庫找出『レヴィオンセイバー・アルベール』放置到場上（本牌組未收錄則落空）。",
        options: [{ uid: "no", label: "維持3PP" }, { uid: "yes", label: "費用+2（共5PP）" }],
        min: 1,
        max: 1,
        data: { uid, zone },
      };
      return state;
    }
    if (id === "tentacleBite") {
      state.pending = {
        kind: "single",
        effect: "tentacleActCost",
        title: "テンタクルバイト",
        prompt: "選擇1體要橫置的雷維翁・從者作為追加費用。",
        options: state.player.field
          .filter((item) => isLevinCard(item.cardId) && isFollower(item) && !item.tapped)
          .map((item) => ({ uid: item.uid, cardId: item.cardId })),
        min: 1,
        max: 1,
        data: { uid, zone },
      };
      return state;
    }
    if (id === "brutalGeno") {
      const sacrifices = state.player.field.filter((item) => isLevinCard(baseCardId(item)) && isFollower(item) && definition(baseCardId(item)).cost <= 3);
      if (sacrifices.length) {
        const options: ChoiceOption[] = sacrifices.map((item) => ({ uid: item.uid, cardId: item.cardId }));
        if (state.player.pp >= 4 && canFitField(state, "player")) options.push({ uid: "full", label: "不犧牲，支付全額4PP" });
        state.pending = {
          kind: "single",
          effect: "brutalGenoPlay",
          title: "暴威の武人・ジェノ",
          prompt: "可以將1體原本費用3以下的雷維翁・從者置入墓場，使費用-3（支付1PP）。",
          options,
          min: 1,
          max: 1,
          data: { uid, zone },
        };
        return state;
      }
    }
  }
  playCardMutable(state, "player", uid, zone);
  runTasks(state);
  return state;
}

/** 雙策略訓練用；player 仍走原本含追加費用提示的互動入口。 */
export function playCardForSide(input: GameState, side: Side, uid: string, zone: Zone): GameState {
  if (side === "player") return playCard(input, uid, zone);
  const state = clone(input);
  if (state.aiControl !== "manual" || state.pending) return state;
  playCardMutable(state, side, uid, zone);
  runTasks(state);
  return state;
}

function resolveSpell(state: GameState, task: Task): void {
  const cardId = task.cardId;
  const side = task.side ?? "player";
  if (!cardId) return;
  if (side === "player" && cardId === "antiAir") {
    state.pending = {
      kind: "single",
      effect: "antiAirOwn",
      title: "対空射撃",
      prompt: "先選擇要返回手牌的自己場上[エルフ]卡牌。衍生卡會從遊戲中移除。",
      options: state.player.field.map((item) => ({ uid: item.uid, cardId: item.cardId })),
      min: 1,
      max: 1,
    };
    return;
  }
  if (side === "player" && cardId === "tentacleBite") {
    const targets = state.ai.field.filter(isFollower);
    if (!targets.length) return;
    state.pending = {
      kind: "single",
      effect: "tentacleTarget",
      title: "テンタクルバイト",
      prompt: "選擇對方1體從者，對其造成4點傷害；你的主戰者體力+2。",
      options: targets.map((item) => ({ uid: item.uid, cardId: item.cardId })),
      min: 1,
      max: 1,
    };
    return;
  }
  if (side === "player" && cardId === "levinSisters") {
    const names = ["levinMaim", "levinMiim", "levinMeim"];
    if (task.data?.kicker) {
      const candidates = names
        .map((id) => searchDeckInstances(state, "player", (item) => item.cardId === id)[0])
        .filter(Boolean) as CardInstance[];
      const capacity = Math.max(0, 5 - state.player.field.length);
      if (!candidates.length || capacity === 0) {
        addLog(state, !candidates.length ? "牌庫中找不到任何一位姊妹（可以檢索失敗）。" : "場上已滿，可以讓三姊妹的檢索全部落空。");
        shuffleAfterSearch(state, "player", "レヴィオンシスターズ登場！");
        return;
      }
      state.pending = {
        kind: "multi",
        effect: "sistersDeploy",
        title: "レヴィオンシスターズ登場！",
        prompt: `可以選擇マイム、ミイム、メイム各最多1張同時放置到場上；也可以少選或不選。場上目前可放${capacity}張。之後洗牌。`,
        options: candidates.map((item) => ({ uid: item.uid, cardId: item.cardId })),
        min: 0,
        max: Math.min(3, capacity, candidates.length),
      };
      return;
    }
    const candidates = names
      .map((id) => searchDeckInstances(state, "player", (item) => item.cardId === id)[0])
      .filter(Boolean) as CardInstance[];
    if (!candidates.length) {
      addLog(state, "牌庫中找不到任何一位姊妹（可以檢索失敗）。");
      shuffleAfterSearch(state, "player", "レヴィオンシスターズ登場！");
      return;
    }
    state.pending = {
      kind: "multi",
      effect: "sistersPick",
      title: "レヴィオンシスターズ登場！",
      prompt: "可以選擇1張加入手牌；也可以不選。之後洗牌。",
      options: candidates.map((item) => ({ uid: item.uid, cardId: item.cardId })),
      min: 0,
      max: 1,
    };
    return;
  }
  if (side === "player" && cardId === "levinJustice") {
    const targets = state.ai.field.filter(isFollower);
    if (targets.length) {
      state.pending = {
        kind: "single",
        effect: "justiceTarget",
        title: "レヴィオンの正義",
        prompt: "選擇對方1體從者，對其造成3點傷害。之後從牌庫將ユリウス放置到場上。",
        options: targets.map((item) => ({ uid: item.uid, cardId: item.cardId })),
        min: 1,
        max: 1,
        data: { kicker: task.data?.kicker },
      };
    }
    return;
  }
  if (side !== "ai") return;
  if (state.aiControl === "manual") {
    if (cardId === "destructionJoy") {
      if (idolField(state).length) state.ai.pp = Math.min(state.ai.maxPP, state.ai.pp + 1);
      if (state.ai.field.some((item) => item.cardId === "originalLishenna")) drawCards(state, "ai", 1);
      return;
    }
    if (cardId === "annihilationSong") {
      state.pending = {
        kind: "single",
        effect: "manualAnnihilationMode",
        title: "殲滅の歌声",
        prompt: "選擇放置白蛋，或破壞自己場上1張偶像卡牌並抽1張。",
        options: [
          { uid: "white", label: "放置新約・白の章" },
          ...idolField(state).map((item) => ({ uid: item.uid, cardId: item.cardId, label: `破壞${cardName(item.cardId)}並抽1張` })),
        ],
        min: 1,
        max: 1,
        side: "ai",
      };
      return;
    }
    if (cardId === "whiteBlackChapter") {
      const targets = state.player.field.filter(isFollower);
      if (targets.length) state.pending = {
        kind: "single",
        effect: "manualChapterTarget",
        title: "白の章・黒の章",
        prompt: "選擇對方場上1體從者，對其造成2點傷害。",
        options: targets.map((item) => ({ uid: item.uid, cardId: item.cardId })),
        min: 1,
        max: 1,
        side: "ai",
      };
      return;
    }
    if (cardId === "returningDissonance") {
      const grave = state.ai.grave.filter((item) => isLishenna(item.cardId));
      if (grave.length) state.pending = {
        kind: "single",
        effect: "manualReturningCost",
        title: "舞い戻る奏絶：追加費用",
        prompt: "選擇墓場中1張リーシェナ從者消滅。",
        options: grave.map((item) => ({ uid: item.uid, cardId: item.cardId })),
        min: 1,
        max: 1,
        side: "ai",
      };
      return;
    }
    if (cardId === "solo") {
      const targets = state.player.field.filter(isFollower);
      if (targets.length) state.pending = {
        kind: "single",
        effect: "manualSoloTarget",
        title: "奏絶の独唱",
        prompt: "選擇傷害目標；下一步選擇要橫置的任意張偶像卡牌。",
        options: targets.map((item) => ({ uid: item.uid, cardId: item.cardId })),
        min: 1,
        max: 1,
        side: "ai",
      };
      return;
    }
    return;
  }
  const target = bestFollower(state, "player", "kill");
  if (cardId === "destructionJoy") {
    if (idolField(state).length) state.ai.pp = Math.min(state.ai.maxPP, state.ai.pp + 1);
    if (state.ai.field.some((item) => item.cardId === "originalLishenna")) drawCards(state, "ai", 1);
    return;
  }
  if (cardId === "annihilationSong") {
    if (!eggField(state).length && canFitField(state, "ai")) {
      spawnToken(state, "ai", "newWhite");
      addLog(state, "破壞巫以殲滅の歌声優先產出『新約・白の章』（白蛋）。");
    } else {
      const sacrifice = aiSacrificeCandidate(state);
      if (!sacrifice) return;
      moveFieldToGrave(state, sacrifice, "殲滅の歌声破壞");
      drawCards(state, "ai", 1);
    }
    return;
  }
  if (cardId === "whiteBlackChapter" && target) {
    dealDamageToFollower(state, target, 2, "白の章・黒の章");
    if (idolField(state).length >= 2) {
      damageLeader(state, "player", 2);
      healLeader(state, "ai", 2);
    }
    return;
  }
  if (cardId === "returningDissonance") {
    const grave = state.ai.grave.filter((item) => isLishenna(item.cardId));
    const cost = grave.sort((a, b) => definition(b).cost - definition(a).cost)[0];
    if (cost) moveCardToBanished(state, "ai", cost, "grave");
    if (idolField(state).length >= 2 && canFitField(state, "ai")) {
      const choices = searchDeckInstances(state, "ai", (item) => isLishenna(item.cardId));
      const picked = choices.sort((a, b) => aiLishennaFetchPriority(state, b.cardId) - aiLishennaFetchPriority(state, a.cardId))[0];
      if (picked) {
        removeDeckInstance(state, "ai", picked.uid);
        putExistingIntoField(state, picked, "ai", true);
        addLog(state, `舞い戻る奏絶將${cardName(picked.cardId)}從牌庫放到場上。`);
      }
    }
    return;
  }
  if (cardId === "solo" && target) {
    const needed = Math.min(idolField(state).filter((item) => !item.tapped).length, Math.ceil(remainingHealthOf(target) / 2));
    const tappable = idolField(state)
      .filter((item) => !item.tapped)
      .sort((a, b) => aiChapterTapPenalty(state, a) - aiChapterTapPenalty(state, b))
      .slice(0, needed);
    for (const card of tappable) card.tapped = true;
    dealDamageToFollower(state, target, tappable.length * 2, "奏絶の独唱");
  }
}

function canEvolve(state: GameState, side: Side, card: CardInstance, payment: "pp" | "ep", superEvolve = false): string | undefined {
  const def = definition(card);
  if (state.turnSide !== side) return "不是自己的回合";
  if (state.evolvedThisTurn) return "本回合已經進化過";
  if (!def.evolveId || def.evolveCost === undefined) return "這張卡不能進化";
  if ((ps(state, side).evolveRemaining[def.evolveId] ?? 0) <= 0) return "進化區沒有對應卡";
  const evolveCost = card.flags.freeEvolve ? 0 : def.evolveCost;
  const ppCost = Math.max(0, evolveCost - (payment === "ep" ? 1 : 0));
  if (payment === "ep" && ps(state, side).ep <= 0) return "沒有EP";
  if (ps(state, side).pp < ppCost) return "PP不足";
  if (superEvolve && !isSuperEligible(state, side)) return "尚未到超進化回合或SEP已使用";
  return undefined;
}

function evolveMutable(state: GameState, side: Side, uid: string, payment: "pp" | "ep", superEvolve = false): boolean {
  const card = ps(state, side).field.find((item) => item.uid === uid);
  if (!card) return false;
  const reason = canEvolve(state, side, card, payment, superEvolve);
  if (reason) return false;
  const oldDef = definition(card);
  const evolveId = oldDef.evolveId!;
  const evolveCost = card.flags.freeEvolve ? 0 : oldDef.evolveCost!;
  if (payment === "ep") {
    ps(state, side).ep -= 1;
    ps(state, side).pp -= Math.max(0, evolveCost - 1);
  } else ps(state, side).pp -= evolveCost;
  if (superEvolve) ps(state, side).sep -= 1;
  ps(state, side).evolveRemaining[evolveId] -= 1;
  ps(state, side).evolveUsed[evolveId] = (ps(state, side).evolveUsed[evolveId] ?? 0) + 1;
  card.baseCardId ??= card.cardId;
  card.cardId = evolveId;
  card.evolvedThisTurn = true;
  if (superEvolve) {
    card.attackBuff += 1;
    card.healthBuff += 1;
  }
  state.evolvedThisTurn = true;
  addLog(state, `${side === "player" ? "你" : "破壞巫"}使${cardName(card.baseCardId)}${superEvolve ? "超進化" : "進化"}。`);
  addEvent(state, side, superEvolve ? "superEvolve" : "evolve", { cardId: card.baseCardId, pp: ps(state, side).pp, detail: `payment=${payment}` });

  const tasks: Task[] = [];
  if (evolveId === "naturalAriaEvo") tasks.push({ type: "spawn", side, cardId: "fairy", label: "自然の妖精姫・アリア的進化時" });
  if (evolveId === "forestFairyEvo") tasks.push({ type: "addEx", side, cardIds: ["fairyWisp", "fairy"], data: { after: [{ type: "forestHealCheck", side }] }, label: "フォレストフェアリー的進化時" });
  if (evolveId === "reverseAmatsuEvo") tasks.push({ type: "reverseEvolve", side, sourceUid: uid, label: "リバースブレイダー・アマツ的進化時" });
  if (evolveId === "miasmaAriaEvo") tasks.push({ type: "miasmaEvolve", side, sourceUid: uid, label: "瘴気の妖精姫・アリア的進化時" });
  if (evolveId === "levinRunesEvo") tasks.push({ type: "runesEvolve", side, sourceUid: uid, label: "ルネス的進化時" });
  if (evolveId === "levinMaimEvo") tasks.push({ type: "maimEvolve", side, sourceUid: uid, label: "マイム的進化時" });
  if (evolveId === "destructionServantEvo") tasks.push({ type: "servantEvolve", side, sourceUid: uid, label: "破壊の従者的進化時" });
  if (evolveId === "axiaEvo") tasks.push({ type: "axiaEvolve", side, sourceUid: uid, data: { superEvolve }, label: "アクシア的進化時" });
  if (evolveId === "destructiveLishennaEvo") tasks.push({ type: "lishennaEvolve", side, sourceUid: uid, label: "奏絶の破壊・リーシェナ的進化時" });
  if (superEvolve && evolveId === "naturalAriaEvo") tasks.push({ type: "naturalAriaSuper", side, sourceUid: uid, label: "自然の妖精姫・アリア的超進化時" });
  if (side === "player") setTriggerOrder(state, tasks);
  else queueFront(state, ...tasks);
  runTasks(state);
  return true;
}

export function evolveCard(input: GameState, uid: string, payment: "pp" | "ep", superEvolve = false): GameState {
  const state = clone(input);
  if (!state.pending) evolveMutable(state, "player", uid, payment, superEvolve);
  runTasks(state);
  return state;
}

export function evolveCardForSide(
  input: GameState,
  side: Side,
  uid: string,
  payment: "pp" | "ep",
  superEvolve = false,
): GameState {
  if (side === "player") return evolveCard(input, uid, payment, superEvolve);
  const state = clone(input);
  if (state.aiControl !== "manual" || state.pending) return state;
  evolveMutable(state, side, uid, payment, superEvolve);
  runTasks(state);
  return state;
}

function setBottomOrder(state: GameState, side: Side, cards: CardInstance[]): void {
  if (cards.length <= 1) {
    putBottom(state, side, cards);
    return;
  }
  state.pending = {
    kind: "order",
    effect: "bottomOrder",
    title: "排列牌庫底",
    prompt: "依序點選。第一張會放在最底部，最後一張會在這批牌的最上方。",
    options: cards.map((item) => ({ uid: item.uid, cardId: item.cardId })),
    min: cards.length,
    max: cards.length,
    side,
    data: { side, cards },
  };
}

function bounceFollower(state: GameState, target: CardInstance): void {
  const owner = target.owner;
  if (!removeFromZone(ps(state, owner), "field", target.uid)) return;
  const id = baseCardId(target);
  cleanupEvolveCard(state, target);
  if (definition(id).token) {
    addLog(state, `${cardName(id)}返回手牌時，因為是衍生卡而從遊戲中移除。`);
    return;
  }
  target.zone = "hand";
  target.damage = 0;
  target.attackBuff = 0;
  target.healthBuff = 0;
  target.tapped = false;
  ps(state, owner).hand.push(target);
  addLog(state, `${cardName(id)}回到手牌。`);
}

function resolveAttackMutable(state: GameState, attackerUid: string, targetUid: string): void {
  const attacker = state.player.field.find((item) => item.uid === attackerUid);
  if (!attacker || attacker.tapped || !isFollower(attacker)) return;
  const legal = attackTargets(state, attacker).map((item) => item.uid);
  if (!legal.includes(targetUid)) return;
  attacker.tapped = true;
  if (attacker.cardId === "levinMeim" && graveLevin(state) >= 5) {
    attacker.attackBuff += 2;
    attacker.healthBuff += 1;
    addLog(state, "メイム的【攻擊時】：+2/+1。 ");
  }
  addLog(state, `${cardName(attacker.cardId)}宣告攻擊${targetUid === "ai-leader" ? "破壞巫主戰者" : "從者"}。`);
  {
    const targetCard = state.ai.field.find((item) => item.uid === targetUid);
    addEvent(state, "player", "attack", { cardId: attacker.cardId, detail: `atk=${attackOf(attacker)} target=${targetUid === "ai-leader" ? "leader" : targetCard?.cardId ?? targetUid}` });
  }
  if (state.aiControl === "manual") {
    const quick = state.ai.hand.find((item) => item.cardId === "whiteBlackChapter");
    const targets = state.player.field.filter((item) => isFollower(item) && !item.flags.aura);
    if (quick && state.ai.pp >= 2 && targets.length) {
      state.pending = {
        kind: "single",
        effect: "manualAiQuick",
        title: "破壞方【快速】時機",
        prompt: "可以使用白の章・黒の章並指定1體對方從者；或略過。",
        options: [
          { uid: "pass", label: "不使用【快速】" },
          ...targets.map((item) => ({ uid: item.uid, cardId: item.cardId })),
        ],
        min: 1,
        max: 1,
        side: "ai",
        data: { window: "attack", attackerUid, targetUid },
      };
      return;
    }
  }
  aiQuickWindow(state, attacker);
  runTasks(state);
  resolvePlayerAttackCombat(state, attackerUid, targetUid);
}

function resolvePlayerAttackCombat(state: GameState, attackerUid: string, targetUid: string): void {
  const attacker = state.player.field.find((item) => item.uid === attackerUid);
  if (!attacker || state.status === "gameover") return;
  if (targetUid === "ai-leader") {
    damageLeader(state, "ai", attackOf(attacker));
    return;
  }
  const target = state.ai.field.find((item) => item.uid === targetUid);
  if (!target) return;
  const attackerDamage = attackOf(target);
  const targetDamage = attackOf(attacker);
  dealDamageToFollower(state, target, targetDamage, "交戰");
  const stillAttacker = state.player.field.find((item) => item.uid === attackerUid);
  if (stillAttacker) dealDamageToFollower(state, stillAttacker, attackerDamage, "交戰");
}

function canAttackNow(state: GameState, card: CardInstance): boolean {
  if (!isFollower(card) || card.tapped || state.turnSide !== card.owner) return false;
  if (card.enteredAt < state.globalTurn) return true;
  if (card.evolvedThisTurn) return true;
  return hasKeyword(state, card, "storm") || hasKeyword(state, card, "rush");
}

export function attackTargets(state: GameState, attacker: CardInstance): ChoiceOption[] {
  if (!canAttackNow(state, attacker)) return [];
  const opponent = ps(state, otherSide(attacker.owner));
  const wards = opponent.field.filter((item) => isFollower(item) && item.tapped && hasKeyword(state, item, "ward"));
  if (wards.length) return wards.map((item) => ({ uid: item.uid, cardId: item.cardId }));
  const survived = attacker.enteredAt < state.globalTurn;
  const canFace = survived || hasKeyword(state, attacker, "storm");
  const canHitStand = hasKeyword(state, attacker, "designated");
  const targets = opponent.field.filter((item) =>
    isFollower(item) && (item.tapped || canHitStand) && !(attacker.owner === "ai" && item.flags.intimidate));
  const options: ChoiceOption[] = targets.map((item) => ({ uid: item.uid, cardId: item.cardId }));
  if (canFace) options.unshift({ uid: `${otherSide(attacker.owner)}-leader`, label: otherSide(attacker.owner) === "ai" ? "破壞巫主戰者" : "你的主戰者" });
  return options;
}

function canActivate(state: GameState, card: CardInstance, actionId: string): string | undefined {
  if (state.turnSide !== "player" || state.phase !== "main") return "只能在自己的主階段使用";
  if (actionId === "albertRestand") {
    if (!card.tapped) return "此卡已經直立";
    if (state.player.pp < 3) return "需要3PP";
    if (graveRoyalFollowers(state) < 10) return "墓場[ロイヤル]從者不足10張";
    if (card.flags.albertRestandUsed) return "本回合已使用過";
    return undefined;
  }
  if (card.tapped) return "卡片已經橫置";
  if (actionId === "dukePing" && !state.ai.field.some(isFollower)) return "對方沒有從者";
  if (actionId === "archerSnipe") {
    if (!state.ai.field.some(isFollower)) return "對方沒有從者";
  }
  if (actionId === "amatsuStorm") {
    const candidates = state.player.field.filter((item) => item.uid !== card.uid && isFollower(item) && isFairyCard(item.cardId) && definition(item).cost <= 1);
    if (!candidates.length) return "場上沒有原費用1以下的妖精從者";
  }
  if (actionId === "bouquetBounce") {
    if (!state.player.ex.length) return "EX區沒有可消滅的卡";
    if (!state.ai.field.some(isFollower)) return "對方沒有從者";
  }
  if (actionId === "gardenDamage" && !state.ai.field.some(isFollower)) return "對方沒有從者";
  if (actionId === "wonderDraw" && exFairyFollowers(state).length < 3) return "EX區妖精從者不足3張";
  if (actionId === "wingDestroy") {
    if (state.player.pp < 1) return "需要1PP";
    if (exFairyCards(state).length !== 5) return "EX區妖精卡必須正好5張";
    if (!state.ai.field.some(isFollower)) return "對方沒有從者";
  }
  return undefined;
}

export function cardActions(state: GameState, uid: string, zone: Zone, side: Side = "player"): CardAction[] {
  const owned = findOwned(state, side, uid);
  if (!owned || owned.zone !== zone) return [];
  const card = owned.card;
  const actions: CardAction[] = [];
  if ((zone === "hand" || zone === "ex") && (side === "player" || state.aiControl === "manual")) {
    const reason = playableReason(state, card, zone, side);
    actions.push({ id: "play", label: "出場／使用", enabled: !reason, reason });
  }
  if (zone !== "field" || (side === "ai" && state.aiControl !== "manual")) return actions;
  const targets = attackTargets(state, card);
  actions.push({ id: "attack", label: "攻擊", enabled: targets.length > 0, reason: targets.length ? undefined : "現在沒有合法攻擊目標" });

  const def = definition(card);
  if (def.evolveId && !card.baseCardId) {
    const ppReason = canEvolve(state, side, card, "pp", false);
    actions.push({ id: "evolve-pp", label: `以${def.evolveCost} PP進化`, enabled: !ppReason, reason: ppReason, payment: "pp" });
    const epReason = canEvolve(state, side, card, "ep", false);
    actions.push({ id: "evolve-ep", label: `以1 EP${(def.evolveCost ?? 1) > 1 ? `＋${(def.evolveCost ?? 1) - 1} PP` : ""}進化`, enabled: !epReason, reason: epReason, payment: "ep" });
    if (isSuperEligible(state, side)) {
      const spp = canEvolve(state, side, card, "pp", true);
      const sep = canEvolve(state, side, card, "ep", true);
      actions.push({ id: "super-pp", label: "以PP＋SEP超進化", enabled: !spp, reason: spp, payment: "pp", superEvolve: true });
      actions.push({ id: "super-ep", label: "以EP＋SEP超進化", enabled: !sep, reason: sep, payment: "ep", superEvolve: true });
    }
  }
  const actByCard: Record<string, [string, string]> = side === "player" ? {
    fairyBladeAmatsu: ["amatsuStorm", "橫置：給最多2體妖精疾走"],
    bouquetFairy: ["bouquetBounce", "橫置＋消滅EX：敵方從者返回手牌"],
    riotousGarden: ["gardenDamage", "橫置＋置入墓場：造成妖精數量傷害"],
    wonderTree: ["wonderDraw", "橫置＋置入墓場：抽2張"],
    wingQueen: ["wingDestroy", "1PP＋橫置＋置入墓場：破壞敵方從者"],
    levinDuke: ["dukePing", "橫置：對敵方從者造成1點傷害"],
    levinArcher: ["archerSnipe", "橫置：若墓場雷維翁5+，造成3點傷害"],
    levinAlbert: ["albertRestand", "3PP：使此卡直立（墓場ロイヤル從者10+，每回合1次）"],
  } : {};
  const activation = actByCard[baseCardId(card)] ?? actByCard[card.cardId];
  if (activation) {
    const reason = canActivate(state, card, activation[0]);
    actions.push({ id: activation[0], label: activation[1], enabled: !reason, reason });
  }
  return actions;
}

export function activateFieldCard(input: GameState, uid: string, actionId: string): GameState {
  const state = clone(input);
  if (state.pending) return state;
  const source = state.player.field.find((item) => item.uid === uid);
  if (!source || canActivate(state, source, actionId)) return state;
  if (actionId === "attack") {
    const options = attackTargets(state, source);
    state.pending = {
      kind: "single",
      effect: "attackTarget",
      title: `${cardName(source.cardId)}攻擊`,
      prompt: "選擇攻擊目標。若對方場上有橫置狀態的守護從者，必須先攻擊守護。",
      options,
      min: 1,
      max: 1,
      data: { sourceUid: uid },
    };
  } else if (actionId === "amatsuStorm") {
    const options = state.player.field
      .filter((item) => item.uid !== uid && isFollower(item) && isFairyCard(item.cardId) && definition(item).cost <= 1)
      .map((item) => ({ uid: item.uid, cardId: item.cardId }));
    state.pending = {
      kind: "multi",
      effect: "amatsuTargets",
      title: "フェアリーブレイダー・アマツ",
      prompt: "選擇最多2體原本費用1以下的妖精從者。確認後フェアリーブレイダー・アマツ會橫置。",
      options,
      min: 0,
      max: Math.min(2, options.length),
      data: { sourceUid: uid },
    };
  } else if (actionId === "bouquetBounce") {
    state.pending = {
      kind: "single",
      effect: "bouquetExCost",
      title: "花束の妖精",
      prompt: "先選擇要從EX區消滅的1張卡。",
      options: state.player.ex.map((item) => ({ uid: item.uid, cardId: item.cardId })),
      min: 1,
      max: 1,
      data: { sourceUid: uid },
    };
  } else if (actionId === "gardenDamage" || actionId === "wingDestroy") {
    state.pending = {
      kind: "single",
      effect: actionId === "gardenDamage" ? "gardenTarget" : "wingDestroyTarget",
      title: cardName(source.cardId),
      prompt: "選擇對方1體從者。確認後支付起動能力的全部費用。",
      options: state.ai.field.filter(isFollower).map((item) => ({ uid: item.uid, cardId: item.cardId })),
      min: 1,
      max: 1,
      data: { sourceUid: uid },
    };
  } else if (actionId === "wonderDraw") {
    source.tapped = true;
    addEvent(state, "player", "activate", { cardId: "wonderTree" });
    moveFieldToGrave(state, source, "ワンダーツリー起動費用");
    drawCards(state, "player", 2);
    addLog(state, "ワンダーツリー使你抽2張。 ");
  } else if (actionId === "dukePing" || actionId === "archerSnipe") {
    state.pending = {
      kind: "single",
      effect: actionId === "dukePing" ? "dukeTarget" : "archerTarget",
      title: cardName(source.cardId),
      prompt: actionId === "dukePing" ? "選擇要受到1點傷害的對方從者。" : "選擇對方1體從者。結算時若墓場雷維翁卡牌為5張以上，對其造成3點傷害；未達5張仍會橫置此卡。",
      options: state.ai.field.filter(isFollower).map((item) => ({ uid: item.uid, cardId: item.cardId })),
      min: 1,
      max: 1,
      data: { sourceUid: uid },
    };
  } else if (actionId === "albertRestand") {
    state.player.pp -= 3;
    source.tapped = false;
    source.flags.albertRestandUsed = true;
    addEvent(state, "player", "activate", { cardId: "levinAlbert", detail: "restand" });
    addLog(state, "アルベール支付3PP重新直立，可以再次攻擊。 ");
  }
  runTasks(state);
  return state;
}

function normalizedSelection(pending: PendingChoice, selected: string[]): string[] {
  const allowed = new Set(pending.options.map((item) => item.uid));
  const unique = [...new Set(selected)].filter((uid) => allowed.has(uid));
  if (unique.length < pending.min || unique.length > pending.max) return [];
  return unique;
}

export function resolveChoice(input: GameState, selected: string[]): GameState {
  const state = clone(input);
  const pending = state.pending;
  if (!pending) return state;
  const picked = normalizedSelection(pending, selected);
  if (picked.length < pending.min || picked.length > pending.max) return state;
  delete state.pending;

  switch (pending.effect) {
    case "triggerOrder": {
      const tasks = (pending.data?.tasks as Task[]) ?? [];
      const index = Number(picked[0]);
      const chosen = tasks[index];
      const remaining = tasks.filter((_, taskIndex) => taskIndex !== index);
      if (chosen) queueFront(state, chosen, ...(remaining.length ? [{ type: "triggerGroup", side: "player" as Side, data: { tasks: remaining } }] : []));
      break;
    }
    case "wardOnEntry": {
      const source = state.player.field.find((item) => item.uid === pending.data?.sourceUid);
      if (picked[0] === "yes" && source && hasKeyword(state, source, "ward")) {
        source.tapped = true;
        addLog(state, `${cardName(source.cardId)}以橫置狀態進場，守護生效。`);
      }
      const tasks = (pending.data?.tasks as Task[]) ?? [];
      if (tasks.length) setTriggerOrder(state, tasks);
      break;
    }
    case "addExSubset": {
      const cardIds = (pending.data?.cardIds as string[]) ?? [];
      const selectedIds = picked.map((uid) => cardIds[Number(uid.replace("new-", ""))]).filter(Boolean);
      addExDirect(state, (pending.data?.side as Side) ?? "player", selectedIds);
      const after = (pending.data?.after as Task[]) ?? [];
      if (after.length) queueFront(state, ...after);
      break;
    }
    case "fairyArcherPick": {
      const cards = (pending.data?.cards as CardInstance[]) ?? [];
      const eligible = new Set((pending.data?.eligible as string[]) ?? []);
      const chosenUid = picked.find((uid) => eligible.has(uid));
      if (chosenUid) {
        const card = cards.find((item) => item.uid === chosenUid);
        if (card) {
          card.zone = "hand";
          state.player.hand.push(card);
          cards.splice(cards.findIndex((item) => item.uid === card.uid), 1);
          addLog(state, `妖精の弓使い將${cardName(card.cardId)}加入手牌。`);
        }
      }
      setBottomOrder(state, "player", cards);
      break;
    }
    case "bottomOrder": {
      const cards = (pending.data?.cards as CardInstance[]) ?? [];
      const byUid = new Map(cards.map((item) => [item.uid, item]));
      putBottom(state, (pending.data?.side as Side) ?? "player", picked.map((uid) => byUid.get(uid)).filter(Boolean) as CardInstance[]);
      break;
    }
    case "manualAiTopSearch": {
      const cards = (pending.data?.cards as CardInstance[]) ?? [];
      const eligible = new Set((pending.data?.eligible as string[]) ?? []);
      const chosenUid = picked.find((uid) => eligible.has(uid));
      if (chosenUid) {
        const card = cards.find((item) => item.uid === chosenUid);
        if (card) {
          card.zone = "hand";
          state.ai.hand.push(card);
          cards.splice(cards.findIndex((item) => item.uid === card.uid), 1);
          addLog(state, `破壞方將${cardName(card.cardId)}加入手牌。`);
        }
      }
      setBottomOrder(state, "ai", cards);
      break;
    }
    case "manualHermitTarget": {
      const target = state.player.field.find((item) => item.uid === picked[0]);
      if (target && idolField(state).length >= 3) dealDamageToFollower(state, target, 3, "破壊の隠者");
      break;
    }
    case "manualManifestedMode": {
      if (picked[0] === "white") {
        if (canFitField(state, "ai")) spawnToken(state, "ai", "newWhite");
      } else {
        const sacrifice = state.ai.field.find((item) => item.uid === picked[0] && isIdolCard(item.cardId));
        if (sacrifice) {
          moveFieldToGrave(state, sacrifice, "奏絶の顕現・リーシェナ破壞");
          if (state.ai.ex.length < 5) addExTask(state, "ai", ["solo"]);
        }
      }
      break;
    }
    case "manualFanaticTarget": {
      const target = state.player.field.find((item) => item.uid === picked[0]);
      if (target && idolField(state).length >= 3) {
        destroyFollower(state, target, "破壊の狂信者");
        damageLeader(state, "player", 2);
        healLeader(state, "ai", 2);
      }
      break;
    }
    case "manualZelgeneaTarget": {
      const target = state.player.field.find((item) => item.uid === picked[0]);
      if (target && state.ai.field.filter(isFollower).length === 1) {
        destroyFollower(state, target, "《世界》・ゼルガネイア");
        drawCards(state, "ai", 1);
      }
      break;
    }
    case "manualAnnihilationMode": {
      if (picked[0] === "white") {
        if (canFitField(state, "ai")) spawnToken(state, "ai", "newWhite");
      } else {
        const sacrifice = state.ai.field.find((item) => item.uid === picked[0] && isIdolCard(item.cardId));
        if (sacrifice) {
          moveFieldToGrave(state, sacrifice, "殲滅の歌声破壞");
          drawCards(state, "ai", 1);
        }
      }
      break;
    }
    case "manualChapterTarget": {
      const target = state.player.field.find((item) => item.uid === picked[0]);
      if (target) {
        dealDamageToFollower(state, target, 2, "白の章・黒の章");
        if (idolField(state).length >= 2) {
          damageLeader(state, "player", 2);
          healLeader(state, "ai", 2);
        }
      }
      break;
    }
    case "manualAiQuick": {
      if (picked[0] !== "pass") {
        const quick = state.ai.hand.find((item) => item.cardId === "whiteBlackChapter");
        const target = state.player.field.find((item) => item.uid === picked[0] && !item.flags.aura);
        if (quick && target && state.ai.pp >= 2) {
          removeFromZone(state.ai, "hand", quick.uid);
          quick.zone = "grave";
          state.ai.grave.push(quick);
          state.ai.pp -= 2;
          addEvent(state, "ai", "quick", { cardId: quick.cardId, pp: state.ai.pp, detail: `target=${target.cardId}` });
          dealDamageToFollower(state, target, 2, "白の章・黒の章（快速）");
          if (idolField(state).length >= 2) {
            damageLeader(state, "player", 2);
            healLeader(state, "ai", 2);
          }
        }
      }
      if (pending.data?.window === "attack") queue(state, {
        type: "manualPlayerAttackCombat",
        side: "player",
        data: { attackerUid: pending.data?.attackerUid, targetUid: pending.data?.targetUid },
      });
      else queue(state, { type: "continuePlayerEnd", side: "player", label: "破壞方快速效果後繼續結束階段" });
      break;
    }
    case "manualReturningCost": {
      const cost = state.ai.grave.find((item) => item.uid === picked[0] && isLishenna(item.cardId));
      if (cost) moveCardToBanished(state, "ai", cost, "grave");
      const choices = searchDeckInstances(state, "ai", (item) => isLishenna(item.cardId));
      if (idolField(state).length >= 2 && canFitField(state, "ai") && choices.length) state.pending = {
        kind: "single",
        effect: "manualReturningTutor",
        title: "舞い戻る奏絶：檢索",
        prompt: "選擇牌庫中1張リーシェナ從者放置到場上。",
        options: choices.map((item) => ({ uid: item.uid, cardId: item.cardId })),
        min: 1,
        max: 1,
        side: "ai",
      };
      else shuffleAfterSearch(state, "ai", "舞い戻る奏絶");
      break;
    }
    case "manualReturningTutor": {
      const card = removeDeckInstance(state, "ai", picked[0]);
      if (card && canFitField(state, "ai")) putExistingIntoField(state, card, "ai", true);
      shuffleAfterSearch(state, "ai", "舞い戻る奏絶");
      break;
    }
    case "manualSoloTarget": {
      const tappable = idolField(state).filter((item) => !item.tapped);
      state.pending = {
        kind: "multi",
        effect: "manualSoloCosts",
        title: "奏絶の独唱：追加費用",
        prompt: "選擇要橫置的任意張數偶像卡牌；每張使傷害增加2。",
        options: tappable.map((item) => ({ uid: item.uid, cardId: item.cardId })),
        min: 0,
        max: tappable.length,
        side: "ai",
        data: { targetUid: picked[0] },
      };
      break;
    }
    case "manualSoloCosts": {
      for (const uid of picked) {
        const card = state.ai.field.find((item) => item.uid === uid && isIdolCard(item.cardId) && !item.tapped);
        if (card) card.tapped = true;
      }
      const target = state.player.field.find((item) => item.uid === pending.data?.targetUid);
      if (target) dealDamageToFollower(state, target, picked.length * 2, "奏絶の独唱");
      break;
    }
    case "manualAxiaPing": {
      const target = state.player.field.find((item) => item.uid === picked[0]);
      if (target) dealDamageToFollower(state, target, Number(pending.data?.amount ?? 2), "アクシア");
      break;
    }
    case "manualServantEvolveTarget": {
      const target = state.player.field.find((item) => item.uid === picked[0]);
      if (target) {
        dealDamageToFollower(state, target, 2, "破壊の従者進化時");
        damageLeader(state, "player", 2);
      }
      break;
    }
    case "manualAxiaSacrifice": {
      if (picked.length) {
        const sacrifice = state.ai.field.find((item) => item.uid === picked[0] && isIdolCard(item.cardId));
        if (sacrifice) moveFieldToGrave(state, sacrifice, "アクシア進化費用");
        const choices = searchDeckInstances(state, "ai", (item) => isLishenna(item.cardId));
        if (choices.length) state.pending = {
          kind: "single",
          effect: "manualAxiaTutor",
          title: "アクシア（EVOLVE）：檢索",
          prompt: "選擇牌庫中1張リーシェナ從者加入手牌。",
          options: choices.map((item) => ({ uid: item.uid, cardId: item.cardId })),
          min: 1,
          max: 1,
          side: "ai",
        };
        else shuffleAfterSearch(state, "ai", "アクシア（EVOLVE）");
      }
      if (pending.data?.superEvolve) damageLeader(state, "player", idolField(state, "ai").length);
      break;
    }
    case "manualAxiaTutor": {
      const card = removeDeckInstance(state, "ai", picked[0]);
      if (card) {
        card.zone = "hand";
        state.ai.hand.push(card);
      }
      shuffleAfterSearch(state, "ai", "アクシア（EVOLVE）");
      break;
    }
    case "manualLishennaEvolvePick": {
      const cards = (pending.data?.cards as CardInstance[]) ?? [];
      const eligible = new Set((pending.data?.eligible as string[]) ?? []);
      for (const uid of picked.filter((item) => eligible.has(item))) {
        const card = cards.find((item) => item.uid === uid);
        if (!card || state.ai.ex.length >= 5) continue;
        card.zone = "ex";
        state.ai.ex.push(card);
        cards.splice(cards.findIndex((item) => item.uid === uid), 1);
      }
      setBottomOrder(state, "ai", cards);
      break;
    }
    case "wingQueenTutor":
    case "miasmaAmuletTutor": {
      const card = removeDeckInstance(state, "player", picked[0]);
      if (card) {
        putExistingIntoField(state, card, "player", true);
        addLog(state, `${pending.effect === "wingQueenTutor" ? "翅の女王・ティターニア" : "瘴気の妖精姫・アリア"}將${cardName(card.cardId)}從牌庫放到場上。`);
      }
      break;
    }
    case "miasmaLastWord": {
      if (picked[0] === "yes" && state.player.ex.length < 5) {
        const uid = pending.data?.sourceUid as string;
        const card = removeFromZone(state.player, "grave", uid);
        if (card) {
          card.zone = "ex";
          state.player.ex.push(card);
          addLog(state, "瘴気の妖精姫・アリア由墓場移到EX區。 ");
        }
      }
      break;
    }
    case "antiAirOwn": {
      const own = state.player.field.find((item) => item.uid === picked[0]);
      if (!own) break;
      removeFromZone(state.player, "field", own.uid);
      const id = baseCardId(own);
      cleanupEvolveCard(state, own);
      if (definition(id).token) addLog(state, `${cardName(id)}因対空射撃回手而移除遊戲。`);
      else {
        own.zone = "hand";
        own.damage = 0;
        own.attackBuff = 0;
        own.healthBuff = 0;
        own.tapped = false;
        state.player.hand.push(own);
      }
      state.pending = {
        kind: "single",
        effect: "antiAirEnemy",
        title: "対空射撃",
        prompt: "選擇要受到3點傷害的對方從者。",
        options: state.ai.field.filter(isFollower).map((item) => ({ uid: item.uid, cardId: item.cardId })),
        min: 1,
        max: 1,
      };
      break;
    }
    case "antiAirEnemy": {
      const target = state.ai.field.find((item) => item.uid === picked[0]);
      if (target) dealDamageToFollower(state, target, 3, "対空射撃");
      break;
    }
    case "reverseEvolveTarget": {
      const target = state.ai.field.find((item) => item.uid === picked[0]);
      if (target && state.player.ex.length >= 3) destroyFollower(state, target, "リバースブレイダー・アマツ");
      if (exFairyCards(state).length >= 3) damageLeader(state, "ai", 2);
      break;
    }
    case "amatsuTargets": {
      const source = state.player.field.find((item) => item.uid === pending.data?.sourceUid);
      if (!source || source.tapped) break;
      source.tapped = true;
      for (const uid of picked) {
        const target = state.player.field.find((item) => item.uid === uid);
        if (target) target.tempStorm = true;
      }
      addLog(state, `フェアリーブレイダー・アマツ給${picked.length}體妖精疾走。`);
      addEvent(state, "player", "activate", { cardId: "fairyBladeAmatsu", detail: `storm=${picked.length}` });
      break;
    }
    case "bouquetExCost": {
      const sourceUid = pending.data?.sourceUid as string;
      const card = state.player.ex.find((item) => item.uid === picked[0]);
      if (!card) break;
      moveCardToBanished(state, "player", card, "ex");
      state.pending = {
        kind: "single",
        effect: "bouquetTarget",
        title: "花束の妖精",
        prompt: "選擇要回到手牌的對方從者。",
        options: state.ai.field.filter(isFollower).map((item) => ({ uid: item.uid, cardId: item.cardId })),
        min: 1,
        max: 1,
        data: { sourceUid },
      };
      break;
    }
    case "bouquetTarget": {
      const source = state.player.field.find((item) => item.uid === pending.data?.sourceUid);
      const target = state.ai.field.find((item) => item.uid === picked[0]);
      if (source && target) {
        source.tapped = true;
        addEvent(state, "player", "activate", { cardId: "bouquetFairy", detail: `bounce=${target.cardId}` });
        bounceFollower(state, target);
      }
      break;
    }
    case "gardenTarget": {
      const source = state.player.field.find((item) => item.uid === pending.data?.sourceUid);
      const target = state.ai.field.find((item) => item.uid === picked[0]);
      if (source && target) {
        source.tapped = true;
        addEvent(state, "player", "activate", { cardId: "riotousGarden", detail: `target=${target.cardId}` });
        moveFieldToGrave(state, source, "繚乱の庭起動費用");
        const amount = fieldFairyTokens(state).length + state.player.ex.filter((item) => definition(item).token && isFairyCard(item.cardId) && isFollower(item)).length;
        dealDamageToFollower(state, target, amount, "繚乱の庭");
      }
      break;
    }
    case "wingDestroyTarget": {
      const source = state.player.field.find((item) => item.uid === pending.data?.sourceUid);
      const target = state.ai.field.find((item) => item.uid === picked[0]);
      if (source && target && state.player.pp >= 1 && exFairyCards(state).length === 5) {
        state.player.pp -= 1;
        source.tapped = true;
        addEvent(state, "player", "activate", { cardId: "wingQueen", detail: `destroy=${target.cardId}` });
        moveFieldToGrave(state, source, "翅の女王・ティターニア起動費用");
        destroyFollower(state, target, "翅の女王・ティターニア");
        spawnToken(state, "ai", "fairy");
      }
      break;
    }
    case "attackTarget":
      resolveAttackMutable(state, pending.data?.sourceUid as string, picked[0]);
      break;
    case "sistersKicker":
    case "justiceKicker": {
      const uid = pending.data?.uid as string;
      const zone = (pending.data?.zone as Zone) ?? "hand";
      const kicker = picked[0] === "yes";
      const extra = pending.effect === "sistersKicker" ? 4 : 2;
      const card = findOwned(state, "player", uid)?.card;
      const baseCost = card ? definition(card).cost : 0;
      playCardMutable(state, "player", uid, zone, undefined, { costOverride: baseCost + (kicker ? extra : 0), kicker });
      break;
    }
    case "tentacleActCost": {
      const cost = state.player.field.find((item) => item.uid === picked[0]);
      if (!cost || cost.tapped) break;
      cost.tapped = true;
      addLog(state, `你橫置${cardName(cost.cardId)}作為テンタクルバイト的追加費用。`);
      playCardMutable(state, "player", pending.data?.uid as string, (pending.data?.zone as Zone) ?? "hand");
      break;
    }
    case "brutalGenoPlay": {
      if (picked[0] === "full") {
        playCardMutable(state, "player", pending.data?.uid as string, (pending.data?.zone as Zone) ?? "hand");
        break;
      }
      const sacrifice = state.player.field.find((item) => item.uid === picked[0]);
      if (!sacrifice) break;
      moveFieldToGrave(state, sacrifice, "暴威の武人・ジェノ的費用");
      playCardMutable(state, "player", pending.data?.uid as string, (pending.data?.zone as Zone) ?? "hand", undefined, { costOverride: 1 });
      break;
    }
    case "miimDiscard": {
      if (!picked.length) break;
      if (discardPlayerCard(state, picked[0], "ミイム的入場曲")) {
        drawCards(state, "player", 1);
        addLog(state, "ミイム使你抽1張。 ");
      }
      break;
    }
    case "runesDiscard": {
      if (!picked.length) break;
      if (discardPlayerCard(state, picked[0], "ルネス的入場曲")) {
        beginRunesAlbertSearch(state);
      }
      break;
    }
    case "runesAlbertPick": {
      const card = picked.length ? removeDeckInstance(state, "player", picked[0]) : undefined;
      if (card && isAlbert(card.cardId)) {
        card.zone = "hand";
        state.player.hand.push(card);
        addLog(state, `ルネス將${cardName(card.cardId)}加入手牌。`);
      } else addLog(state, "你選擇讓ルネス的檢索落空。");
      shuffleAfterSearch(state, "player", "ルネス");
      break;
    }
    case "genoDigPick": {
      const top = state.player.deck[state.player.deck.length - 1];
      const expectedUid = pending.data?.topUid as string | undefined;
      if (picked.length && top?.uid === expectedUid && picked[0] === expectedUid && isLevinCard(top.cardId)) {
        state.player.deck.pop();
        top.zone = "hand";
        state.player.hand.push(top);
        addLog(state, `ジェノ的捨棄能力：公開牌庫頂的${cardName(top.cardId)}並加入手牌。`);
      } else if (top?.uid === expectedUid) {
        addLog(state, `ジェノ查看牌庫頂的${cardName(top.cardId)}後，讓它留在牌庫頂。`);
      }
      break;
    }
    case "maimTarget": {
      const target = state.ai.field.find((item) => item.uid === picked[0]);
      if (target && graveLevin(state) >= 5) dealDamageToFollower(state, target, 3, "レヴィオンの副団長・マイム");
      else if (target) addLog(state, "マイム的入場曲選定了目標，但墓場雷維翁・卡牌未達5張，因此沒有造成傷害。");
      break;
    }
    case "archerLevinPick": {
      const cards = (pending.data?.cards as CardInstance[]) ?? [];
      const chosen = cards.find((item) => item.uid === picked[0] && isLevinCard(item.cardId));
      if (chosen) {
        chosen.zone = "hand";
        state.player.hand.push(chosen);
        cards.splice(cards.findIndex((item) => item.uid === chosen.uid), 1);
        addLog(state, `レヴィオンの弓使い將${cardName(chosen.cardId)}加入手牌。`);
      }
      for (const rest of cards) {
        rest.zone = "grave";
        state.player.grave.push(rest);
      }
      if (cards.length) addLog(state, `其餘${cards.length}張置入墓場。`);
      break;
    }
    case "transcendRevealChoice": {
      if (picked[0] !== "yes") {
        addLog(state, "你沒有支付超越者・ユリウス的公開費用，入場曲不發動。");
        break;
      }
      const levinHand = state.player.hand.filter((item) => isLevinCard(item.cardId));
      if (levinHand.length < 2) break;
      state.pending = {
        kind: "multi",
        effect: "transcendRevealCards",
        title: "レヴィオンの超越者・ユリウス",
        prompt: "選擇正好2張手牌中的雷維翁・卡牌公開。",
        options: levinHand.map((item) => ({ uid: item.uid, cardId: item.cardId })),
        min: 2,
        max: 2,
      };
      break;
    }
    case "transcendRevealCards": {
      const revealed = state.player.hand.filter((item) => picked.includes(item.uid) && isLevinCard(item.cardId));
      if (revealed.length !== 2) break;
      addLog(state, `你公開手牌的${revealed.map((item) => cardName(item.cardId)).join("、")}。`);
      const maxModes = graveLevin(state) >= 5 ? 3 : 1;
      const options: ChoiceOption[] = [
        { uid: "burn", label: "②對對方主戰者造成3點傷害" },
        { uid: "draw", label: "③抽2張牌，捨棄自己1張手牌" },
      ];
      if (state.ai.field.some(isFollower)) options.unshift({ uid: "destroy", label: "①破壞對方場上1體從者" });
      state.pending = {
        kind: "multi",
        effect: "transcendModes",
        title: "レヴィオンの超越者・ユリウス",
        prompt: `選擇${maxModes === 3 ? "最多3項" : "1項"}效果。`,
        options,
        min: 1,
        max: Math.min(maxModes, options.length),
      };
      break;
    }
    case "transcendModes": {
      const order = ["destroy", "burn", "draw"].filter((mode) => picked.includes(mode));
      const tasks: Task[] = order.map((mode) => {
        if (mode === "destroy") return { type: "transcendDestroy", side: "player" as Side, label: "超越者・ユリウス①" };
        if (mode === "burn") return { type: "leaderDamage", side: "ai" as Side, amount: 3, label: "超越者・ユリウス②" };
        return { type: "transcendDraw", side: "player" as Side, label: "超越者・ユリウス③" };
      });
      queueFront(state, ...tasks);
      break;
    }
    case "transcendDestroy": {
      const target = state.ai.field.find((item) => item.uid === picked[0]);
      if (target) destroyFollower(state, target, "レヴィオンの超越者・ユリウス");
      break;
    }
    case "transcendDiscard": {
      if (picked.length) discardPlayerCard(state, picked[0], "超越者・ユリウス的效果");
      break;
    }
    case "brutalGenoTarget": {
      const target = state.ai.field.find((item) => item.uid === picked[0]);
      if (target) dealDamageToFollower(state, target, 4, "暴威の武人・ジェノ");
      break;
    }
    case "albertBuff": {
      const target = state.player.field.find((item) => item.uid === picked[0]);
      if (target) {
        target.attackBuff += 1;
        addLog(state, `アルベール使${cardName(target.cardId)}攻擊力+1。`);
      }
      break;
    }
    case "tentacleTarget": {
      const target = state.ai.field.find((item) => item.uid === picked[0]);
      if (target) dealDamageToFollower(state, target, 4, "テンタクルバイト");
      healLeader(state, "player", 2);
      if (state.player.field.some((item) => cardName(item.cardId).includes("ユリウス"))) {
        drawCards(state, "player", 1);
        addLog(state, "場上有ユリウス，テンタクルバイト使你抽1張。 ");
      }
      break;
    }
    case "sistersPick": {
      const card = picked.length ? removeDeckInstance(state, "player", picked[0]) : undefined;
      if (card) {
        card.zone = "hand";
        state.player.hand.push(card);
        addLog(state, `レヴィオンシスターズ登場！將${cardName(card.cardId)}加入手牌。`);
      } else addLog(state, "你選擇讓レヴィオンシスターズ登場！的檢索落空。");
      shuffleAfterSearch(state, "player", "レヴィオンシスターズ登場！");
      break;
    }
    case "sistersDeploy": {
      deploySistersSimultaneously(state, picked);
      if (!picked.length) addLog(state, "你選擇讓三姊妹的檢索全部落空。");
      shuffleAfterSearch(state, "player", "レヴィオンシスターズ登場！");
      break;
    }
    case "justiceTarget": {
      const target = state.ai.field.find((item) => item.uid === picked[0]);
      if (target) dealDamageToFollower(state, target, 3, "レヴィオンの正義");
      beginJusticeDukeSearch(state, Boolean(pending.data?.kicker));
      break;
    }
    case "justiceDukePick": {
      finishJusticeDukeSearch(state, Boolean(pending.data?.kicker), picked[0]);
      break;
    }
    case "justiceSaberPick": {
      const card = picked.length ? removeDeckInstance(state, "player", picked[0]) : undefined;
      if (card && canFitField(state, "player")) {
        putExistingIntoField(state, card, "player", true);
        addLog(state, "レヴィオンの正義將レヴィオンセイバー・アルベール放置到場上。");
      } else addLog(state, "你選擇讓レヴィオンセイバー・アルベール的追加檢索落空。");
      shuffleAfterSearch(state, "player", "レヴィオンの正義（追加檢索）");
      break;
    }
    case "runesSnipe": {
      if (!picked.length || state.player.pp < 1) break;
      const target = state.ai.field.find((item) => item.uid === picked[0]);
      if (target) {
        state.player.pp -= 1;
        dealDamageToFollower(state, target, 3, "ルネス（EVOLVE）");
      }
      break;
    }
    case "runesPutAlbert": {
      if (!picked.length) break;
      const card = state.player.hand.find((item) => item.uid === picked[0]);
      if (card && canFitField(state, "player")) {
        removeFromZone(state.player, "hand", card.uid);
        putExistingIntoField(state, card, "player", true);
        addLog(state, `ルネス（EVOLVE）將${cardName(card.cardId)}放置到場上。`);
      }
      break;
    }
    case "maimEvoPick": {
      const cards = (pending.data?.cards as CardInstance[]) ?? [];
      const eligible = new Set((pending.data?.eligible as string[]) ?? []);
      const chosenUid = picked.find((uid) => eligible.has(uid));
      if (chosenUid) {
        const card = cards.find((item) => item.uid === chosenUid);
        if (card) {
          card.zone = "hand";
          state.player.hand.push(card);
          cards.splice(cards.findIndex((item) => item.uid === card.uid), 1);
          addLog(state, `マイム（EVOLVE）將${cardName(card.cardId)}加入手牌。`);
        }
      }
      setBottomOrder(state, "player", cards);
      break;
    }
    case "dukeTarget": {
      const source = state.player.field.find((item) => item.uid === pending.data?.sourceUid);
      const target = state.ai.field.find((item) => item.uid === picked[0]);
      if (source && target && !source.tapped) {
        source.tapped = true;
        addEvent(state, "player", "activate", { cardId: "levinDuke", detail: `target=${target.cardId}` });
        dealDamageToFollower(state, target, 1, "レヴィオンデューク・ユリウス");
      }
      break;
    }
    case "dukeQuickAttack": {
      const attackerUid = pending.data?.attackerUid as string;
      const targetUid = pending.data?.targetUid as string;
      const remaining = (pending.data?.remaining as string[]) ?? [];
      const finish = Boolean(pending.data?.finish);
      if (picked[0] === "pass") {
        resolveAiAttackCombat(state, attackerUid, targetUid);
        queueAiAttackContinuation(state, remaining, finish);
        break;
      }
      const duke = state.player.field.find((item) => item.uid === picked[0] && baseCardId(item) === "levinDuke" && !item.tapped);
      if (!duke) {
        resolveAiAttackCombat(state, attackerUid, targetUid);
        queueAiAttackContinuation(state, remaining, finish);
        break;
      }
      const targets = state.ai.field.filter(isFollower);
      if (!targets.length) {
        queueAiAttackContinuation(state, remaining, finish);
        break;
      }
      state.pending = {
        kind: "single",
        effect: "dukeQuickAttackTarget",
        title: `${cardName(duke.cardId)}【快速】`,
        prompt: "選擇對方場上1體從者，對其造成1點傷害。",
        options: targets.map((item) => ({ uid: item.uid, cardId: item.cardId })),
        min: 1,
        max: 1,
        data: { dukeUid: duke.uid, attackerUid, targetUid, remaining, finish },
      };
      break;
    }
    case "dukeQuickAttackTarget": {
      const duke = state.player.field.find((item) => item.uid === pending.data?.dukeUid && baseCardId(item) === "levinDuke" && !item.tapped);
      const target = state.ai.field.find((item) => item.uid === picked[0]);
      const attackerUid = pending.data?.attackerUid as string;
      const targetUid = pending.data?.targetUid as string;
      const remaining = (pending.data?.remaining as string[]) ?? [];
      const finish = Boolean(pending.data?.finish);
      if (duke && target) {
        duke.tapped = true;
        addEvent(state, "player", "quick", { cardId: "levinDuke", detail: `attackWindow target=${target.cardId}` });
        dealDamageToFollower(state, target, 1, "レヴィオンデューク・ユリウス【快速】");
      }
      if (state.ai.field.some((item) => item.uid === attackerUid)) {
        queue(state, { type: "aiAttackQuickWindow", side: "player", data: { attackerUid, targetUid, remaining, finish } });
      } else {
        addLog(state, "攻擊從者被【快速】能力擊破，本次攻擊不造成傷害。");
        queueAiAttackContinuation(state, remaining, finish);
      }
      break;
    }
    case "dukeQuickEnd": {
      if (picked[0] === "pass") {
        finishAiEndPhase(state);
        break;
      }
      const duke = state.player.field.find((item) => item.uid === picked[0] && baseCardId(item) === "levinDuke" && !item.tapped);
      if (!duke) {
        finishAiEndPhase(state);
        break;
      }
      const targets = state.ai.field.filter(isFollower);
      if (!targets.length) {
        finishAiEndPhase(state);
        break;
      }
      state.pending = {
        kind: "single",
        effect: "dukeQuickEndTarget",
        title: `${cardName(duke.cardId)}【快速】`,
        prompt: "選擇對方場上1體從者，對其造成1點傷害。",
        options: targets.map((item) => ({ uid: item.uid, cardId: item.cardId })),
        min: 1,
        max: 1,
        data: { dukeUid: duke.uid },
      };
      break;
    }
    case "dukeQuickEndTarget": {
      const duke = state.player.field.find((item) => item.uid === pending.data?.dukeUid && baseCardId(item) === "levinDuke" && !item.tapped);
      const target = state.ai.field.find((item) => item.uid === picked[0]);
      if (duke && target) {
        duke.tapped = true;
        addEvent(state, "player", "quick", { cardId: "levinDuke", detail: `endWindow target=${target.cardId}` });
        dealDamageToFollower(state, target, 1, "レヴィオンデューク・ユリウス【快速】");
      }
      queue(state, { type: "aiEndQuick", side: "player" });
      break;
    }
    case "archerTarget": {
      const source = state.player.field.find((item) => item.uid === pending.data?.sourceUid);
      const target = state.ai.field.find((item) => item.uid === picked[0]);
      if (source && target && !source.tapped) {
        source.tapped = true;
        if (graveLevin(state) >= 5) dealDamageToFollower(state, target, 3, "レヴィオンの弓使い");
        else addLog(state, "弓使い支付橫置費用，但墓場雷維翁・卡牌未達5張，因此沒有造成傷害。");
      }
      break;
    }
    case "guardChoice":
      for (const uid of picked) {
        const ward = state.player.field.find((item) => item.uid === uid);
        if (ward && hasKeyword(state, ward, "ward") && !ward.tapped) ward.tapped = true;
      }
      addLog(state, `你讓${picked.length}體守護從者進入橫置守護狀態。`);
      continueEndTurnMutable(state);
      break;
    case "discardToSeven":
      for (const uid of picked) discardPlayerCard(state, uid, "手牌上限");
      // 捨棄本身可能觸發傑諾等需要玩家回答的效果。換回合任務排在
      // 這些觸發之後，確保最後一個選擇完成才讓對手開始行動。
      queue(state, { type: "finishPlayerTurn", side: "player", label: "手牌上限處理後結束回合" });
      break;
    case "aiDiscardToSeven":
      for (const uid of picked) {
        const card = removeFromZone(state.ai, "hand", uid);
        if (!card) continue;
        card.zone = "grave";
        if (!cardIsToken(card)) state.ai.grave.push(card);
        addLog(state, `破壞巫因手牌上限捨棄${cardName(card.cardId)}。`);
      }
      finishTurnSwitchMutable(state);
      break;
    default:
      break;
  }
  runTasks(state);
  return state;
}

function aiQuickWindow(state: GameState, attacking?: CardInstance): void {
  if (state.turnSide !== "player" || state.ai.pp < 2) return;
  const quick = state.ai.hand.find((item) => item.cardId === "whiteBlackChapter");
  const target = attacking ?? bestFollower(state, "player", "kill");
  if (!quick || !target || !state.player.field.some((item) => item.uid === target.uid)) return;
  if (target.flags.aura) return;
  const kills = remainingHealthOf(target) <= 2;
  const highThreat = attackOf(target) >= 4 || target.cardId === "queenCynthia" || target.cardId === "fairyBladeAmatsu"
    || target.cardId === "levinAlbert" || target.cardId === "levinMeim";
  // 目標太廉價（低費衍生物、低攻）就不值得燒掉快速章與2PP。
  const worthTarget = definition(baseCardId(target)).cost >= 2 || attackOf(target) >= 3;
  if (!highThreat && !(kills && worthTarget)) return;
  // 玩家已進斬殺圈時，章要留著當致命直傷，不做防守性交換——除非我方自己快被沖死。
  if (!highThreat && state.player.hp <= 4 && idolField(state).length >= 2 && !aiInDanger(state)) return;
  removeFromZone(state.ai, "hand", quick.uid);
  quick.zone = "grave";
  state.ai.grave.push(quick);
  state.ai.pp -= 2;
  addLog(state, `破壞巫在快速時機使用${cardName(quick.cardId)}，目標為${cardName(target.cardId)}。`);
  addEvent(state, "ai", "quick", { cardId: quick.cardId, pp: state.ai.pp, detail: `target=${target.cardId}` });
  dealDamageToFollower(state, target, 2, "白の章・黒の章（快速）");
  if (idolField(state).length >= 2) {
    damageLeader(state, "player", 2);
    healLeader(state, "ai", 2);
  }
}

function finishPlayerEndAfterQuickMutable(state: GameState): void {
  if (state.status !== "playing" || state.turnSide !== "player") return;
  const excess = state.player.hand.length - 7;
  if (excess > 0) {
    state.pending = {
      kind: "multi",
      effect: "discardToSeven",
      title: "手牌上限",
      prompt: `結束階段手牌超過7張，請選擇${excess}張捨棄。`,
      options: state.player.hand.map((item) => ({ uid: item.uid, cardId: item.cardId })),
      min: excess,
      max: excess,
    };
    return;
  }
  finishTurnSwitchMutable(state);
}

/** 將指定攻擊者與目標視為一個完整策略動作；AI 攻擊後仍保留玩家的【快速】回應窗口。 */
export function attackCardForSide(input: GameState, side: Side, attackerUid: string, targetUid: string): GameState {
  const state = clone(input);
  if (state.pending || state.status !== "playing" || state.turnSide !== side) return state;
  if (side === "player") {
    resolveAttackMutable(state, attackerUid, targetUid);
    runTasks(state);
    return state;
  }
  if (state.aiControl !== "manual") return state;
  const attacker = state.ai.field.find((item) => item.uid === attackerUid);
  if (!attacker || !declareAiAttack(state, attacker, targetUid)) return state;
  queueFront(state, {
    type: "aiAttackQuickWindow",
    side: "player",
    data: { attackerUid, targetUid, remaining: [], finish: false },
    label: `${cardName(attacker.cardId)}攻擊後的快速時機`,
  });
  runTasks(state);
  return state;
}

function continueEndTurnMutable(state: GameState): void {
  if (state.status !== "playing" || state.turnSide !== "player") return;
  if (state.aiControl === "manual") {
    const quick = state.ai.hand.find((item) => item.cardId === "whiteBlackChapter");
    const targets = state.player.field.filter((item) => isFollower(item) && !item.flags.aura);
    if (quick && state.ai.pp >= 2 && targets.length) {
      state.pending = {
        kind: "single",
        effect: "manualAiQuick",
        title: "玩家結束階段：破壞方【快速】時機",
        prompt: "可以使用白の章・黒の章並指定1體對方從者；或略過。",
        options: [
          { uid: "pass", label: "不使用【快速】" },
          ...targets.map((item) => ({ uid: item.uid, cardId: item.cardId })),
        ],
        min: 1,
        max: 1,
        side: "ai",
        data: { window: "end" },
      };
      return;
    }
  }
  aiQuickWindow(state);
  runTasks(state);
  if (state.status !== "playing") return;
  if (state.pending) {
    // 快速效果可能擊破帶謝幕曲選擇的從者；必須等玩家完成整條效果鏈，
    // 才能結算手牌上限並切到對手回合。
    queue(state, { type: "continuePlayerEnd", side: "player", label: "快速效果後繼續結束階段" });
    return;
  }
  finishPlayerEndAfterQuickMutable(state);
}

export function endTurn(input: GameState): GameState {
  const state = clone(input);
  if (state.pending || state.status !== "playing" || state.turnSide !== "player" || state.phase !== "main") return state;
  state.phase = "end";
  for (const miim of state.player.field.filter((item) => item.cardId === "levinMiim")) {
    if (graveLevin(state) >= 5) {
      healLeader(state, "player", 1);
      addLog(state, `${cardName(miim.cardId)}的結束階段效果發動。`);
    }
  }
  const wards = state.player.field.filter((item) => isFollower(item) && !item.tapped && hasKeyword(state, item, "ward"));
  if (wards.length) {
    state.pending = {
      kind: "multi",
      effect: "guardChoice",
      title: "結束階段：守護",
      prompt: "選擇要橫置並進入守護狀態的從者。可以選任意數量，也可以全部不選。",
      options: wards.map((item) => ({ uid: item.uid, cardId: item.cardId })),
      min: 0,
      max: wards.length,
    };
  } else continueEndTurnMutable(state);
  return state;
}

function aiCardValue(cardId: string): number {
  const values: Record<string, number> = {
    greatZelgenea: 50,
    destructionFanatic: 35,
    returningDissonance: 32,
    originalLishenna: 30,
    destructiveLishenna: 28,
    manifestedLishenna: 27,
    axia: 25,
    annihilationSong: 24,
    destructionWilderness: 23,
    dissonanceWorshipper: 22,
    whiteArtifact: 22,
    blackArtifact: 27,
    destructionServant: 20,
    destructionPrayer: 19,
    destructionHermit: 18,
    whiteBlackChapter: 14,
    destructionJoy: 12,
    solo: 10,
    zelgenea: 28,
    fairy: 3,
  };
  return values[cardId] ?? 5;
}

function aiQuickReserve(state: GameState): number {
  const hasQuick = state.ai.hand.some((item) => item.cardId === "whiteBlackChapter");
  if (!hasQuick || state.ai.pp < 2 || state.ai.maxPP < 3) return 0;
  const target = bestFollower(state, "player", "kill");
  if (!target) return 0;
  const kills = remainingHealthOf(target) <= 2;
  const highThreat = attackOf(target) >= 4 || target.cardId === "queenCynthia" || target.cardId === "fairyBladeAmatsu";
  return kills || highThreat ? 2 : 0;
}

function aiPlayScore(state: GameState, card: CardInstance, zone: Zone): number {
  const def = definition(card);
  if (playableReason(state, card, zone, "ai")) return -999;
  let score = aiCardValue(card.cardId) - def.cost * 1.2;
  // 進化目標湊不齊時降低優先度而不是完全卡手：寧可裸出2/2也不要浪費整回合PP。
  if (def.evolveId && !aiCanCommitEvolveFollower(state, card)) score -= 16;
  const target = bestFollower(state, "player");
  const idols = idolField(state).length;
  const idolPermanentAfterPlay = isIdolCard(card.cardId) && (isFollower(card) || isAmulet(card)) ? idols + 1 : idols;
  const reserve = card.cardId === "whiteBlackChapter" ? 0 : aiQuickReserve(state);
  if (reserve && state.ai.pp - def.cost < reserve) score -= 28 * (reserve - (state.ai.pp - def.cost));

  if (card.cardId === "destructionJoy") {
    const refundsPp = idols > 0 && state.ai.pp < state.ai.maxPP;
    const drawsCard = state.ai.field.some((item) => item.cardId === "originalLishenna");
    if (!refundsPp && !drawsCard) return -999;
    score += (refundsPp ? 20 : 0) + (drawsCard ? 18 : 0);
  }
  if (card.cardId === "annihilationSong") score += !eggField(state).length && canFitField(state, "ai") ? 30 : 18;
  if (card.cardId === "manifestedLishenna") score += !eggField(state).length && canFitField(state, "ai") ? 22 : 16;
  if (card.cardId === "destructionWilderness" && idols < 3) score += 8;
  if (card.cardId === "dissonanceWorshipper" && state.ai.hand.length <= 5) score += 8;
  if (card.cardId === "destructionHermit") score += idolPermanentAfterPlay >= 3 && target ? 24 + (aiInDanger(state) ? 12 : 0) : 0;
  if (card.cardId === "destructionFanatic") score += idolPermanentAfterPlay >= 3 && target ? 42 + (aiInDanger(state) ? 12 : 0) : -24;
  // 従者主線：湊滿3偶像→入場免費進化→4/4身材＋2點解場＋2點打臉，是解妖精鋪場的核心。
  if (card.cardId === "destructionServant") score += idolPermanentAfterPlay >= 3 && target ? 34 : 0;
  // 絶傑リーシェナ：解場壓力小才值得拍4費白板；第二張（神器已上線）幾乎多餘。
  if (card.cardId === "originalLishenna") {
    const artifactsOnline = [...state.ai.ex, ...state.ai.field].some((item) => item.cardId === "whiteArtifact" || item.cardId === "blackArtifact")
      || [...state.ai.field, ...state.ai.grave, ...state.ai.banished].some((item) => baseCardId(item) === "originalLishenna");
    if (artifactsOnline) score -= 16;
    if (aiInDanger(state)) score -= 24;
    else if (playerBoardThreat(state) <= 4) score += 8;
  }
  if (card.cardId === "zelgenea" && aiInDanger(state)) score += 15;
  if (card.cardId === "destructiveLishenna") score += idolPermanentAfterPlay >= 3 && state.ai.ex.length < 5 ? 22 : 0;
  if (card.cardId === "whiteBlackChapter") {
    const bonusActive = idols >= 2;
    const lethal = bonusActive && state.player.hp <= 2;
    const copies = state.ai.hand.filter((item) => item.cardId === "whiteBlackChapter").length;
    const killsWorthTarget = target && remainingHealthOf(target) <= 2 && followerValue(state, target) >= 7;
    if (lethal) score += 100;
    else if (copies >= 2 && killsWorthTarget) score += 26;
    else return -999;
  }
  if (card.cardId === "solo") {
    const standingIdols = idolField(state).filter((item) => !item.tapped).length;
    const kills = target && standingIdols * 2 >= remainingHealthOf(target);
    const dangerous = target && (attackOf(target) >= 4 || target.cardId === "queenCynthia" || target.cardId === "fairyBladeAmatsu"
      || (aiInDanger(state) && attackOf(target) >= 2));
    score += target && (kills || dangerous) ? Math.min(42, standingIdols * 7) : -60;
  }
  if (card.cardId === "returningDissonance") score += idols >= 2 ? 15 : -50;
  if (card.cardId === "whiteArtifact") score += state.ai.hp <= 12 ? 28 : state.ai.hp < 20 ? 12 : -4;
  if (card.cardId === "blackArtifact") score += state.player.hp <= 10 ? 34 : 18;
  if (card.cardId === "zelgenea") score += state.ai.hp <= 10 ? 40 : state.player.field.length ? 8 : -8;
  if (card.cardId === "greatZelgenea") score += state.player.hp <= 10 ? 60 : 20;
  if ((isFollower(card) || isAmulet(card)) && state.ai.field.length >= 4) score -= 8;
  return score;
}

function aiCanCommitEvolveFollower(state: GameState, card: CardInstance): boolean {
  const def = definition(card);
  if (!def.evolveId || def.evolveCost === undefined) return true;
  if (state.evolvedThisTurn || (state.ai.evolveRemaining[def.evolveId] ?? 0) <= 0) return false;
  if (card.cardId === "axia" && !idolField(state).length) return false;
  if (card.cardId === "destructionServant" && !bestFollower(state, "player")) return false;

  const ppAfterPlay = state.ai.pp - def.cost;
  const idolCountAfterPlay = idolField(state).length + (isIdolCard(card.cardId) ? 1 : 0);
  const evolveCost = card.cardId === "destructionServant" && idolCountAfterPlay >= 3 ? 0 : def.evolveCost;
  const cheapestPayment = state.ai.ep > 0 ? Math.max(0, evolveCost - 1) : evolveCost;
  return ppAfterPlay >= cheapestPayment;
}

function aiBestPlayable(state: GameState): { card: CardInstance; zone: Zone; score: number } | undefined {
  const options: { card: CardInstance; zone: Zone; score: number }[] = [];
  for (const zone of ["hand", "ex"] as Zone[]) {
    for (const card of ps(state, "ai")[zone]) options.push({ card, zone, score: aiPlayScore(state, card, zone) });
  }
  return options.filter((item) => item.score > 0).sort((a, b) => b.score - a.score || definition(b.card).cost - definition(a.card).cost)[0];
}

function aiEvolvePriority(state: GameState, card: CardInstance): number {
  if (card.baseCardId || !definition(card).evolveId) return -999;
  const target = bestFollower(state, "player");
  if (card.cardId === "destructionServant" && target) return card.flags.freeEvolve ? 52 : 35;
  if (card.cardId === "axia" && aiSacrificeCandidate(state, card.uid)) {
    const hasEgg = eggField(state).length > 0;
    const superTurn = isSuperEligible(state, "ai");
    return 44 + (hasEgg ? 8 : 0) + (superTurn ? 12 : 0);
  }
  if (card.cardId === "destructiveLishenna") return state.ai.ex.length <= 3 ? 43 : 30;
  return -999;
}

function tryAiEvolve(state: GameState): boolean {
  if (state.evolvedThisTurn) return false;
  const candidates = state.ai.field
    .filter(isFollower)
    .map((card) => ({ card, score: aiEvolvePriority(state, card) }))
    .filter((item) => item.score > 0)
    .sort((a, b) => b.score - a.score);
  for (const { card } of candidates) {
    const def = definition(card);
    const originalCost = def.evolveCost ?? 0;
    const free = Boolean(card.flags.freeEvolve);
    if (free) {
      card.flags.originalEvolveCost = originalCost;
      const patched = CARDS[card.cardId];
      const actual = patched.evolveCost;
      patched.evolveCost = 0;
      const result = evolveMutable(state, "ai", card.uid, "pp", isSuperEligible(state, "ai") && card.cardId === "axia");
      patched.evolveCost = actual;
      return result;
    }
    const payment: "pp" | "ep" = state.ai.ep > 0 ? "ep" : "pp";
    if (!canEvolve(state, "ai", card, payment, false)) return evolveMutable(state, "ai", card.uid, payment, isSuperEligible(state, "ai") && card.cardId === "axia");
  }
  return false;
}

function aiUsePrayer(state: GameState, faceOnly = false): boolean {
  const source = state.ai.field.find((item) => item.cardId === "destructionPrayer" && !item.tapped);
  const sacrifice = aiSacrificeCandidate(state, source?.uid);
  if (!source || !sacrifice) return false;
  const target = bestFollower(state, "player", "kill");
  const racing = faceOnly || state.player.hp <= 8;
  const killable = Boolean(target && remainingHealthOf(target) <= 2 && followerValue(state, target) >= 5);
  const shouldUse = killable || racing || state.ai.field.length >= 5;
  if (!shouldUse) return false;
  const hitFace = racing || !killable;
  source.tapped = true;
  moveFieldToGrave(state, sacrifice, "破壊の祈祷者起動費用");
  addEvent(state, "ai", "activate", { cardId: "destructionPrayer", detail: !hitFace && target ? `target=${target.cardId}` : "target=leader" });
  if (!hitFace && target) dealDamageToFollower(state, target, 2, "破壊の祈祷者");
  else damageLeader(state, "player", 2);
  addLog(state, "破壞巫使用破壊の祈祷者的起動能力。 ");
  runTasks(state);
  return true;
}

function aiChapterTapPenalty(state: GameState, card: CardInstance): number {
  if (card.cardId === "destructionWilderness") return 1;
  if (card.cardId === "manifestedLishenna") return 2;
  if (card.cardId === "destructionHermit" || card.cardId === "dissonanceWorshipper") return 3;
  if (card.cardId === "whiteArtifact") return state.ai.hp <= 12 ? 15 : 8;
  if (card.cardId === "blackArtifact") return state.player.hp <= 10 ? 18 : 11;
  if (card.cardId === "destructionPrayer") return 16;
  if (card.cardId === "axiaEvo") return 20;
  if (isFollower(card) && !canAttackNow(state, card)) return 2 + attackOf(card);
  if (isFollower(card)) return 7 + attackOf(card) * 4;
  return 6;
}

function aiChapterEffectValue(state: GameState, chapter: CardInstance): number {
  let value = chapter.cardId === "newBlack" ? 2 : state.ai.hp < 20 ? 1.5 : 0.5;
  const axiaReady = state.ai.field.some((item) => item.cardId === "axiaEvo" && !item.flags.axiaTriggered);
  if (axiaReady && state.player.field.some(isFollower)) value += 3;
  if (chapter.cardId === "newBlack" && state.player.hp <= 8) value += 4;
  if (chapter.cardId === "newBlack" && state.player.hp <= 3) value += 20;
  if (chapter.cardId === "newWhite" && state.ai.hp <= 8) value += 10;
  return value;
}

function aiCycleChapter(state: GameState, preserveAttackers = true): boolean {
  const chapter = [...state.ai.field]
    .filter((item) => (item.cardId === "newWhite" || item.cardId === "newBlack") && !item.tapped)
    .sort((a, b) => Number(b.cardId === "newBlack") - Number(a.cardId === "newBlack"))[0];
  const standIdols = idolField(state).filter((item) => !item.tapped);
  if (!chapter || standIdols.length < 3) return false;
  const otherCosts = standIdols
    .filter((item) => item.uid !== chapter.uid)
    .sort((a, b) => aiChapterTapPenalty(state, a) - aiChapterTapPenalty(state, b));
  const costs = [chapter, ...otherCosts.slice(0, 2)];
  const lostAttack = costs
    .filter((item) => item.uid !== chapter.uid && canAttackNow(state, item))
    .reduce((total, item) => total + attackOf(item), 0);
  if (preserveAttackers && lostAttack > aiChapterEffectValue(state, chapter)) return false;
  for (const card of costs) card.tapped = true;
  const payment = costs.map((card) => cardName(card.cardId)).join("、");
  addEvent(state, "ai", "eggCycle", { cardId: chapter.cardId, detail: `taps=${payment}` });
  moveFieldToGrave(state, chapter, `${cardName(chapter.cardId)}起動費用`);
  addLog(state, `破壞巫橫置「${payment}」3張偶像卡牌，破壞${chapter.cardId === "newBlack" ? "黑蛋" : "白蛋"}以觸發謝幕曲。`);
  runTasks(state);
  return true;
}

function aiUseWilderness(state: GameState): boolean {
  const wild = state.ai.field.find((item) => item.cardId === "destructionWilderness");
  const sacrifice = aiSacrificeCandidate(state, wild?.uid);
  if (!wild || !sacrifice || state.ai.field.length < 5) return false;
  addEvent(state, "ai", "activate", { cardId: "destructionWilderness", detail: `sacrifice=${sacrifice.cardId}` });
  moveFieldToGrave(state, wild, "破壊の荒野起動費用");
  moveFieldToGrave(state, sacrifice, "破壊の荒野起動費用");
  addLog(state, "破壞巫起動破壊の荒野，清出場地空間。 ");
  runTasks(state);
  return true;
}

function aiGraveWorld(state: GameState): boolean {
  if (state.ai.pp < 10 || state.ai.ex.length >= 5 || (state.ai.evolveRemaining.greatZelgenea ?? 0) <= 0) return false;
  const world = state.ai.grave.find((item) => item.cardId === "zelgenea");
  if (!world) return false;
  state.ai.pp -= 10;
  moveCardToBanished(state, "ai", world, "grave");
  state.ai.evolveRemaining.greatZelgenea -= 1;
  state.ai.evolveUsed.greatZelgenea = (state.ai.evolveUsed.greatZelgenea ?? 0) + 1;
  addExDirect(state, "ai", ["greatZelgenea"]);
  addLog(state, "破壞巫從墓場消滅《世界》・ゼルガネイア，將大いなる《世界》放入EX。 ");
  addEvent(state, "ai", "activate", { cardId: "zelgenea", detail: "greatZelgenea→EX" });
  return true;
}

export type ManualAiActivation = {
  key: string;
  abilityId: "prayer" | "wilderness" | "chapterCycle" | "graveWorld";
  sourceUid: string;
  selected: string[];
  label: string;
};

/** 破壞方主階段的所有非攻擊起動能力；複合費用與目標直接列成候選，避免策略背後仍藏著腳本選擇。 */
export function manualAiActivations(state: GameState): ManualAiActivation[] {
  if (
    state.aiControl !== "manual" || state.status !== "playing" || state.turnSide !== "ai"
    || state.phase !== "main" || state.pending
  ) return [];
  const actions: ManualAiActivation[] = [];
  const idols = idolField(state, "ai");
  for (const source of state.ai.field.filter((item) => item.cardId === "destructionPrayer" && !item.tapped)) {
    for (const sacrifice of idols.filter((item) => item.uid !== source.uid)) {
      const targets: ChoiceOption[] = [
        { uid: "player-leader", label: "玩家主戰者" },
        ...state.player.field.filter(isFollower).map((item) => ({ uid: item.uid, cardId: item.cardId })),
      ];
      for (const target of targets) actions.push({
        key: `prayer:${source.uid}:${sacrifice.uid}:${target.uid}`,
        abilityId: "prayer",
        sourceUid: source.uid,
        selected: [sacrifice.uid, target.uid],
        label: `祈禱者：犧牲${cardName(sacrifice.cardId)}，對${target.cardId ? cardName(target.cardId) : "主戰者"}造成2點`,
      });
    }
  }
  for (const source of state.ai.field.filter((item) => item.cardId === "destructionWilderness")) {
    for (const sacrifice of idols.filter((item) => item.uid !== source.uid)) actions.push({
      key: `wilderness:${source.uid}:${sacrifice.uid}`,
      abilityId: "wilderness",
      sourceUid: source.uid,
      selected: [sacrifice.uid],
      label: `破壞荒野與${cardName(sacrifice.cardId)}`,
    });
  }
  for (const source of state.ai.field.filter((item) => (item.cardId === "newWhite" || item.cardId === "newBlack") && !item.tapped)) {
    const others = idols.filter((item) => item.uid !== source.uid && !item.tapped);
    for (let left = 0; left < others.length; left += 1) {
      for (let right = left + 1; right < others.length; right += 1) actions.push({
        key: `chapterCycle:${source.uid}:${others[left].uid}:${others[right].uid}`,
        abilityId: "chapterCycle",
        sourceUid: source.uid,
        selected: [others[left].uid, others[right].uid],
        label: `${cardName(source.cardId)}：橫置${cardName(others[left].cardId)}、${cardName(others[right].cardId)}後破壞`,
      });
    }
  }
  const world = state.ai.grave.find((item) => item.cardId === "zelgenea");
  if (world && state.ai.pp >= 10 && state.ai.ex.length < 5 && (state.ai.evolveRemaining.greatZelgenea ?? 0) > 0) {
    actions.push({
      key: `graveWorld:${world.uid}`,
      abilityId: "graveWorld",
      sourceUid: world.uid,
      selected: [],
      label: "10PP：從墓場消滅《世界》，將大世界放入EX",
    });
  }
  return actions;
}

export function activateManualAi(input: GameState, activation: ManualAiActivation): GameState {
  const state = clone(input);
  const legal = manualAiActivations(state).find((candidate) => candidate.key === activation.key);
  if (!legal) return state;
  const source = state.ai.field.find((item) => item.uid === legal.sourceUid);
  if (legal.abilityId === "graveWorld") {
    aiGraveWorld(state);
    return state;
  }
  if (!source) return state;
  if (legal.abilityId === "prayer") {
    const sacrifice = state.ai.field.find((item) => item.uid === legal.selected[0]);
    const targetUid = legal.selected[1];
    if (!sacrifice) return state;
    source.tapped = true;
    moveFieldToGrave(state, sacrifice, "破壊の祈祷者起動費用");
    addEvent(state, "ai", "activate", { cardId: source.cardId, detail: `target=${targetUid}` });
    if (targetUid === "player-leader") damageLeader(state, "player", 2);
    else {
      const target = state.player.field.find((item) => item.uid === targetUid);
      if (target) dealDamageToFollower(state, target, 2, "破壊の祈祷者");
    }
  } else if (legal.abilityId === "wilderness") {
    const sacrifice = state.ai.field.find((item) => item.uid === legal.selected[0]);
    if (!sacrifice) return state;
    addEvent(state, "ai", "activate", { cardId: source.cardId, detail: `sacrifice=${sacrifice.cardId}` });
    moveFieldToGrave(state, source, "破壊の荒野起動費用");
    moveFieldToGrave(state, sacrifice, "破壊の荒野起動費用");
  } else {
    const costs = [source, ...legal.selected.map((uid) => state.ai.field.find((item) => item.uid === uid)).filter(Boolean)] as CardInstance[];
    if (costs.length !== 3) return state;
    for (const card of costs) card.tapped = true;
    addEvent(state, "ai", "eggCycle", { cardId: source.cardId, detail: `taps=${costs.map((card) => card.cardId).join(",")}` });
    moveFieldToGrave(state, source, `${cardName(source.cardId)}起動費用`);
  }
  runTasks(state);
  return state;
}

export function endTurnForSide(input: GameState, side: Side): GameState {
  if (side === "player") return endTurn(input);
  const state = clone(input);
  if (
    state.aiControl !== "manual" || state.pending || state.status !== "playing"
    || state.turnSide !== "ai" || state.phase !== "main"
  ) return state;
  aiEndPhase(state);
  return state;
}

function declareAiAttack(state: GameState, attacker: CardInstance, targetUid: string): boolean {
  if (!canAttackNow(state, attacker)) return false;
  const legal = attackTargets(state, attacker).map((item) => item.uid);
  if (!legal.includes(targetUid)) return false;
  attacker.tapped = true;
  if (targetUid === "player-leader") {
    addLog(state, `${cardName(attacker.cardId)}攻擊你的主戰者。`);
    addEvent(state, "ai", "attack", { cardId: attacker.cardId, detail: `atk=${attackOf(attacker)} target=leader` });
    return true;
  }
  const target = state.player.field.find((item) => item.uid === targetUid);
  if (!target) return false;
  addLog(state, `${cardName(attacker.cardId)}攻擊${cardName(target.cardId)}。`);
  addEvent(state, "ai", "attack", { cardId: attacker.cardId, detail: `atk=${attackOf(attacker)} target=${target.cardId} targetHp=${remainingHealthOf(target)}` });
  return true;
}

function resolveAiAttackCombat(state: GameState, attackerUid: string, targetUid: string): void {
  const attacker = state.ai.field.find((item) => item.uid === attackerUid);
  if (!attacker) {
    addLog(state, "攻擊從者在傷害結算前離場，本次攻擊不造成傷害。");
    return;
  }
  if (targetUid === "player-leader") {
    damageLeader(state, "player", attackOf(attacker));
    return;
  }
  const target = state.player.field.find((item) => item.uid === targetUid);
  if (!target) {
    addLog(state, "原攻擊目標在傷害結算前離場，本次攻擊不重新選擇目標。");
    return;
  }
  const attackerDamage = attackOf(target);
  const targetDamage = attackOf(attacker);
  dealDamageToFollower(state, target, targetDamage, "交戰");
  const still = state.ai.field.find((item) => item.uid === attacker.uid);
  if (still) dealDamageToFollower(state, still, attackerDamage, "交戰");
}

function followerValue(state: GameState, card: CardInstance): number {
  return attackOf(card) * 2 + remainingHealthOf(card)
    + (hasKeyword(state, card, "ward") ? 4 : 0)
    + (card.cardId === "queenCynthia" ? 8 : 0)
    + (card.cardId === "fairyBladeAmatsu" ? 4 : 0)
    + (card.cardId === "levinAlbert" ? 8 : 0)
    + (card.cardId === "levinMeim" ? 4 : 0)
    + (card.flags.permStorm ? 6 : 0);
}

function playerActWards(state: GameState): CardInstance[] {
  return state.player.field.filter((item) => isFollower(item) && item.tapped && hasKeyword(state, item, "ward"));
}

// 玩家場面下回合能打出的臉傷估計（妖精配合シンシア/妖精郷幾乎全員能動）。
function playerBoardThreat(state: GameState): number {
  return state.player.field.filter(isFollower).reduce((total, item) => total + attackOf(item), 0);
}

// 解場壓力大：對面場面攻擊力已經逼近我方血量，再不解場就會被妖精沖死。
function aiInDanger(state: GameState): boolean {
  return playerBoardThreat(state) + 2 >= state.ai.hp;
}

// 撈リーシェナ的優先度：有壓力時要身材與解場（破壊→顕現→絶傑）；
// 沒壓力才撈絶傑鋪神器，且神器已上線時第二張絶傑降到最低。
function aiLishennaFetchPriority(state: GameState, cardId: string): number {
  const pressured = aiInDanger(state) || playerBoardThreat(state) >= 6;
  const priority: Record<string, number> = pressured
    ? { destructiveLishenna: 12, manifestedLishenna: 11, originalLishenna: 10 }
    : { originalLishenna: 12, destructiveLishenna: 11, manifestedLishenna: 10 };
  let value = priority[cardId] ?? 0;
  if (cardId === "originalLishenna") {
    const artifactsOnline = [...state.ai.ex, ...state.ai.field].some((item) => item.cardId === "whiteArtifact" || item.cardId === "blackArtifact")
      || [...state.ai.field, ...state.ai.grave, ...state.ai.banished].some((item) => baseCardId(item) === "originalLishenna");
    if (artifactsOnline) value -= 8;
  }
  return value;
}

// 若本回合全力打臉，攻擊階段能對玩家主戰者造成多少傷害（先扣掉必須清除的橫置守護）。
function aiAttackFaceDamage(state: GameState): number {
  const attackers = state.ai.field.filter((item) => isFollower(item) && canAttackNow(state, item));
  const faceCapable = attackers.filter((item) => item.enteredAt < state.globalTurn || hasKeyword(state, item, "storm"));
  const wardOnly = attackers.filter((item) => !faceCapable.some((hit) => hit.uid === item.uid));
  const wardBreakers = wardOnly.map(attackOf).sort((a, b) => b - a);
  const facePool = faceCapable.map(attackOf).filter((atk) => atk > 0).sort((a, b) => a - b);
  for (const ward of playerActWards(state)) {
    let left = remainingHealthOf(ward);
    while (left > 0) {
      const hit = wardBreakers.shift() ?? facePool.shift();
      if (hit === undefined) return 0;
      left -= hit;
    }
  }
  return facePool.reduce((total, atk) => total + atk, 0);
}

// 攻擊以外、本回合還能擠出的直傷（祈祷者、黑蛋循環、快速章、EX中的大世界）。
// 注意不與 aiAttackFaceDamage 重複計算：能打臉的祈祷者只算攻擊那份。
function aiBurnReach(state: GameState): number {
  let reach = 0;
  const prayers = state.ai.field.filter((item) =>
    item.cardId === "destructionPrayer" && !item.tapped &&
    !(canAttackNow(state, item) && item.enteredAt < state.globalTurn),
  );
  const spareSacrifices = idolField(state).filter((item) => item.cardId !== "destructionPrayer").length;
  reach += Math.min(prayers.length, spareSacrifices) * 2;
  // 蛋循環發生在攻擊之後：只數攻擊後仍會直立的偶像卡。
  const idleIdols = idolField(state).filter((item) =>
    !item.tapped && !(isFollower(item) && canAttackNow(state, item) && attackOf(item) > 0),
  );
  const blackEgg = state.ai.field.some((item) => item.cardId === "newBlack" && !item.tapped);
  if (blackEgg && idleIdols.length >= 3) reach += 1;
  if (
    state.ai.hand.some((item) => item.cardId === "whiteBlackChapter") &&
    state.ai.pp >= 2 &&
    idolField(state).length >= 2 &&
    state.player.field.some(isFollower)
  ) reach += 2;
  if (state.ai.ex.some((item) => item.cardId === "greatZelgenea")) reach += 4;
  return reach;
}

function aiLethalInSight(state: GameState): boolean {
  return aiAttackFaceDamage(state) + aiBurnReach(state) >= state.player.hp;
}

function chooseAiAttackTarget(state: GameState, attacker: CardInstance): string | undefined {
  if (attackOf(attacker) <= 0) return undefined;
  const targets = attackTargets(state, attacker);
  if (!targets.length) return undefined;
  const face = targets.find((option) => option.uid === "player-leader");
  const followerTargets = targets
    .map((option) => state.player.field.find((item) => item.uid === option.uid))
    .filter(Boolean) as CardInstance[];

  // 斬殺在望或已進入收割節奏：全部打臉（守護會被 attackTargets 強制排進目標）。
  const racing = aiLethalInSight(state) || state.player.hp <= 7;
  if (racing && face) return face.uid;

  const kills = followerTargets
    .filter((target) => attackOf(attacker) >= remainingHealthOf(target))
    .map((target) => {
      const attackerDies = attackOf(target) >= remainingHealthOf(attacker);
      return { target, net: followerValue(state, target) - (attackerDies ? followerValue(state, attacker) : 0) };
    })
    .sort((a, b) => b.net - a.net);
  const bestKill = kills[0];

  if (aiInDanger(state)) {
    const defensiveKill = [...kills].sort((a, b) => attackOf(b.target) - attackOf(a.target))[0];
    const chip = [...followerTargets].sort((a, b) => attackOf(b) - attackOf(a))[0];
    const defensiveTarget = defensiveKill?.target ?? chip;
    if (defensiveTarget) return defensiveTarget.uid;
  }

  if (bestKill && bestKill.net > 0) {
    const targetDef = definition(baseCardId(bestKill.target));
    const cheapToken = Boolean(targetDef.token) && attackOf(bestKill.target) <= 2;
    const bigBody = attackOf(attacker) >= 3;
    if (!(cheapToken && bigBody && face)) return bestKill.target.uid;
  }
  if (face) return face.uid;
  const fallback = bestKill?.target ?? followerTargets.sort((a, b) => remainingHealthOf(a) - remainingHealthOf(b))[0];
  return fallback?.uid;
}

function standingDukes(state: GameState): CardInstance[] {
  return state.player.field.filter((item) => baseCardId(item) === "levinDuke" && isFollower(item) && !item.tapped);
}

function queueAiAttackContinuation(state: GameState, remaining: string[], finish: boolean): void {
  if (state.status !== "playing" || state.turnSide !== "ai") return;
  if (remaining.length) queue(state, { type: "aiAttackStep", side: "ai", data: { remaining, finish } });
  else if (finish) queue(state, { type: "aiPostAttack", side: "ai" });
}

function beginAiAttackSequence(state: GameState, finish: boolean): void {
  const remaining = state.ai.field
    .filter((item) => isFollower(item) && canAttackNow(state, item))
    .sort((a, b) => attackOf(b) - attackOf(a))
    .map((item) => item.uid);
  queueFront(state, { type: "aiAttackStep", side: "ai", data: { remaining, finish } });
  runTasks(state);
}

function aiAttackPhase(state: GameState): void {
  beginAiAttackSequence(state, false);
}

function aiDiscardToSeven(state: GameState): void {
  while (state.ai.hand.length > 7) {
    const card = [...state.ai.hand].sort((a, b) => aiCardValue(a.cardId) - aiCardValue(b.cardId))[0];
    removeFromZone(state.ai, "hand", card.uid);
    card.zone = "grave";
    if (!cardIsToken(card)) state.ai.grave.push(card);
    addLog(state, `破壞巫因手牌上限捨棄${cardName(card.cardId)}。`);
  }
}

function finishAiEndPhase(state: GameState): void {
  if (state.status !== "playing" || state.turnSide !== "ai") return;
  if (state.aiControl === "manual" && state.ai.hand.length > 7) {
    const excess = state.ai.hand.length - 7;
    state.pending = {
      kind: "multi",
      effect: "aiDiscardToSeven",
      title: "破壞方手牌上限",
      prompt: `結束階段手牌超過7張，選擇${excess}張捨棄。`,
      options: state.ai.hand.map((item) => ({ uid: item.uid, cardId: item.cardId })),
      min: excess,
      max: excess,
      side: "ai",
    };
    return;
  }
  aiDiscardToSeven(state);
  if (state.status === "playing") finishTurnSwitchMutable(state);
}

function aiEndPhase(state: GameState): void {
  state.phase = "end";
  const world = state.ai.ex.find((item) => item.cardId === "greatZelgenea");
  if (world) {
    damageLeader(state, "player", 4);
    const taskCountBefore = state.tasks.length;
    for (const follower of [...state.player.field.filter(isFollower)]) dealDamageToFollower(state, follower, 4, "大いなる《世界》結束階段");
    groupNewPlayerTasks(state, taskCountBefore, "大いなる《世界》造成的同時離場效果");
  }
  if (state.status === "playing") queue(state, { type: "aiEndQuick", side: "player", label: "對方結束階段的快速時機" });
  runTasks(state);
}

function runAiTurnMutable(state: GameState): void {
  if (state.status !== "playing" || state.turnSide !== "ai" || state.pending) return;
  state.phase = "ai";
  runTasks(state);
  let guard = 0;
  while (state.status === "playing" && state.turnSide === "ai" && !state.pending && guard < 40) {
    guard += 1;
    if (aiGraveWorld(state)) continue;
    if (tryAiEvolve(state)) {
      runTasks(state);
      continue;
    }
    if (aiUsePrayer(state)) continue;
    if (aiUseWilderness(state)) continue;
    const best = aiBestPlayable(state);
    if (best && playCardMutable(state, "ai", best.card.uid, best.zone, `score=${best.score.toFixed(1)}`)) {
      runTasks(state);
      continue;
    }
    if (aiCycleChapter(state, true)) continue;
    break;
  }
  // 斬殺執行：直傷加總足以擊殺時，把保留的快速章與祈祷者先灌臉，攻擊階段會自動全打臉。
  if (state.status === "playing" && aiLethalInSight(state)) {
    const chapter = state.ai.hand.find((item) => item.cardId === "whiteBlackChapter");
    if (chapter && state.ai.pp >= 2 && idolField(state).length >= 2 && state.player.field.some(isFollower)) {
      playCardMutable(state, "ai", chapter.uid, "hand", "lethal-reach");
      runTasks(state);
    }
    let prayerGuard = 0;
    while (state.status === "playing" && prayerGuard < 4 && aiUsePrayer(state, true)) prayerGuard += 1;
  }
  if (state.status === "playing" && state.turnSide === "ai" && state.pending) {
    if (!state.tasks.some((task) => task.type === "aiResumeMain")) queue(state, { type: "aiResumeMain", side: "ai" });
    return;
  }
  if (state.status === "playing" && state.turnSide === "ai") beginAiAttackSequence(state, true);
}

function finishTurnSwitchMutable(state: GameState): void {
  if (state.status !== "playing") return;
  const ending = state.turnSide;
  const next = otherSide(ending);
  addLog(state, `${ending === "player" ? "你的" : "破壞巫的"}回合結束。`);
  beginTurnMutable(state, next);
  runTasks(state);
  if (next === "ai" && state.aiControl === "scripted" && !state.pending) runAiTurnMutable(state);
}

export function restartWithSameSeed(state: GameState, playerFirst = state.playerFirst): GameState {
  return createGame(playerFirst, state.seed, state.playerDeck, { aiControl: state.aiControl });
}

export function publicSummary(state: GameState) {
  return {
    seed: state.seed,
    turn: state.globalTurn,
    side: state.turnSide,
    player: { hp: state.player.hp, pp: state.player.pp, maxPP: state.player.maxPP, ep: state.player.ep, sep: state.player.sep, deck: state.player.deck.length },
    ai: { hp: state.ai.hp, pp: state.ai.pp, maxPP: state.ai.maxPP, ep: state.ai.ep, sep: state.ai.sep, deck: state.ai.deck.length, hand: state.ai.hand.length },
  };
}

export const __testing = {
  makeInstance,
  putExistingIntoField,
  moveFieldToGrave,
  addExDirect,
  addExTask,
  runTasks,
  resolveTask,
  aiCycleChapter,
  aiBestPlayable,
  aiPlayScore,
  aiCanCommitEvolveFollower,
  tryAiEvolve,
  runAiTurnMutable,
  beginTurnMutable,
  aiQuickWindow,
  aiUsePrayer,
  aiAttackPhase,
  aiEndPhase,
  aiAttackFaceDamage,
  aiBurnReach,
  aiLethalInSight,
};
