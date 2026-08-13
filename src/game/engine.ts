import {
  AI_DECK,
  AI_EVOLVE,
  CARDS,
  PLAYER_DECK,
  PLAYER_EVOLVE,
  cardName,
  isFairyCard,
  isIdolCard,
  isLishenna,
  type CardDef,
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
  data?: Record<string, unknown>;
};

export type GameState = {
  version: 1;
  seed: number;
  rng: number;
  uidCounter: number;
  status: "mulligan" | "playing" | "gameover";
  winner?: Side | "draw";
  playerFirst: boolean;
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

function drawOne(state: GameState, side: Side): CardInstance | undefined {
  const player = ps(state, side);
  const top = player.deck.pop();
  if (!top) {
    state.status = "gameover";
    state.winner = otherSide(side);
    addLog(state, `${side === "player" ? "你" : "破壞巫"}無牌可抽，敗北。`);
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
  if (card.owner === "player" && definition(card).token && isFairyCard(card.cardId)) {
    if (keyword === "rush" && miasmaActive(state)) return true;
    if (keyword === "designated" && state.player.field.some((item) => item.cardId === "fairyland")) return true;
    if (keyword === "storm" && state.player.field.some((item) => item.cardId === "queenCynthia")) return true;
    if (keyword === "designated" && state.player.field.some((item) => item.cardId === "queenCynthia")) return true;
  }
  return false;
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

export function createGame(playerFirst: boolean, seed = Date.now()): GameState {
  const state: GameState = {
    version: 1,
    seed: seed >>> 0,
    rng: (seed || 0x9e3779b9) >>> 0,
    uidCounter: 0,
    status: "mulligan",
    playerFirst,
    turnSide: playerFirst ? "player" : "ai",
    globalTurn: 0,
    phase: "setup",
    player: emptyPlayer(PLAYER_EVOLVE),
    ai: emptyPlayer(AI_EVOLVE),
    playedThisTurn: 0,
    evolvedThisTurn: false,
    tasks: [],
    log: [],
  };
  createSide(state, "player", PLAYER_DECK, PLAYER_EVOLVE);
  createSide(state, "ai", AI_DECK, AI_EVOLVE);
  state.player.ep = playerFirst ? 0 : 3;
  state.ai.ep = playerFirst ? 3 : 0;
  addLog(state, `遊戲種子：${state.seed}。你選擇${playerFirst ? "先攻" : "後攻"}。`);
  return state;
}

function aiWantsMulligan(state: GameState): boolean {
  const hand = state.ai.hand.map((item) => item.cardId);
  const hasEarly = hand.some((id) => CARDS[id].cost <= 2 && id !== "returningDissonance");
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
  if (state.status !== "mulligan") return state;
  if (redraw) {
    redrawHand(state, "player");
    addLog(state, "你將起手4張全部放到牌庫底並重抽。―");
  } else {
    addLog(state, "你保留起手。―");
  }
  if (aiWantsMulligan(state)) {
    redrawHand(state, "ai");
    addLog(state, "破壞巫選擇重抽起手。―");
  } else addLog(state, "破壞巫保留起手。―");
  state.status = "playing";
  state.phase = "main";
  beginTurnMutable(state, state.turnSide);
  runTasks(state);
  if (state.turnSide === "ai" && !state.pending) runAiTurnMutable(state);
  return state;
}

function beginTurnMutable(state: GameState, side: Side): void {
  if (state.status !== "playing") return;
  state.globalTurn += 1;
  state.turnSide = side;
  state.phase = side === "player" ? "main" : "ai";
  state.playedThisTurn = 0;
  state.evolvedThisTurn = false;
  for (const card of [...state.player.field, ...state.ai.field]) {
    delete card.flags.axiaTriggered;
    delete card.flags.freeEvolve;
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
  if (side === "ai") {
    for (const card of player.field) {
      if (card.cardId === "whiteArtifact") queue(state, { type: "heal", side: "ai", amount: 2 });
      if (card.cardId === "blackArtifact") queue(state, { type: "leaderDamage", side: "player", amount: 2 });
    }
  }
}

function bestFollower(state: GameState, side: Side, mode: "kill" | "threat" = "threat"): CardInstance | undefined {
  const followers = ps(state, side).field.filter(isFollower);
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
    manifestedLishenna: 15, destructionWilderness: 14, destructiveLishenna: 13,
    annihilationSong: 12, dissonanceWorshipper: 11, destructionFanatic: 10,
    originalLishenna: 9, whiteBlackChapter: 8,
  };
  const picked = [...candidates].sort((a, b) => (priority[b.cardId] ?? 5) - (priority[a.cardId] ?? 5))[0];
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
    default:
      break;
  }
}

function aiSacrificeCandidate(state: GameState, excludeUid?: string): CardInstance | undefined {
  const value: Record<string, number> = {
    newWhite: 2, newBlack: 2, destructionWilderness: 3,
    manifestedLishenna: 4, destructionHermit: 5, destructionPrayer: 7,
    axiaEvo: 10, destructiveLishennaEvo: 10,
  };
  return state.ai.field
    .filter((item) => item.uid !== excludeUid && isIdolCard(item.cardId))
    .sort((a, b) => (value[a.cardId] ?? 6) - (value[b.cardId] ?? 6))[0];
}

function resolveAiFanfare(state: GameState, source: CardInstance | undefined, cardId: string): void {
  const target = bestFollower(state, "player");
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
      const target = bestFollower(state, otherSide(side), "kill");
      if (target) dealDamageToFollower(state, target, task.amount ?? 0, taskLabel(task));
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
      const target = bestFollower(state, otherSide(side), "kill");
      if (target) {
        dealDamageToFollower(state, target, 2, "破壊の従者進化時");
        damageLeader(state, otherSide(side), 2);
      }
      break;
    }
    case "axiaEvolve": {
      const source = actionSource(state, task.sourceUid);
      const sacrifice = aiSacrificeCandidate(state, source?.uid);
      if (sacrifice) {
        moveFieldToGrave(state, sacrifice, "アクシア進化費用");
        const choices = searchDeckInstances(state, side, (item) => isLishenna(item.cardId));
        const priority: Record<string, number> = { originalLishenna: 12, destructiveLishenna: 11, manifestedLishenna: 10 };
        const picked = choices.sort((a, b) => (priority[b.cardId] ?? 0) - (priority[a.cardId] ?? 0))[0];
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
    case "lishennaEvolve": {
      const cards = topCards(state, side, 4);
      const capacity = Math.max(0, 5 - ps(state, side).ex.length);
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
  if (side === "player" && state.phase !== "main") return "目前不是你的主階段";
  if (ps(state, side).pp < definition(card).cost) return "PP不足";
  if ((isFollower(card) || isAmulet(card)) && !canFitField(state, side)) return "場上5格已滿";
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

function playCardMutable(state: GameState, side: Side, uid: string, zone: Zone): boolean {
  const owned = findOwned(state, side, uid);
  if (!owned || owned.zone !== zone) return false;
  const reason = playableReason(state, owned.card, zone, side);
  if (reason) return false;
  const card = removePlayedCard(state, side, zone, uid);
  if (!card) return false;
  ps(state, side).pp -= definition(card).cost;
  state.playedThisTurn += 1;
  const primary: Task = definition(card).kind === "spell"
    ? { type: "spell", side, sourceUid: card.uid, cardId: card.cardId, label: `${cardName(card.cardId)}的效果` }
    : { type: "fanfare", side, sourceUid: card.uid, cardId: card.cardId, label: `${cardName(card.cardId)}的入場曲` };

  if (definition(card).kind === "spell") {
    card.zone = "grave";
    if (!cardIsToken(card)) ps(state, side).grave.push(card);
  } else {
    putExistingIntoField(state, card, side, false);
  }
  addLog(state, `${side === "player" ? "你" : "破壞巫"}使用${cardName(card.cardId)}（支付${definition(card).cost}PP）。`);

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
  playCardMutable(state, "player", uid, zone);
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
  if (side !== "ai") return;
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
      const priority: Record<string, number> = { originalLishenna: 12, destructiveLishenna: 11, manifestedLishenna: 10 };
      const picked = choices.sort((a, b) => (priority[b.cardId] ?? 0) - (priority[a.cardId] ?? 0))[0];
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
    const tappable = idolField(state).filter((item) => !item.tapped).slice(0, needed);
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
  const ppCost = def.evolveCost - (payment === "ep" ? 1 : 0);
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
  const evolveCost = oldDef.evolveCost!;
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

  const tasks: Task[] = [];
  if (evolveId === "naturalAriaEvo") tasks.push({ type: "spawn", side, cardId: "fairy", label: "自然の妖精姫・アリア的進化時" });
  if (evolveId === "forestFairyEvo") tasks.push({ type: "addEx", side, cardIds: ["fairyWisp", "fairy"], data: { after: [{ type: "forestHealCheck", side }] }, label: "フォレストフェアリー的進化時" });
  if (evolveId === "reverseAmatsuEvo") tasks.push({ type: "reverseEvolve", side, sourceUid: uid, label: "リバースブレイダー・アマツ的進化時" });
  if (evolveId === "miasmaAriaEvo") tasks.push({ type: "miasmaEvolve", side, sourceUid: uid, label: "瘴気の妖精姫・アリア的進化時" });
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
  addLog(state, `${cardName(attacker.cardId)}宣告攻擊${targetUid === "ai-leader" ? "破壞巫主戰者" : "從者"}。`);
  aiQuickWindow(state, attacker);
  runTasks(state);
  if (!state.player.field.some((item) => item.uid === attackerUid) || state.status === "gameover") return;
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
  const targets = opponent.field.filter((item) => isFollower(item) && (item.tapped || canHitStand));
  const options: ChoiceOption[] = targets.map((item) => ({ uid: item.uid, cardId: item.cardId }));
  if (canFace) options.unshift({ uid: `${otherSide(attacker.owner)}-leader`, label: otherSide(attacker.owner) === "ai" ? "破壞巫主戰者" : "你的主戰者" });
  return options;
}

function canActivate(state: GameState, card: CardInstance, actionId: string): string | undefined {
  if (state.turnSide !== "player" || state.phase !== "main") return "只能在自己的主階段使用";
  if (card.tapped) return "卡片已經橫置";
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
  if ((zone === "hand" || zone === "ex") && side === "player") {
    const reason = playableReason(state, card, zone, side);
    actions.push({ id: "play", label: "出場／使用", enabled: !reason, reason });
  }
  if (zone !== "field" || side !== "player") return actions;
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
  const actByCard: Record<string, [string, string]> = {
    fairyBladeAmatsu: ["amatsuStorm", "橫置：給最多2體妖精疾走"],
    bouquetFairy: ["bouquetBounce", "橫置＋消滅EX：敵方從者返回手牌"],
    riotousGarden: ["gardenDamage", "橫置＋置入墓場：造成妖精數量傷害"],
    wonderTree: ["wonderDraw", "橫置＋置入墓場：抽2張"],
    wingQueen: ["wingDestroy", "1PP＋橫置＋置入墓場：破壞敵方從者"],
  };
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
    moveFieldToGrave(state, source, "ワンダーツリー起動費用");
    drawCards(state, "player", 2);
    addLog(state, "ワンダーツリー使你抽2張。 ");
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
        bounceFollower(state, target);
      }
      break;
    }
    case "gardenTarget": {
      const source = state.player.field.find((item) => item.uid === pending.data?.sourceUid);
      const target = state.ai.field.find((item) => item.uid === picked[0]);
      if (source && target) {
        source.tapped = true;
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
        moveFieldToGrave(state, source, "翅の女王・ティターニア起動費用");
        destroyFollower(state, target, "翅の女王・ティターニア");
        spawnToken(state, "ai", "fairy");
      }
      break;
    }
    case "attackTarget":
      resolveAttackMutable(state, pending.data?.sourceUid as string, picked[0]);
      break;
    case "guardChoice":
      for (const uid of picked) {
        const ward = state.player.field.find((item) => item.uid === uid);
        if (ward && hasKeyword(state, ward, "ward") && !ward.tapped) ward.tapped = true;
      }
      addLog(state, `你讓${picked.length}體守護從者進入橫置守護狀態。`);
      continueEndTurnMutable(state);
      break;
    case "discardToSeven":
      for (const uid of picked) {
        const card = removeFromZone(state.player, "hand", uid);
        if (!card) continue;
        card.zone = "grave";
        if (!cardIsToken(card)) state.player.grave.push(card);
        addLog(state, `手牌上限：你捨棄${cardName(card.cardId)}。`);
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
  const kills = remainingHealthOf(target) <= 2;
  const highThreat = attackOf(target) >= 4 || target.cardId === "queenCynthia" || target.cardId === "fairyBladeAmatsu";
  if (!kills && !highThreat) return;
  removeFromZone(state.ai, "hand", quick.uid);
  quick.zone = "grave";
  state.ai.grave.push(quick);
  state.ai.pp -= 2;
  addLog(state, `破壞巫在快速時機使用${cardName(quick.cardId)}，目標為${cardName(target.cardId)}。`);
  dealDamageToFollower(state, target, 2, "白の章・黒の章（快速）");
  if (idolField(state).length >= 2) {
    damageLeader(state, "player", 2);
    healLeader(state, "ai", 2);
  }
}

function continueEndTurnMutable(state: GameState): void {
  if (state.status !== "playing" || state.turnSide !== "player") return;
  aiQuickWindow(state);
  runTasks(state);
  if (state.status !== "playing") return;
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

export function endTurn(input: GameState): GameState {
  const state = clone(input);
  if (state.pending || state.status !== "playing" || state.turnSide !== "player" || state.phase !== "main") return state;
  state.phase = "end";
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
    whiteArtifact: 21,
    blackArtifact: 21,
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

function aiPlayScore(state: GameState, card: CardInstance, zone: Zone): number {
  const def = definition(card);
  if (playableReason(state, card, zone, "ai")) return -999;
  if (def.evolveId && !aiCanCommitEvolveFollower(state, card)) return -999;
  let score = aiCardValue(card.cardId) - def.cost * 1.2;
  const target = bestFollower(state, "player");
  const idols = idolField(state).length;
  if (card.cardId === "destructionJoy") score += idols ? 20 : -30;
  if (card.cardId === "annihilationSong") score += !eggField(state).length && canFitField(state, "ai") ? 30 : 18;
  if (card.cardId === "manifestedLishenna") score += !eggField(state).length && canFitField(state, "ai") ? 22 : 16;
  if (card.cardId === "destructionHermit") score += idols >= 3 && target ? 18 : 0;
  if (card.cardId === "destructionFanatic") score += idols >= 3 && target ? 30 : -10;
  if (card.cardId === "whiteBlackChapter") score += target && remainingHealthOf(target) <= 2 ? 16 : -6;
  if (card.cardId === "solo") score += target ? Math.min(20, idolField(state).filter((item) => !item.tapped).length * 4) : -50;
  if (card.cardId === "returningDissonance") score += idols >= 2 ? 15 : -50;
  if (card.cardId === "zelgenea") score += state.ai.hp <= 10 ? 22 : 0;
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
  if (card.cardId === "destructiveLishenna") return 35;
  if (card.cardId === "axia" && aiSacrificeCandidate(state, card.uid)) return 32;
  if (card.cardId === "destructionServant" && target) return card.flags.freeEvolve ? 40 : 25;
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

function aiUsePrayer(state: GameState): boolean {
  const source = state.ai.field.find((item) => item.cardId === "destructionPrayer" && !item.tapped);
  const sacrifice = aiSacrificeCandidate(state, source?.uid);
  if (!source || !sacrifice) return false;
  const target = bestFollower(state, "player", "kill");
  const shouldUse = Boolean(target && remainingHealthOf(target) <= 2) || state.player.hp <= 6 || state.ai.field.length >= 5;
  if (!shouldUse) return false;
  source.tapped = true;
  moveFieldToGrave(state, sacrifice, "破壊の祈祷者起動費用");
  if (target && remainingHealthOf(target) <= 2) dealDamageToFollower(state, target, 2, "破壊の祈祷者");
  else damageLeader(state, "player", 2);
  addLog(state, "破壞巫使用破壊の祈祷者的起動能力。 ");
  runTasks(state);
  return true;
}

function aiCycleChapter(state: GameState): boolean {
  const chapter = [...state.ai.field]
    .filter((item) => (item.cardId === "newWhite" || item.cardId === "newBlack") && !item.tapped)
    .sort((a, b) => Number(b.cardId === "newBlack") - Number(a.cardId === "newBlack"))[0];
  const standIdols = idolField(state).filter((item) => !item.tapped);
  if (!chapter || standIdols.length < 3) return false;
  const costs = [chapter, ...standIdols.filter((item) => item.uid !== chapter.uid)].slice(0, 3);
  for (const card of costs) card.tapped = true;
  moveFieldToGrave(state, chapter, `${cardName(chapter.cardId)}起動費用`);
  addLog(state, `破壞巫橫置3張偶像卡牌，破壞${chapter.cardId === "newBlack" ? "黑蛋" : "白蛋"}以觸發謝幕曲。`);
  runTasks(state);
  return true;
}

function aiUseWilderness(state: GameState): boolean {
  const wild = state.ai.field.find((item) => item.cardId === "destructionWilderness");
  const sacrifice = aiSacrificeCandidate(state, wild?.uid);
  if (!wild || !sacrifice || state.ai.field.length < 5) return false;
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
  return true;
}

function resolveAiAttack(state: GameState, attacker: CardInstance, targetUid: string): void {
  if (!canAttackNow(state, attacker)) return;
  const legal = attackTargets(state, attacker).map((item) => item.uid);
  if (!legal.includes(targetUid)) return;
  attacker.tapped = true;
  if (targetUid === "player-leader") {
    addLog(state, `${cardName(attacker.cardId)}攻擊你的主戰者。`);
    damageLeader(state, "player", attackOf(attacker));
    return;
  }
  const target = state.player.field.find((item) => item.uid === targetUid);
  if (!target) return;
  addLog(state, `${cardName(attacker.cardId)}攻擊${cardName(target.cardId)}。`);
  const attackerDamage = attackOf(target);
  const targetDamage = attackOf(attacker);
  dealDamageToFollower(state, target, targetDamage, "交戰");
  const still = state.ai.field.find((item) => item.uid === attacker.uid);
  if (still) dealDamageToFollower(state, still, attackerDamage, "交戰");
  runTasks(state);
}

function aiAttackPhase(state: GameState): void {
  const attackers = state.ai.field.filter((item) => isFollower(item) && canAttackNow(state, item));
  attackers.sort((a, b) => attackOf(b) - attackOf(a));
  for (const attackerSnapshot of attackers) {
    const attacker = state.ai.field.find((item) => item.uid === attackerSnapshot.uid);
    if (!attacker || state.status !== "playing") continue;
    const targets = attackTargets(state, attacker);
    if (!targets.length) continue;
    const followerTargets = targets
      .map((option) => state.player.field.find((item) => item.uid === option.uid))
      .filter(Boolean) as CardInstance[];
    const favorable = followerTargets
      .filter((target) => attackOf(attacker) >= remainingHealthOf(target))
      .sort((a, b) => attackOf(b) - attackOf(a))[0];
    const face = targets.find((option) => option.uid === "player-leader");
    resolveAiAttack(state, attacker, favorable?.uid ?? face?.uid ?? targets[0].uid);
  }
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

function aiEndPhase(state: GameState): void {
  state.phase = "end";
  const world = state.ai.ex.find((item) => item.cardId === "greatZelgenea");
  if (world) {
    damageLeader(state, "player", 4);
    const taskCountBefore = state.tasks.length;
    for (const follower of [...state.player.field.filter(isFollower)]) dealDamageToFollower(state, follower, 4, "大いなる《世界》結束階段");
    groupNewPlayerTasks(state, taskCountBefore, "大いなる《世界》造成的同時離場效果");
    runTasks(state);
  }
  aiDiscardToSeven(state);
  if (state.status === "playing") finishTurnSwitchMutable(state);
}

function runAiTurnMutable(state: GameState): void {
  if (state.status !== "playing" || state.turnSide !== "ai" || state.pending) return;
  state.phase = "ai";
  runTasks(state);
  let guard = 0;
  while (state.status === "playing" && state.turnSide === "ai" && !state.pending && guard < 40) {
    guard += 1;
    if (aiGraveWorld(state)) continue;
    if (aiCycleChapter(state)) continue;
    if (tryAiEvolve(state)) {
      runTasks(state);
      continue;
    }
    if (aiUsePrayer(state)) continue;
    if (aiUseWilderness(state)) continue;
    const best = aiBestPlayable(state);
    if (best && playCardMutable(state, "ai", best.card.uid, best.zone)) {
      runTasks(state);
      continue;
    }
    break;
  }
  aiAttackPhase(state);
  if (state.status === "playing") aiEndPhase(state);
}

function finishTurnSwitchMutable(state: GameState): void {
  if (state.status !== "playing") return;
  const ending = state.turnSide;
  const next = otherSide(ending);
  addLog(state, `${ending === "player" ? "你的" : "破壞巫的"}回合結束。`);
  beginTurnMutable(state, next);
  runTasks(state);
  if (next === "ai" && !state.pending) runAiTurnMutable(state);
}

export function restartWithSameSeed(state: GameState, playerFirst = state.playerFirst): GameState {
  return createGame(playerFirst, state.seed);
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
  aiCanCommitEvolveFollower,
  tryAiEvolve,
  beginTurnMutable,
};
