import assert from "node:assert/strict";
import test from "node:test";
import {
  __testing,
  activateFieldCard,
  attackTargets,
  cardActions,
  createGame,
  endTurn,
  finishMulligan,
  hasKeyword,
  playCard,
  resolveChoice,
  type CardInstance,
  type GameState,
  type Zone,
} from "../src/game/engine.ts";
import { isLevinCard } from "../src/game/cards.ts";

function playing(seed = 7): GameState {
  return finishMulligan(createGame(true, seed), false);
}

function addField(state: GameState, side: "player" | "ai", cardId: string): CardInstance {
  const card = __testing.makeInstance(state, cardId, side, "field");
  __testing.putExistingIntoField(state, card, side, false);
  return card;
}

test("fixed lists contain 40 main cards and 10 evolve cards", () => {
  const state = createGame(true, 123);
  assert.equal(state.player.deck.length + state.player.hand.length, 40);
  assert.equal(state.ai.deck.length + state.ai.hand.length, 40);
  assert.equal(Object.values(state.player.evolveRemaining).reduce((a, b) => a + b, 0), 10);
  assert.equal(Object.values(state.ai.evolveRemaining).reduce((a, b) => a + b, 0), 10);
});

test("a follower put directly from outside the field still resolves Fanfare", () => {
  const state = playing();
  state.player.field = [];
  state.player.ex = [];
  const water = __testing.makeInstance(state, "pureWaterFairy", "player", "deck");
  __testing.putExistingIntoField(state, water, "player", true);
  __testing.runTasks(state);
  assert.equal(state.player.ex.length, 1);
  assert.equal(state.player.ex[0].cardId, "fairy");
});

test("a ward follower put into play asks whether it should enter ACT before resolving Fanfare", () => {
  const state = playing();
  state.player.field = [];
  const dragon = __testing.makeInstance(state, "fairyDragon", "player", "deck");
  __testing.putExistingIntoField(state, dragon, "player", true);
  __testing.runTasks(state);
  assert.equal(state.pending?.effect, "wardOnEntry");
  const resolved = resolveChoice(state, ["yes"]);
  assert.equal(resolved.player.field[0].tapped, true);
  assert.equal(resolved.pending, undefined);
});

test("tokens trigger leaving-field handling but never remain in the graveyard", () => {
  const state = playing();
  state.player.field = [];
  state.player.grave = [];
  const fairy = addField(state, "player", "fairy");
  __testing.moveFieldToGrave(state, fairy, "測試");
  __testing.runTasks(state);
  assert.equal(state.player.field.length, 0);
  assert.equal(state.player.grave.some((card) => card.cardId === "fairy"), false);
});

test("cards lose field-only damage and buffs when they move to the graveyard", () => {
  const state = playing();
  state.player.field = [];
  state.player.grave = [];
  const follower = addField(state, "player", "pureWaterFairy");
  follower.damage = 1;
  follower.attackBuff = 3;
  follower.healthBuff = 4;
  __testing.moveFieldToGrave(state, follower, "測試");
  assert.equal(state.player.grave[0].damage, 0);
  assert.equal(state.player.grave[0].attackBuff, 0);
  assert.equal(state.player.grave[0].healthBuff, 0);
});

test("Wing Queen's deck summon resolves the summoned follower's Fanfare", () => {
  const state = playing();
  state.player.field = [];
  state.player.ex = [];
  __testing.addExDirect(state, "player", ["fairy", "fairy"]);
  const queen = __testing.makeInstance(state, "wingQueen", "player", "deck");
  __testing.putExistingIntoField(state, queen, "player", true);
  __testing.runTasks(state);
  assert.equal(state.pending?.effect, "wingQueenTutor");
  const water = state.pending?.options.find((option) => option.cardId === "pureWaterFairy");
  assert.ok(water);
  const resolved = resolveChoice(state, [water.uid]);
  assert.equal(resolved.player.field.some((card) => card.cardId === "pureWaterFairy"), true);
  assert.equal(resolved.player.ex.length, 3);
});

test("Fairy Archer lets the player choose a valid top card and explicitly order the rest on the bottom", () => {
  const state = playing();
  state.player.field = [];
  const vista = __testing.makeInstance(state, "vistaElf", "player", "deck");
  const antiAir = __testing.makeInstance(state, "antiAir", "player", "deck");
  const water = __testing.makeInstance(state, "pureWaterFairy", "player", "deck");
  state.player.deck = [vista, antiAir, water];
  const archer = __testing.makeInstance(state, "fairyArcher", "player", "deck");
  __testing.putExistingIntoField(state, archer, "player", true);
  __testing.runTasks(state);
  assert.equal(state.pending?.effect, "fairyArcherPick");
  const afterPick = resolveChoice(state, [water.uid]);
  assert.equal(afterPick.player.hand.some((card) => card.uid === water.uid), true);
  assert.equal(afterPick.pending?.effect, "bottomOrder");
  const ordered = resolveChoice(afterPick, [vista.uid, antiAir.uid]);
  assert.deepEqual(ordered.player.deck.map((card) => card.uid), [vista.uid, antiAir.uid]);
});

test("simultaneous Last Words caused by the Great World ask the player for trigger order", () => {
  const state = playing();
  state.player.field = [];
  addField(state, "player", "pureWaterFairy");
  addField(state, "player", "fairyDragon");
  __testing.resolveTask(state, { type: "fanfare", side: "ai", cardId: "greatZelgenea" });
  __testing.runTasks(state);
  assert.equal(state.pending?.effect, "triggerOrder");
  assert.equal(state.pending?.options.length, 2);
});

test("the destruction AI chooses to produce a white egg whenever Annihilation Song has field space", () => {
  const state = playing();
  state.ai.field = [];
  __testing.resolveTask(state, { type: "spell", side: "ai", cardId: "annihilationSong" });
  __testing.runTasks(state);
  assert.equal(state.ai.field.some((card) => card.cardId === "newWhite"), true);
});

test("the destruction AI never produces a second egg while a white or black egg is already in play", () => {
  const state = playing();
  state.ai.field = [];
  addField(state, "ai", "newWhite");
  __testing.resolveTask(state, { type: "spell", side: "ai", cardId: "annihilationSong" });
  __testing.runTasks(state);
  const eggs = state.ai.field.filter((card) => card.cardId === "newWhite" || card.cardId === "newBlack");
  assert.equal(eggs.length, 1);
  assert.equal(eggs[0].cardId, "newBlack");
});

test("Manifested Lishenna prioritizes a white egg over Solo even when a target exists", () => {
  const state = playing();
  state.ai.field = [];
  state.ai.ex = [];
  state.player.field = [];
  addField(state, "ai", "destructionHermit");
  const lishenna = addField(state, "ai", "manifestedLishenna");
  addField(state, "player", "fairyDragon");
  __testing.resolveTask(state, { type: "fanfare", side: "ai", sourceUid: lishenna.uid, cardId: "manifestedLishenna" });
  __testing.runTasks(state);
  assert.equal(state.ai.field.some((card) => card.cardId === "newWhite"), true);
  assert.equal(state.ai.ex.some((card) => card.cardId === "solo"), false);
});

test("the destruction AI immediately taps three Idol cards to break an available egg", () => {
  const state = playing();
  state.ai.field = [];
  const white = addField(state, "ai", "newWhite");
  const hermit = addField(state, "ai", "destructionHermit");
  const wilderness = addField(state, "ai", "destructionWilderness");
  const used = __testing.aiCycleChapter(state);
  assert.equal(used, true);
  assert.equal(state.ai.field.some((card) => card.uid === white.uid), false);
  assert.equal(state.ai.field.some((card) => card.cardId === "newBlack"), true);
  assert.equal(hermit.tapped, true);
  assert.equal(wilderness.tapped, true);
  assert.equal(state.ai.hp, 21);
});

test("egg cycling logs every field card used as its three-card ACT cost", () => {
  const state = playing();
  state.turnSide = "ai";
  state.ai.field = [];
  const white = addField(state, "ai", "newWhite");
  const attacker = addField(state, "ai", "originalLishenna");
  const wilderness = addField(state, "ai", "destructionWilderness");
  const freshHermit = addField(state, "ai", "destructionHermit");
  attacker.enteredAt = state.globalTurn - 1;
  wilderness.enteredAt = state.globalTurn;
  freshHermit.enteredAt = state.globalTurn;
  __testing.addExDirect(state, "ai", ["whiteArtifact", "blackArtifact"]);

  assert.equal(__testing.aiCycleChapter(state), true);
  const paymentLog = state.log.find((entry) => entry.includes("3張偶像卡牌"));
  assert.ok(paymentLog?.includes("新約・白の章"));
  assert.ok(paymentLog?.includes("破壊の荒野"));
  assert.ok(paymentLog?.includes("破壊の隠者"));
  assert.equal(paymentLog?.includes("アーティファクト"), false);
  assert.equal(attacker.tapped, false);
  assert.equal(state.ai.field.some((card) => card.uid === white.uid), false);
});

test("egg cycling preserves attackers when their face damage is worth more than the chapter trigger", () => {
  const state = playing();
  state.turnSide = "ai";
  state.ai.field = [];
  addField(state, "ai", "newBlack");
  const lishenna = addField(state, "ai", "originalLishenna");
  const fanatic = addField(state, "ai", "destructionFanatic");
  lishenna.enteredAt = state.globalTurn - 1;
  fanatic.enteredAt = state.globalTurn - 1;

  assert.equal(__testing.aiCycleChapter(state), false);
  assert.equal(lishenna.tapped, false);
  assert.equal(fanatic.tapped, false);
  assert.equal(__testing.aiCycleChapter(state, false), true);
});

test("the destruction AI does not waste Destruction's Joy at full PP without its draw condition", () => {
  const state = playing();
  state.turnSide = "ai";
  state.ai.field = [];
  addField(state, "ai", "destructionHermit");
  state.ai.maxPP = 3;
  state.ai.pp = 3;
  const joy = __testing.makeInstance(state, "destructionJoy", "ai", "hand");
  assert.equal(__testing.aiPlayScore(state, joy, "hand"), -999);
  state.ai.pp = 2;
  assert.ok(__testing.aiPlayScore(state, joy, "hand") > 0);
});

test("the destruction AI keeps White Chapter Black Chapter as Quick unless its leader damage is lethal", () => {
  const state = playing();
  state.turnSide = "ai";
  state.ai.field = [];
  state.player.field = [];
  addField(state, "ai", "destructionHermit");
  addField(state, "ai", "destructionWilderness");
  addField(state, "player", "pureWaterFairy");
  state.ai.maxPP = 4;
  state.ai.pp = 4;
  const quick = __testing.makeInstance(state, "whiteBlackChapter", "ai", "hand");
  assert.equal(__testing.aiPlayScore(state, quick, "hand"), -999);
  state.player.hp = 2;
  assert.ok(__testing.aiPlayScore(state, quick, "hand") > 0);
});

test("the destruction AI deploys the black artifact before the white artifact while healthy", () => {
  const state = playing();
  state.turnSide = "ai";
  state.ai.field = [];
  state.ai.ex = [];
  state.ai.hand = [];
  state.ai.maxPP = 2;
  state.ai.pp = 2;
  __testing.addExDirect(state, "ai", ["whiteArtifact", "blackArtifact"]);
  const best = __testing.aiBestPlayable(state);
  assert.equal(best?.card.cardId, "blackArtifact");
});

test("the destruction AI develops only cards that leave enough PP for a known Quick target", () => {
  const state = playing();
  state.turnSide = "ai";
  state.ai.field = [];
  state.player.field = [];
  addField(state, "ai", "destructionHermit");
  addField(state, "ai", "destructionWilderness");
  addField(state, "player", "pureWaterFairy");
  state.ai.maxPP = 4;
  state.ai.pp = 4;
  const quick = __testing.makeInstance(state, "whiteBlackChapter", "ai", "hand");
  const worshipper = __testing.makeInstance(state, "dissonanceWorshipper", "ai", "hand");
  state.ai.hand = [quick, worshipper];
  assert.equal(__testing.aiBestPlayable(state)?.card.cardId, "dissonanceWorshipper");
  state.ai.pp = 3;
  assert.equal(__testing.aiBestPlayable(state), undefined);
});

test("the destruction AI evolves Axia before considering an egg cycle", () => {
  const state = playing();
  state.turnSide = "ai";
  state.phase = "ai";
  state.evolvedThisTurn = false;
  state.ai.field = [];
  state.ai.hand = [];
  state.ai.ex = [];
  state.ai.pp = 0;
  state.ai.maxPP = 3;
  state.ai.ep = 1;
  state.ai.ownTurn = 3;
  addField(state, "ai", "axia");
  addField(state, "ai", "newWhite");
  addField(state, "ai", "destructionHermit");

  __testing.runAiTurnMutable(state);
  const evolveIndex = state.log.findIndex((entry) => entry.includes("破壊の継承者・アクシア進化"));
  const cycleIndex = state.log.findIndex((entry) => entry.includes("3張偶像卡牌"));
  assert.ok(evolveIndex >= 0);
  if (cycleIndex >= 0) assert.ok(cycleIndex < evolveIndex, "the later cycle should appear above evolve in the reverse-chronological log");
});

test("Axia is held until it can evolve immediately and has another Idol card to sacrifice", () => {
  const state = playing();
  state.turnSide = "ai";
  state.ai.field = [];
  state.ai.pp = 2;
  state.ai.maxPP = 2;
  const axia = __testing.makeInstance(state, "axia", "ai", "hand");
  assert.equal(__testing.aiCanCommitEvolveFollower(state, axia), false);
  addField(state, "ai", "destructionHermit");
  assert.equal(__testing.aiCanCommitEvolveFollower(state, axia), true);
  state.evolvedThisTurn = true;
  assert.equal(__testing.aiCanCommitEvolveFollower(state, axia), false);
});

test("Axia uses super evolution as a power turn when SEP timing is available", () => {
  const state = playing();
  state.turnSide = "ai";
  state.ai.field = [];
  state.ai.pp = 1;
  state.ai.maxPP = 6;
  state.ai.ownTurn = 6;
  state.ai.sep = 1;
  state.evolvedThisTurn = false;
  const axia = addField(state, "ai", "axia");
  addField(state, "ai", "destructionHermit");
  const evolved = __testing.tryAiEvolve(state);
  assert.equal(evolved, true);
  assert.equal(axia.cardId, "axiaEvo");
  assert.equal(state.ai.sep, 0);
  assert.equal(state.player.hp, 19);
});

test("when EX has one slot, a two-token effect asks which new token enters", () => {
  const state = playing();
  state.player.ex = [];
  __testing.addExDirect(state, "player", ["fairy", "fairy", "fairy", "fairy"]);
  __testing.addExTask(state, "player", ["fairyWisp", "fairy"]);
  assert.equal(state.pending?.effect, "addExSubset");
  assert.equal(state.pending?.min, 1);
  const wispOption = state.pending?.options.find((option) => option.cardId === "fairyWisp");
  assert.ok(wispOption);
  const resolved = resolveChoice(state, [wispOption.uid]);
  assert.equal(resolved.player.ex.length, 5);
  assert.equal(resolved.player.ex.at(-1)?.cardId, "fairyWisp");
});

test("an ACT ward is the only legal attack target", () => {
  const state = playing();
  state.player.field = [];
  state.ai.field = [];
  state.turnSide = "player";
  const attacker = addField(state, "player", "tailwindFairy");
  attacker.enteredAt = state.globalTurn - 1;
  const ward = addField(state, "ai", "reverseAmatsu");
  ward.tapped = true;
  const other = addField(state, "ai", "destructionHermit");
  other.tapped = true;
  assert.deepEqual(attackTargets(state, attacker).map((target) => target.uid), [ward.uid]);
});

test("end phase offers every standing ward as an optional ACT choice", () => {
  const state = playing();
  state.player.field = [];
  addField(state, "player", "reverseAmatsu");
  const ended = endTurn(state);
  assert.equal(ended.pending?.effect, "guardChoice");
  assert.equal(ended.pending?.min, 0);
  assert.equal(ended.pending?.options.length, 1);
});

test("the destruction AI goes all-in on face when board attacks alone are lethal", () => {
  const state = playing();
  state.turnSide = "ai";
  state.player.field = [];
  state.ai.field = [];
  state.player.hp = 5;
  const fanatic = addField(state, "ai", "destructionFanatic");
  const hermit = addField(state, "ai", "destructionHermit");
  fanatic.enteredAt = state.globalTurn - 1;
  hermit.enteredAt = state.globalTurn - 1;
  // 場上放一隻可交換的目標（非守護）：舊AI會去換血，新AI必須直接打臉斬殺。
  const bait = addField(state, "player", "breathFairyDancer");
  bait.tapped = true;
  __testing.aiAttackPhase(state);
  assert.equal(state.status, "gameover");
  assert.equal(state.winner, "ai");
});

test("the reserved Quick chapter is spent proactively when it completes lethal", () => {
  const state = playing();
  state.turnSide = "ai";
  state.phase = "ai";
  state.player.field = [];
  state.ai.field = [];
  state.ai.hand = [];
  state.player.hp = 4;
  state.ai.pp = 2;
  state.ai.maxPP = 4;
  const hermit = addField(state, "ai", "destructionHermit");
  const wilderness = addField(state, "ai", "destructionWilderness");
  hermit.enteredAt = state.globalTurn - 1;
  wilderness.enteredAt = state.globalTurn - 1;
  addField(state, "player", "pureWaterFairy");
  const quick = __testing.makeInstance(state, "whiteBlackChapter", "ai", "hand");
  state.ai.hand = [quick];
  // 章2點直傷＋隠者2點攻擊＝4，正好斬殺。
  assert.equal(__testing.aiLethalInSight(state), true);
  __testing.runAiTurnMutable(state);
  assert.equal(state.status, "gameover");
  assert.equal(state.winner, "ai");
});

test("the Quick window is not wasted on a cheap token attacker", () => {
  const state = playing();
  state.turnSide = "player";
  state.player.field = [];
  state.ai.field = [];
  state.ai.pp = 4;
  state.ai.maxPP = 4;
  addField(state, "ai", "destructionHermit");
  addField(state, "ai", "destructionWilderness");
  const quick = __testing.makeInstance(state, "whiteBlackChapter", "ai", "hand");
  state.ai.hand = [quick];
  const token = addField(state, "player", "fairy");
  __testing.aiQuickWindow(state, token);
  assert.equal(state.ai.hand.length, 1, "chapter should stay in hand against a 1/1 token");
  const amatsu = addField(state, "player", "fairyBladeAmatsu");
  __testing.aiQuickWindow(state, amatsu);
  assert.equal(state.ai.hand.length, 0, "chapter should still answer a real threat");
});

test("the prayer channels burn at the leader while racing", () => {
  const state = playing();
  state.turnSide = "ai";
  state.player.field = [];
  state.ai.field = [];
  state.player.hp = 8;
  addField(state, "ai", "destructionPrayer");
  addField(state, "ai", "destructionHermit");
  const kill = addField(state, "player", "pureWaterFairy");
  kill.tapped = true;
  assert.equal(__testing.aiUsePrayer(state), true);
  assert.equal(state.player.hp, 6, "at 8hp the prayer must hit face, not the 2/1 follower");
});

test("Original Lishenna is a low-pressure play only, and a second copy is deprioritized", () => {
  const state = playing();
  state.turnSide = "ai";
  state.ai.field = [];
  state.player.field = [];
  state.ai.pp = 4;
  state.ai.maxPP = 4;
  const lishenna = __testing.makeInstance(state, "originalLishenna", "ai", "hand");
  const calm = __testing.aiPlayScore(state, lishenna, "hand");
  // 壓力大（對面場面攻擊力逼近血量）時大幅降分
  state.ai.hp = 8;
  const cynthia = addField(state, "player", "queenCynthia");
  const dancer = addField(state, "player", "breathFairyDancer");
  const pressured = __testing.aiPlayScore(state, lishenna, "hand");
  assert.ok(pressured < calm - 20, `pressured score ${pressured} should be far below calm score ${calm}`);
  // 神器已上線時的第二張也降分
  state.player.field = state.player.field.filter((card) => card.uid !== cynthia.uid && card.uid !== dancer.uid);
  state.ai.hp = 20;
  __testing.addExDirect(state, "ai", ["whiteArtifact"]);
  const secondCopy = __testing.aiPlayScore(state, lishenna, "hand");
  assert.ok(secondCopy < calm, `second-copy score ${secondCopy} should be below first-copy score ${calm}`);
});

test("Destruction Servant is prioritized when three Idols enable its free evolve clear", () => {
  const state = playing();
  state.turnSide = "ai";
  state.ai.field = [];
  state.ai.pp = 2;
  state.ai.maxPP = 2;
  addField(state, "player", "queenCynthia");
  const servant = __testing.makeInstance(state, "destructionServant", "ai", "hand");
  const without = __testing.aiPlayScore(state, servant, "hand");
  addField(state, "ai", "newWhite");
  addField(state, "ai", "destructionHermit");
  const withCombo = __testing.aiPlayScore(state, servant, "hand");
  assert.ok(withCombo > without + 25, `combo score ${withCombo} should far exceed off-combo score ${without}`);
});

test("under lethal pressure the AI trades into the biggest attacker instead of going face", () => {
  const state = playing();
  state.turnSide = "ai";
  state.player.field = [];
  state.ai.field = [];
  state.ai.hp = 6;
  state.player.hp = 20;
  const cynthia = addField(state, "player", "queenCynthia");
  cynthia.tapped = true;
  const fanatic = addField(state, "ai", "destructionFanatic");
  const lishenna = addField(state, "ai", "destructiveLishenna");
  fanatic.enteredAt = state.globalTurn - 1;
  lishenna.enteredAt = state.globalTurn - 1;
  __testing.aiAttackPhase(state);
  assert.equal(state.player.field.some((card) => card.uid === cynthia.uid), false, "Cynthia must be focused down, not ignored for face damage");
  assert.equal(state.player.hp, 20);
});

function levinPlaying(seed = 7): GameState {
  return finishMulligan(createGame(true, seed, "levin"), false);
}

function fillGrave(state: GameState, cardId: string, count: number): void {
  for (let index = 0; index < count; index += 1) {
    state.player.grave.push(__testing.makeInstance(state, cardId, "player", "grave"));
  }
}

test("the Levin graveyard engine turns on keywords at five Levin cards", () => {
  const state = levinPlaying();
  state.player.field = [];
  state.player.grave = [];
  const geno = addField(state, "player", "levinAxeGeno");
  const meim = addField(state, "player", "levinMeim");
  fillGrave(state, "levinMiim", 4);
  assert.equal(hasKeyword(state, geno, "storm"), false);
  assert.equal(hasKeyword(state, meim, "designated"), false);
  fillGrave(state, "levinSisters", 1);
  assert.equal(hasKeyword(state, geno, "storm"), true);
  assert.equal(hasKeyword(state, meim, "designated"), true);
});

test("Brutal Geno is correctly tagged as a Levin card", () => {
  assert.equal(isLevinCard("brutalGeno"), true);
});

test("Levin Sisters with the +4 kicker summons all three sisters from the deck", () => {
  let state = levinPlaying();
  state.player.field = [];
  state.ai.field = [];
  state.player.pp = 5;
  state.player.maxPP = 5;
  const sisters = __testing.makeInstance(state, "levinSisters", "player", "hand");
  state.player.hand = [sisters];
  state = playCard(state, sisters.uid, "hand");
  assert.equal(state.pending?.effect, "sistersKicker");
  state = resolveChoice(state, ["yes"]);
  assert.equal(state.pending?.effect, "sistersDeploy");
  state = resolveChoice(state, state.pending!.options.map((option) => option.uid));
  assert.equal(state.pending?.effect, "triggerOrder", "all three Fanfare abilities should trigger simultaneously after every sister enters");
  for (const id of ["levinMaim", "levinMiim", "levinMeim"]) {
    assert.equal(state.player.field.some((card) => card.cardId === id), true, `${id} should be on the field`);
  }
  state = autoResolve(state);
  assert.equal(state.player.pp, 0);
});

test("Brutal Geno can be played for one PP by sacrificing a cheap Levin follower", () => {
  let state = levinPlaying();
  state.player.field = [];
  state.ai.field = [];
  state.player.pp = 1;
  state.player.maxPP = 4;
  const runes = addField(state, "player", "levinRunes");
  const geno = __testing.makeInstance(state, "brutalGeno", "player", "hand");
  state.player.hand = [geno];
  state = playCard(state, geno.uid, "hand");
  assert.equal(state.pending?.effect, "brutalGenoPlay");
  state = autoResolve(resolveChoice(state, [runes.uid]));
  assert.equal(state.player.field.some((card) => card.cardId === "brutalGeno"), true);
  assert.equal(state.player.grave.some((card) => card.uid === runes.uid), true);
  assert.equal(state.player.pp, 0);
});

test("Albert restands for three PP once per turn with ten Royal followers in the graveyard", () => {
  let state = levinPlaying();
  state.player.field = [];
  state.player.grave = [];
  state.player.pp = 6;
  const albert = addField(state, "player", "levinAlbert");
  albert.tapped = true;
  fillGrave(state, "gawain", 10);
  state = activateFieldCard(state, albert.uid, "albertRestand");
  const restood = state.player.field.find((card) => card.cardId === "levinAlbert");
  assert.equal(restood?.tapped, false);
  assert.equal(state.player.pp, 3);
  restood!.tapped = true;
  const again = activateFieldCard(state, restood!.uid, "albertRestand");
  assert.equal(again.player.field.find((card) => card.cardId === "levinAlbert")?.tapped, true, "second restand in one turn must be rejected");
});

test("discarding Levin Axe Geno shows the top card and lets the player take or leave it", () => {
  const state = levinPlaying();
  state.player.field = [];
  state.player.grave = [];
  const miim = addField(state, "player", "levinMiim");
  const geno = __testing.makeInstance(state, "levinAxeGeno", "player", "hand");
  const topLevin = __testing.makeInstance(state, "levinMeim", "player", "deck");
  const miimDraw = __testing.makeInstance(state, "gawain", "player", "deck");
  state.player.hand = [geno];
  // ミイム先完成「抽1張」，之後才處理因捨棄而觸發的ジェノ能力。
  state.player.deck.push(topLevin, miimDraw);
  __testing.resolveTask(state, { type: "fanfare", side: "player", sourceUid: miim.uid, cardId: "levinMiim" });
  assert.equal(state.pending?.effect, "miimDiscard");
  const afterDiscard = resolveChoice(state, [geno.uid]);
  assert.equal(afterDiscard.pending?.effect, "genoDigPick");
  assert.equal(afterDiscard.player.grave.some((card) => card.cardId === "levinAxeGeno"), true);
  assert.equal(afterDiscard.player.deck.at(-1)?.uid, topLevin.uid, "the top card must remain in the deck until the player decides");

  const left = resolveChoice(afterDiscard, []);
  assert.equal(left.player.deck.at(-1)?.uid, topLevin.uid);
  assert.equal(left.player.hand.some((card) => card.uid === topLevin.uid), false);

  const taken = resolveChoice(afterDiscard, [topLevin.uid]);
  assert.equal(taken.player.hand.some((card) => card.uid === topLevin.uid), true);
});

test("Transcendent Julius is playable without two Levin cards and its reveal cost is optional", () => {
  let state = levinPlaying();
  state.player.field = [];
  state.ai.field = [];
  state.player.pp = 3;
  const lone = __testing.makeInstance(state, "levinTranscend", "player", "hand");
  state.player.hand = [lone];
  assert.equal(cardActions(state, lone.uid, "hand").find((action) => action.id === "play")?.enabled, true);
  const playedWithoutCost = playCard(state, lone.uid, "hand");
  assert.equal(playedWithoutCost.player.field.some((card) => card.uid === lone.uid), true);
  assert.equal(playedWithoutCost.pending, undefined);

  state = levinPlaying();
  state.player.field = [];
  state.ai.field = [];
  state.player.pp = 3;
  const transcend = __testing.makeInstance(state, "levinTranscend", "player", "hand");
  const miim = __testing.makeInstance(state, "levinMiim", "player", "hand");
  const meim = __testing.makeInstance(state, "levinMeim", "player", "hand");
  state.player.hand = [transcend, miim, meim];
  state = playCard(state, transcend.uid, "hand");
  assert.equal(state.pending?.effect, "transcendRevealChoice");
  const declined = resolveChoice(state, ["no"]);
  assert.equal(declined.pending, undefined);
  assert.equal(declined.ai.hp, 20);

  let accepted = resolveChoice(state, ["yes"]);
  assert.equal(accepted.pending?.effect, "transcendRevealCards");
  accepted = resolveChoice(accepted, [miim.uid, meim.uid]);
  assert.equal(accepted.pending?.effect, "transcendModes");
  accepted = resolveChoice(accepted, ["burn"]);
  assert.equal(accepted.ai.hp, 17);
});

test("Runes asks which Albert to take and shuffles even when the search is declined", () => {
  let state = levinPlaying();
  state.player.field = [];
  state.player.grave = [];
  const runes = addField(state, "player", "levinRunes");
  const discard = __testing.makeInstance(state, "levinMiim", "player", "hand");
  const albert = __testing.makeInstance(state, "levinAlbert", "player", "deck");
  const fillerA = __testing.makeInstance(state, "gawain", "player", "deck");
  const fillerB = __testing.makeInstance(state, "levinMaim", "player", "deck");
  state.player.hand = [discard];
  state.player.deck = [albert, fillerA, fillerB];
  __testing.resolveTask(state, { type: "fanfare", side: "player", sourceUid: runes.uid, cardId: "levinRunes" });
  assert.equal(state.pending?.effect, "runesDiscard");
  state = resolveChoice(state, [discard.uid]);
  assert.equal(state.pending?.effect, "runesAlbertPick");
  const rngBeforeSearch = state.rng;
  state = resolveChoice(state, []);
  assert.equal(state.player.deck.some((card) => card.uid === albert.uid), true);
  assert.notEqual(state.rng, rngBeforeSearch);
});

test("Levin Justice requires a follower target and its Duke search can intentionally fail", () => {
  let state = levinPlaying();
  state.player.field = [];
  state.ai.field = [];
  state.player.pp = 5;
  const noTargetJustice = __testing.makeInstance(state, "levinJustice", "player", "hand");
  state.player.hand = [noTargetJustice];
  const playAction = cardActions(state, noTargetJustice.uid, "hand").find((action) => action.id === "play");
  assert.equal(playAction?.enabled, false);
  const rejected = playCard(state, noTargetJustice.uid, "hand");
  assert.equal(rejected.player.hand.some((card) => card.uid === noTargetJustice.uid), true);
  assert.equal(rejected.player.pp, 5);

  state = levinPlaying();
  state.player.field = [];
  state.ai.field = [];
  state.player.pp = 3;
  const justice = __testing.makeInstance(state, "levinJustice", "player", "hand");
  const duke = __testing.makeInstance(state, "levinDuke", "player", "deck");
  const fillerA = __testing.makeInstance(state, "gawain", "player", "deck");
  const fillerB = __testing.makeInstance(state, "levinMiim", "player", "deck");
  state.player.hand = [justice];
  state.player.deck = [duke, fillerA, fillerB];
  const target = addField(state, "ai", "destructionPrayer");
  state = playCard(state, justice.uid, "hand");
  state = resolveChoice(state, [target.uid]);
  assert.equal(state.pending?.effect, "justiceDukePick");
  const rngBeforeSearch = state.rng;
  const skipped = resolveChoice(state, []);
  assert.equal(skipped.player.field.some((card) => card.cardId === "levinDuke"), false);
  assert.notEqual(skipped.rng, rngBeforeSearch, "the deck must be shuffled even when the legal search is declined");
});

test("Brutal Geno can sacrifice a cheap Levin follower even while all five field slots are occupied", () => {
  let state = levinPlaying();
  state.player.field = [];
  state.ai.field = [];
  state.player.pp = 1;
  const sacrifice = addField(state, "player", "levinRunes");
  for (let index = 0; index < 4; index += 1) addField(state, "player", "levinMiim");
  const geno = __testing.makeInstance(state, "brutalGeno", "player", "hand");
  state.player.hand = [geno];
  assert.equal(cardActions(state, geno.uid, "hand").find((action) => action.id === "play")?.enabled, true);
  state = playCard(state, geno.uid, "hand");
  assert.equal(state.pending?.effect, "brutalGenoPlay");
  state = autoResolve(resolveChoice(state, [sacrifice.uid]));
  assert.equal(state.player.field.length, 5);
  assert.equal(state.player.field.some((card) => card.cardId === "brutalGeno"), true);
});

test("Maim and the Levin Archer still choose a target and pay ACT below five Levin cards", () => {
  let state = levinPlaying();
  state.player.field = [];
  state.player.grave = [];
  state.ai.field = [];
  const target = addField(state, "ai", "destructionPrayer");
  const maim = addField(state, "player", "levinMaim");
  __testing.resolveTask(state, { type: "fanfare", side: "player", sourceUid: maim.uid, cardId: "levinMaim" });
  assert.equal(state.pending?.effect, "maimTarget");
  state = resolveChoice(state, [target.uid]);
  assert.equal(state.ai.field.find((card) => card.uid === target.uid)?.damage, 0);

  const archer = addField(state, "player", "levinArcher");
  const archerAction = cardActions(state, archer.uid, "field").find((action) => action.id === "archerSnipe");
  assert.equal(archerAction?.enabled, true);
  state = activateFieldCard(state, archer.uid, "archerSnipe");
  state = resolveChoice(state, [target.uid]);
  assert.equal(state.player.field.find((card) => card.uid === archer.uid)?.tapped, true);
  assert.equal(state.ai.field.find((card) => card.uid === target.uid)?.damage, 0);
});

test("Duke can use Quick after an AI attack declaration to destroy the attacker", () => {
  let state = levinPlaying();
  state.turnSide = "ai";
  state.phase = "ai";
  state.player.field = [];
  state.ai.field = [];
  const duke = addField(state, "player", "levinDuke");
  const attacker = addField(state, "ai", "destructionHermit");
  attacker.enteredAt = state.globalTurn - 1;
  attacker.damage = 1;
  const hpBefore = state.player.hp;
  __testing.aiAttackPhase(state);
  assert.equal(state.pending?.effect, "dukeQuickAttack");
  state = resolveChoice(state, [duke.uid]);
  assert.equal(state.pending?.effect, "dukeQuickAttackTarget");
  state = resolveChoice(state, [attacker.uid]);
  assert.equal(state.ai.field.some((card) => card.uid === attacker.uid), false);
  assert.equal(state.player.hp, hpBefore, "a destroyed attacker must not deal combat damage");
});

test("Duke receives a Quick window during the AI end phase", () => {
  let state = levinPlaying();
  state.turnSide = "ai";
  state.phase = "ai";
  state.player.field = [];
  state.ai.field = [];
  const duke = addField(state, "player", "levinDuke");
  const target = addField(state, "ai", "destructionHermit");
  target.damage = 1;
  __testing.aiEndPhase(state);
  assert.equal(state.pending?.effect, "dukeQuickEnd");
  state = resolveChoice(state, [duke.uid]);
  assert.equal(state.pending?.effect, "dukeQuickEndTarget");
  state = resolveChoice(state, [target.uid]);
  assert.equal(state.ai.field.some((card) => card.uid === target.uid), false);
  assert.equal(state.turnSide, "player");
});

test("fifty deterministic Levin games advance without an unresolved loop", () => {
  for (let seed = 1; seed <= 50; seed += 1) {
    let state = autoResolve(finishMulligan(createGame(seed % 2 === 0, seed, "levin"), false));
    let actions = 0;
    while (state.status === "playing" && state.globalTurn < 20 && actions < 250) {
      state = autoResolve(state);
      if (state.turnSide !== "player" || state.phase !== "main") {
        actions += 1;
        continue;
      }
      const playable = ([...state.player.hand.map((card) => ({ card, zone: "hand" as Zone })), ...state.player.ex.map((card) => ({ card, zone: "ex" as Zone }))])
        .filter(({ card, zone }) => cardActions(state, card.uid, zone).some((action) => action.id === "play" && action.enabled))
        .sort((a, b) => b.card.cardId.localeCompare(a.card.cardId))[0];
      if (playable) state = autoResolve(playCard(state, playable.card.uid, playable.zone));
      else state = autoResolve(endTurn(state));
      actions += 1;
    }
    assert.ok(actions < 250, `seed ${seed} entered an action loop`);
    assert.equal(state.pending, undefined, `seed ${seed} left a pending choice`);
  }
});

function autoResolve(state: GameState): GameState {
  let current = state;
  let safety = 0;
  while (current.pending && safety < 50) {
    const pending = current.pending;
    const eligible = pending.options.filter((option) => !option.description?.includes("不符合"));
    const count = pending.kind === "order" ? pending.max : pending.min;
    current = resolveChoice(current, eligible.slice(0, count).map((option) => option.uid));
    safety += 1;
  }
  return current;
}

test("fifty deterministic games advance through repeated player and AI turns without an unresolved loop", () => {
  for (let seed = 1; seed <= 50; seed += 1) {
    let state = autoResolve(finishMulligan(createGame(seed % 2 === 0, seed), false));
    let actions = 0;
    while (state.status === "playing" && state.globalTurn < 20 && actions < 250) {
      state = autoResolve(state);
      if (state.turnSide !== "player" || state.phase !== "main") {
        actions += 1;
        continue;
      }
      const playable = ([...state.player.hand.map((card) => ({ card, zone: "hand" as Zone })), ...state.player.ex.map((card) => ({ card, zone: "ex" as Zone }))])
        .filter(({ card, zone }) => cardActions(state, card.uid, zone).some((action) => action.id === "play" && action.enabled))
        .sort((a, b) => b.card.cardId.localeCompare(a.card.cardId))[0];
      if (playable) state = autoResolve(playCard(state, playable.card.uid, playable.zone));
      else state = autoResolve(endTurn(state));
      actions += 1;
    }
    assert.ok(actions < 250, `seed ${seed} entered an action loop`);
    assert.equal(state.pending, undefined, `seed ${seed} left a pending choice`);
  }
});
