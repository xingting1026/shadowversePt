import assert from "node:assert/strict";
import test from "node:test";
import {
  __testing,
  attackTargets,
  cardActions,
  createGame,
  endTurn,
  finishMulligan,
  playCard,
  resolveChoice,
  type CardInstance,
  type GameState,
  type Zone,
} from "../src/game/engine.ts";

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

test("ten deterministic games advance through repeated player and AI turns without an unresolved loop", () => {
  for (let seed = 1; seed <= 10; seed += 1) {
    let state = autoResolve(finishMulligan(createGame(seed % 2 === 0, seed), false));
    let actions = 0;
    while (state.status === "playing" && state.globalTurn < 12 && actions < 120) {
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
    assert.ok(actions < 120, `seed ${seed} entered an action loop`);
    assert.equal(state.pending, undefined, `seed ${seed} left a pending choice`);
  }
});
