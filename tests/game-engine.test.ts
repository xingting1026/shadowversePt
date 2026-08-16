import assert from "node:assert/strict";
import test from "node:test";
import {
  __testing,
  activateFieldCard,
  attackCardForSide,
  attackTargets,
  cardActions,
  createGame,
  definition,
  endTurn,
  evolveCard,
  finishManualMulligan,
  finishMulligan,
  hasKeyword,
  playCard,
  playCardForSide,
  resolveChoice,
  restartWithSameSeed,
  type CardInstance,
  type GameState,
  type Zone,
} from "../src/game/engine.ts";
import { CARDS, PLAYER_DECKS, SEKKA_DECK, SEKKA_EVOLVE, isLevinCard } from "../src/game/cards.ts";
import {
  applyTrainingAction,
  createTrainingReplay,
  finalizeTrainingReplay,
  recordTrainingDecision,
  replayTrainingGame,
  trainingLegalActions,
  trainingObservation,
  trainingActor,
  trainingReward,
  type TrainingAction,
} from "../src/game/training.ts";
import {
  encodeTrainingAction,
  encodeTrainingState,
  FROZEN_CARD_ORDER,
  TRAINING_CARD_IDS,
  TRAINING_ENCODING_METADATA,
} from "../src/game/training-encoding.ts";
import { buildBrowserPolicyInputs } from "../src/game/browser-policy-input.ts";

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

test("seed mixing keeps replays deterministic while decorrelating neighboring seeds", () => {
  const signature = (seed: number) => {
    const state = createGame(true, seed, "fairy");
    return [...state.player.hand, ...state.ai.hand].map((card) => card.cardId).join(",");
  };
  assert.equal(signature(50_000_000), signature(50_000_000));
  const signatures = new Set(Array.from({ length: 64 }, (_, index) => signature(50_000_000 + index)));
  assert.ok(signatures.size >= 60, `neighboring seeds produced only ${signatures.size} distinct openings`);
});

test("the training observation exposes the player's hand but never the AI hand or either deck order", () => {
  const state = createGame(true, 123);
  const observation = trainingObservation(state);
  assert.equal(observation.self.hand?.length, 4);
  assert.equal(observation.opponent.hand, undefined);
  assert.equal(observation.self.deckCount, 36);
  assert.equal(observation.opponent.deckCount, 36);
  assert.equal("deck" in observation.self, false);
  assert.equal("deck" in observation.opponent, false);
  assert.equal(Object.values(observation.ownDeckList).reduce((sum, count) => sum + count, 0), 40);
  assert.equal(Object.values(observation.opponentDeckList).reduce((sum, count) => sum + count, 0), 40);
});

test("the neural encoding is fixed-size and invariant to hidden deck order and opponent hidden-card allocation", () => {
  const state = createGame(true, 456, "fairy");
  const encoded = encodeTrainingState(state);
  assert.equal(encoded.scalars.length, TRAINING_ENCODING_METADATA.scalarSize);
  assert.equal(encoded.zones.length, TRAINING_ENCODING_METADATA.zoneNames.length);
  assert.ok(encoded.field.length <= TRAINING_ENCODING_METADATA.fieldSlots);
  assert.ok(encoded.recentEvents.length <= TRAINING_ENCODING_METADATA.recentEventSlots);

  const reordered = structuredClone(state);
  reordered.player.deck.reverse();
  assert.deepEqual(encodeTrainingState(reordered), encoded);

  const opponentHiddenSwap = structuredClone(state);
  const handCard = opponentHiddenSwap.ai.hand[0];
  const deckCard = opponentHiddenSwap.ai.deck[0];
  opponentHiddenSwap.ai.hand[0] = deckCard;
  deckCard.zone = "hand";
  opponentHiddenSwap.ai.deck[0] = handCard;
  handCard.zone = "deck";
  assert.deepEqual(encodeTrainingState(opponentHiddenSwap), encoded);

  for (const action of trainingLegalActions(state)) {
    const actionEncoding = encodeTrainingAction(state, action);
    assert.equal(actionEncoding.numbers.length, TRAINING_ENCODING_METADATA.actionNumberSize);
    assert.ok(actionEncoding.selectedCards.length <= TRAINING_ENCODING_METADATA.actionSelectionSlots);
    assert.ok(actionEncoding.selectedSpecials.length <= TRAINING_ENCODING_METADATA.actionSelectionSlots);
  }
});

test("browser policy tensors match the training tensor schema on a real AI mulligan", () => {
  let state = createGame(false, 20260816, "levin", { aiControl: "manual" });
  state = finishManualMulligan(state, "player", false);
  assert.equal(trainingActor(state), "ai");
  const actions = trainingLegalActions(state);
  const inputs = buildBrowserPolicyInputs(state, actions);
  assert.deepEqual(inputs.scalars.dims, [1, TRAINING_ENCODING_METADATA.scalarSize]);
  assert.deepEqual(inputs.zone_counts.dims, [1, TRAINING_ENCODING_METADATA.zoneNames.length, TRAINING_ENCODING_METADATA.cardVocabularySize]);
  assert.deepEqual(inputs.field_numbers.dims, [1, TRAINING_ENCODING_METADATA.fieldSlots, TRAINING_ENCODING_METADATA.fieldNumberSize]);
  assert.deepEqual(inputs.numbers.dims, [1, actions.length, TRAINING_ENCODING_METADATA.actionNumberSize]);
  assert.deepEqual(inputs.mask.dims, [1, actions.length]);
  assert.deepEqual(Array.from(inputs.mask.data as Uint8Array), new Array(actions.length).fill(1));
  const actionNumbers = Array.from(inputs.numbers.data as Float32Array);
  assert.equal(actionNumbers[12], 0, "keep action must encode redraw=false");
  assert.equal(actionNumbers[TRAINING_ENCODING_METADATA.actionNumberSize + 12], 1, "redraw action must encode redraw=true");

  const restarted = restartWithSameSeed(state);
  assert.equal(restarted.aiControl, "manual", "a model game must not silently restart with scripted AI");
});

test("the training environment enumerates mulligan, ordered choices, and optional subsets", () => {
  let state = createGame(true, 123);
  assert.deepEqual(trainingLegalActions(state).map((action) => action.key), ["mulligan:player:keep", "mulligan:player:redraw"]);
  state = applyTrainingAction(state, trainingLegalActions(state)[0]);
  state.pending = {
    kind: "order",
    effect: "testOrder",
    title: "測試排序",
    prompt: "排序三張卡",
    options: [
      { uid: "a", label: "A" },
      { uid: "b", label: "B" },
      { uid: "c", label: "C" },
    ],
    min: 3,
    max: 3,
  };
  assert.equal(trainingLegalActions(state).length, 6);
  state.pending = {
    kind: "multi",
    effect: "testSubset",
    title: "測試子集",
    prompt: "任選",
    options: [{ uid: "a" }, { uid: "b" }, { uid: "c" }],
    min: 0,
    max: 3,
  };
  assert.equal(trainingLegalActions(state).length, 8);
});

test("manual-AI matches wait for both mulligans and expose the Destruction turn instead of auto-playing it", () => {
  let state = createGame(false, 8080, "fairy", { aiControl: "manual" });
  assert.deepEqual(trainingLegalActions(state).map((action) => action.key), ["mulligan:player:keep", "mulligan:player:redraw"]);
  state = applyTrainingAction(state, trainingLegalActions(state)[0]);
  assert.equal(state.status, "mulligan");
  assert.deepEqual(trainingLegalActions(state).map((action) => action.key), ["mulligan:ai:keep", "mulligan:ai:redraw"]);
  state = applyTrainingAction(state, trainingLegalActions(state)[0]);
  assert.equal(state.status, "playing");
  assert.equal(state.turnSide, "ai");
  assert.equal(state.phase, "main");
  assert.equal(state.events.filter((event) => event.type === "play").length, 0);
  assert.ok(trainingLegalActions(state).some((action) => action.kind === "end" && action.key === "end:ai"));
  assert.equal(trainingObservation(state).actor, "ai");
  assert.deepEqual(trainingObservation(state).self.hand, state.ai.hand.map((card) => card.cardId));
  assert.equal(trainingObservation(state).opponent.hand, undefined);
});

test("manual Destruction policy receives explicit play, attack-target, and compound activation candidates", () => {
  let state = createGame(false, 8181, "fairy", { aiControl: "manual" });
  state = applyTrainingAction(state, trainingLegalActions(state)[0]);
  state = applyTrainingAction(state, trainingLegalActions(state)[0]);
  state.ai.pp = 10;
  state.ai.maxPP = 10;
  state.ai.hand = [__testing.makeInstance(state, "destructionJoy", "ai", "hand")];
  const play = trainingLegalActions(state).find((action) => action.kind === "play");
  assert.ok(play);
  state = applyTrainingAction(state, play);
  assert.equal(state.events.at(-1)?.side, "ai");

  const prayer = addField(state, "ai", "destructionPrayer");
  const sacrifice = addField(state, "ai", "destructionWilderness");
  const attacker = addField(state, "ai", "destructionHermit");
  attacker.enteredAt = state.globalTurn - 1;
  const target = addField(state, "player", "pureWaterFairy");
  target.tapped = true;
  const actions = trainingLegalActions(state);
  assert.ok(actions.some((action) => action.kind === "attack" && action.uid === attacker.uid && action.targetUid === target.uid));
  const prayerHit = actions.find((action) =>
    action.kind === "activate" && action.abilityId === "prayer"
    && action.uid === prayer.uid && action.selected?.includes(sacrifice.uid) && action.selected?.includes(target.uid));
  assert.ok(prayerHit);
  state = applyTrainingAction(state, prayerHit);
  assert.equal(prayer.tapped, false, "the original input remains immutable");
  assert.equal(state.ai.field.find((card) => card.uid === prayer.uid)?.tapped, true);
  assert.equal(state.ai.field.some((card) => card.uid === sacrifice.uid), false);
});

test("manual Destruction answers its own Quick window and can remove an attacker before combat", () => {
  let state = createGame(true, 8282, "fairy", { aiControl: "manual" });
  state = applyTrainingAction(state, trainingLegalActions(state)[0]);
  state = applyTrainingAction(state, trainingLegalActions(state)[0]);
  state.player.field = [];
  state.ai.field = [];
  state.ai.hand = [__testing.makeInstance(state, "whiteBlackChapter", "ai", "hand")];
  state.ai.pp = 2;
  const attacker = addField(state, "player", "pureWaterFairy");
  attacker.enteredAt = state.globalTurn - 1;
  const attack = trainingLegalActions(state).find((action) => action.kind === "attack" && action.targetUid === "ai-leader");
  assert.ok(attack);
  state = applyTrainingAction(state, attack);
  assert.equal(state.pending?.effect, "manualAiQuick");
  assert.equal(trainingActor(state), "ai");
  const quick = trainingLegalActions(state).find((action) => action.kind === "choice" && action.selected[0] === attacker.uid);
  assert.ok(quick);
  state = applyTrainingAction(state, quick);
  assert.equal(state.player.field.some((card) => card.uid === attacker.uid), false);
  assert.equal(state.ai.hp, 20, "a removed attacker does not deal combat damage");
});

test("manual Destruction fanfares expose target selection to the Destruction actor", () => {
  let state = createGame(false, 8383, "fairy", { aiControl: "manual" });
  state = applyTrainingAction(state, trainingLegalActions(state)[0]);
  state = applyTrainingAction(state, trainingLegalActions(state)[0]);
  state.ai.field = [];
  state.ai.hand = [__testing.makeInstance(state, "destructionHermit", "ai", "hand")];
  state.ai.pp = 5;
  addField(state, "ai", "destructionWilderness");
  addField(state, "ai", "destructionPrayer");
  const targetA = addField(state, "player", "fairyDragon");
  const targetB = addField(state, "player", "pureWaterFairy");
  const play = trainingLegalActions(state).find((action) => action.kind === "play" && action.cardId === "destructionHermit");
  assert.ok(play);
  state = applyTrainingAction(state, play);
  assert.equal(state.pending?.effect, "manualHermitTarget");
  assert.equal(trainingActor(state), "ai");
  const chooseB = trainingLegalActions(state).find((action) => action.kind === "choice" && action.selected[0] === targetB.uid);
  assert.ok(chooseB);
  state = applyTrainingAction(state, chooseB);
  assert.ok(state.player.field.some((card) => card.uid === targetA.uid));
  assert.equal(state.player.field.some((card) => card.uid === targetB.uid), false);
});

function baselineTrainingAction(actions: TrainingAction[]): TrainingAction {
  const choice = actions.find((action) => action.kind === "choice");
  if (choice) return choice;
  const play = actions
    .filter((action): action is Extract<TrainingAction, { kind: "play" }> => action.kind === "play")
    .sort((a, b) => definition(b.cardId).cost - definition(a.cardId).cost)[0];
  if (play) return play;
  const evolve = actions.find((action) => action.kind === "evolve");
  if (evolve) return evolve;
  const attack = actions.find((action) => action.kind === "activate" && action.abilityId === "attack");
  if (attack) return attack;
  const activate = actions.find((action) => action.kind === "activate");
  if (activate) return activate;
  const mulligan = actions.find((action) => action.kind === "mulligan" && !action.redraw);
  if (mulligan) return mulligan;
  const end = actions.find((action) => action.kind === "end");
  if (end) return end;
  throw new Error("training environment returned no selectable action");
}

test("headless Fairy versus Destruction games can run to completion through only the training API", () => {
  for (let seed = 1; seed <= 12; seed += 1) {
    let state = createGame(seed % 2 === 0, seed, "fairy");
    let decisions = 0;
    while (state.status !== "gameover" && decisions < 400) {
      const actions = trainingLegalActions(state);
      assert.ok(actions.length > 0, `seed ${seed} has no legal training action`);
      state = applyTrainingAction(state, baselineTrainingAction(actions));
      decisions += 1;
    }
    assert.equal(state.status, "gameover", `seed ${seed} did not finish`);
    assert.ok([-1, 0, 1].includes(trainingReward(state)));
  }
});

test("a training replay reconstructs both sides' events and rejects a divergent action set", () => {
  let state = createGame(false, 90210, "fairy");
  let replay = createTrainingReplay(state);
  let decisions = 0;
  while (state.status !== "gameover" && decisions < 400) {
    const legal = trainingLegalActions(state);
    const action = baselineTrainingAction(legal);
    replay = recordTrainingDecision(replay, state, action, {
      value: 0,
      policy: [{ actionKey: action.key, probability: 1 }],
    });
    state = applyTrainingAction(state, action);
    decisions += 1;
  }
  assert.equal(state.status, "gameover");
  replay = finalizeTrainingReplay(replay, state);

  const reconstructed = replayTrainingGame(replay);
  assert.equal(reconstructed.states.length, replay.decisions.length + 1);
  assert.deepEqual(reconstructed.finalState.events, state.events);
  assert.deepEqual(reconstructed.finalState.log, state.log);
  assert.deepEqual(replay.result, {
    status: state.status,
    winner: state.winner,
    reward: trainingReward(state),
    globalTurn: state.globalTurn,
    playerHp: state.player.hp,
    aiHp: state.ai.hp,
    eventCount: state.events.length,
  });

  const divergent = structuredClone(replay);
  divergent.decisions[0].legalActionKeys.push("impossible-action");
  assert.throws(() => replayTrainingGame(divergent), /legal actions diverged/);
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

test("a Last Words choice caused by end-phase Quick resolves before the AI turn continues", () => {
  let state = playing(1002629);
  state.player.field = [];
  state.ai.field = [];
  state.ai.hand = [];
  state.ai.pp = 2;
  const aria = addField(state, "player", "miasmaAriaEvo");
  aria.baseCardId = "miasmaAria";
  aria.damage = Math.max(0, (definition(aria).health ?? 0) - 2);
  const quick = __testing.makeInstance(state, "whiteBlackChapter", "ai", "hand");
  state.ai.hand.push(quick);

  state = endTurn(state);
  assert.equal(state.turnSide, "player");
  assert.equal(state.pending?.effect, "miasmaLastWord");

  state = resolveChoice(state, ["no"]);
  assert.equal(state.turnSide, "player");
  assert.equal(state.phase, "main");
  assert.equal(state.pending, undefined);
});

test("a discard trigger caused by the hand limit resolves before the AI turn continues", () => {
  let state = playing(1001519);
  state.player.field = [];
  state.ai.field = [];
  state.ai.hand = [];
  const geno = __testing.makeInstance(state, "levinAxeGeno", "player", "hand");
  state.player.hand.push(geno);
  while (state.player.hand.length < 8) {
    state.player.hand.push(__testing.makeInstance(state, "levinMiim", "player", "hand"));
  }
  state.player.deck.push(__testing.makeInstance(state, "levinMiim", "player", "deck"));

  state = endTurn(state);
  assert.equal(state.pending?.effect, "discardToSeven");
  state = resolveChoice(state, [geno.uid]);
  assert.equal(state.turnSide, "player");
  assert.equal(state.pending?.effect, "genoDigPick");

  state = resolveChoice(state, []);
  assert.equal(state.turnSide, "player");
  assert.equal(state.phase, "main");
  assert.equal(state.pending, undefined);
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

// ───────────────────────── 任意牌組 vs 任意牌組（aiDeck）─────────────────────────

/** 固定 seed 的 PRNG（mulberry32），讓隨機對局測試可重現。 */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function pairPlaying(playerDeck: "fairy" | "levin" | "sekka", aiDeck: "fairy" | "levin" | "sekka", seed = 7, playerFirst = true): GameState {
  let state = createGame(playerFirst, seed, playerDeck, { aiControl: "manual", aiDeck });
  state = finishManualMulligan(state, "player", false);
  state = finishManualMulligan(state, "ai", false);
  return state;
}

function runRandomPairLeague(playerDeck: "fairy" | "levin" | "sekka", aiDeck: "fairy" | "levin" | "sekka"): void {
  for (let seed = 1; seed <= 50; seed += 1) {
    const rand = mulberry32(seed * 7919 + 17);
    let state = createGame(seed % 2 === 0, seed, playerDeck, { aiControl: "manual", aiDeck });
    let decisions = 0;
    while (state.status !== "gameover" && decisions < 500) {
      const actions = trainingLegalActions(state);
      assert.ok(actions.length > 0, `seed ${seed} (${playerDeck} vs ${aiDeck}) has no legal action after ${decisions} decisions`);
      state = applyTrainingAction(state, actions[Math.floor(rand() * actions.length)]);
      decisions += 1;
    }
    if (state.status !== "gameover") {
      assert.ok(trainingLegalActions(state).length > 0, `seed ${seed} (${playerDeck} vs ${aiDeck}) is stuck on a pending choice at the decision cap`);
    }
  }
}

test("a non-destruction AI deck demands manual control", () => {
  assert.throws(() => createGame(true, 1, "levin", { aiDeck: "fairy" }), /manual/);
  assert.throws(() => createGame(true, 1, "levin", { aiControl: "scripted", aiDeck: "fairy" }), /manual/);
  const state = createGame(true, 1, "levin", { aiControl: "manual", aiDeck: "fairy" });
  assert.equal(state.aiDeck, "fairy");
  assert.equal(createGame(true, 1, "levin").aiDeck, "destruction");
});

test("fifty random Levin-versus-Fairy pair games advance without a stall or a throw", () => {
  runRandomPairLeague("levin", "fairy");
});

test("fifty random Fairy-versus-Levin pair games advance without a stall or a throw", () => {
  runRandomPairLeague("fairy", "levin");
});

test("an AI-slot Miim counts its own side's hand and draws for the AI side", () => {
  const state = pairPlaying("fairy", "levin");
  state.ai.hand = [];
  const discardTarget = __testing.makeInstance(state, "levinSisters", "ai", "hand");
  state.ai.hand.push(discardTarget);
  const drawTarget = __testing.makeInstance(state, "gawain", "ai", "deck");
  state.ai.deck.push(drawTarget);
  const playerHandBefore = state.player.hand.length;
  const miim = addField(state, "ai", "levinMiim");
  __testing.resolveTask(state, { type: "fanfare", side: "ai", sourceUid: miim.uid, cardId: "levinMiim" });
  assert.equal(state.pending?.effect, "miimDiscard");
  assert.equal(state.pending?.side, "ai", "the AI side must answer its own Fanfare");
  assert.deepEqual(state.pending?.options.map((option) => option.uid), [discardTarget.uid], "only the AI hand is offered");
  const resolved = resolveChoice(state, [discardTarget.uid]);
  assert.equal(resolved.ai.grave.some((card) => card.uid === discardTarget.uid), true, "the discard goes to the AI graveyard");
  assert.equal(resolved.ai.hand.some((card) => card.uid === drawTarget.uid), true, "the draw comes from the AI deck");
  assert.equal(resolved.player.hand.length, playerHandBefore, "the player hand must stay untouched");
});

test("an AI-slot Cynthia grants Storm to AI-side fairy tokens only", () => {
  const state = pairPlaying("levin", "fairy");
  addField(state, "ai", "queenCynthia");
  const aiToken = addField(state, "ai", "fairy");
  const playerToken = addField(state, "player", "fairy");
  assert.equal(hasKeyword(state, aiToken, "storm"), true);
  assert.equal(hasKeyword(state, aiToken, "designated"), true);
  assert.equal(hasKeyword(state, playerToken, "storm"), false, "the player token must not borrow the AI-side Cynthia");
});

test("an AI-slot Brutal Geno discounts by sacrificing an AI-side Levin follower", () => {
  let state = pairPlaying("fairy", "levin", 11, false);
  assert.equal(state.turnSide, "ai");
  state.player.field = [];
  state.ai.field = [];
  state.ai.pp = 1;
  state.ai.maxPP = 4;
  const runes = addField(state, "ai", "levinRunes");
  const geno = __testing.makeInstance(state, "brutalGeno", "ai", "hand");
  state.ai.hand = [geno];
  state = playCardForSide(state, "ai", geno.uid, "hand");
  assert.equal(state.pending?.effect, "brutalGenoPlay");
  assert.equal(state.pending?.side, "ai");
  assert.deepEqual(state.pending?.options.map((option) => option.uid), [runes.uid], "only the AI-side cheap Levin follower is a legal sacrifice");
  state = resolveChoice(state, [runes.uid]);
  assert.equal(state.ai.field.some((card) => card.cardId === "brutalGeno"), true);
  assert.equal(state.ai.grave.some((card) => card.uid === runes.uid), true, "the sacrifice lands in the AI graveyard");
  assert.equal(state.ai.pp, 0, "the discounted cost is one PP");
});

test("a pair-mode replay records the AI deck and reconstructs the full game", () => {
  let state = createGame(false, 4242, "levin", { aiControl: "manual", aiDeck: "fairy" });
  let replay = createTrainingReplay(state);
  assert.equal(replay.aiDeck, "fairy");
  const rand = mulberry32(4242);
  let decisions = 0;
  while (state.status !== "gameover" && decisions < 500) {
    const legal = trainingLegalActions(state);
    const action = legal[Math.floor(rand() * legal.length)];
    replay = recordTrainingDecision(replay, state, action);
    state = applyTrainingAction(state, action);
    decisions += 1;
  }
  replay = finalizeTrainingReplay(replay, state);
  const reconstructed = replayTrainingGame(replay);
  assert.deepEqual(reconstructed.finalState.events, state.events);
  assert.equal(reconstructed.finalState.winner, state.winner);
});

// ───────────────────────── 雪華獸（セッカ）牌組 ─────────────────────────

function sekkaPlaying(seed = 7): GameState {
  return finishMulligan(createGame(true, seed, "sekka"), false);
}

test("the frozen card order is a strict prefix of TRAINING_CARD_IDS which covers every card exactly once", () => {
  assert.ok(FROZEN_CARD_ORDER.length <= TRAINING_CARD_IDS.length);
  FROZEN_CARD_ORDER.forEach((cardId, index) => {
    assert.equal(TRAINING_CARD_IDS[index], cardId, `frozen index ${index} moved`);
  });
  assert.equal(new Set(TRAINING_CARD_IDS).size, TRAINING_CARD_IDS.length, "TRAINING_CARD_IDS contains duplicates");
  assert.deepEqual([...TRAINING_CARD_IDS].sort(), Object.keys(CARDS).sort(), "TRAINING_CARD_IDS must cover CARDS exactly");
});

test("the Sekka deck lists 40 main and 10 evolve cards that all exist and carry the elf class trait", () => {
  assert.equal(SEKKA_DECK.reduce((total, [, count]) => total + count, 0), 40);
  assert.equal(SEKKA_EVOLVE.reduce((total, [, count]) => total + count, 0), 10);
  assert.equal(PLAYER_DECKS.sekka.main, SEKKA_DECK);
  assert.equal(PLAYER_DECKS.sekka.evolve, SEKKA_EVOLVE);
  for (const [cardId] of [...SEKKA_DECK, ...SEKKA_EVOLVE]) {
    assert.ok(CARDS[cardId], `${cardId} is missing from CARDS`);
    assert.ok(CARDS[cardId].traits.includes("エルフ"), `${cardId} lacks the エルフ class trait`);
  }
  assert.ok(CARDS.greenManifest.traits.includes("エルフ"));
  assert.equal(CARDS.greenManifest.token, true);
  const state = createGame(true, 5, "sekka");
  assert.equal(state.player.deck.length + state.player.hand.length, 40);
  assert.equal(Object.values(state.player.evolveRemaining).reduce((a, b) => a + b, 0), 10);
  assert.equal(state.player.evolveRemaining.sekkaAdvance, 3);
});

test("Foxfire Sekka ignites from the graveyard: pays 3PP, banishes three cards, and fields the Advance with its Fanfare", () => {
  let state = sekkaPlaying();
  state.player.field = [];
  state.player.grave = [];
  state.player.banished = [];
  state.player.pp = 5;
  const fox = __testing.makeInstance(state, "foxfireSekka", "player", "grave");
  const costA = __testing.makeInstance(state, "owlman", "player", "grave");
  const costB = __testing.makeInstance(state, "castel", "player", "grave");
  state.player.grave.push(fox, costA, costB);

  const ignite = cardActions(state, fox.uid, "grave").find((action) => action.id === "sekkaIgnite");
  assert.equal(ignite?.enabled, true);
  assert.ok(
    trainingLegalActions(state).some((action) => action.kind === "activate" && action.abilityId === "sekkaIgnite" && action.uid === fox.uid),
    "the training environment must enumerate graveyard activations",
  );

  state = activateFieldCard(state, fox.uid, "sekkaIgnite");
  assert.equal(state.pending?.effect, "sekkaIgniteCost");
  state = resolveChoice(state, [costA.uid, costB.uid]);
  assert.equal(state.player.pp, 2, "3PP paid");
  assert.equal(state.player.grave.length, 0);
  assert.equal(state.player.banished.length, 3, "the fox and both costs are banished");
  assert.equal(state.player.field.some((card) => card.cardId === "sekkaAdvance"), true);
  assert.equal(state.player.evolveRemaining.sekkaAdvance, 2);
  assert.equal(state.player.evolveUsed.sekkaAdvance, 1);

  assert.equal(state.pending?.effect, "sekkaAdvanceMode", "the Advance Fanfare offers its two modes");
  state = resolveChoice(state, ["deck"]);
  assert.equal(state.pending?.effect, "sekkaAdvanceDeckPick");
  const pick = state.pending!.options[0];
  const rngBefore = state.rng;
  state = resolveChoice(state, [pick.uid]);
  assert.equal(state.player.hand.some((card) => card.uid === pick.uid && card.cardId === "ninetailResolve"), true);
  assert.notEqual(state.rng, rngBefore, "the deck is shuffled after the search");
});

test("a graveyard Foxfire without 3PP, two extra cards, or an Advance copy cannot ignite", () => {
  const state = sekkaPlaying();
  state.player.grave = [];
  const fox = __testing.makeInstance(state, "foxfireSekka", "player", "grave");
  state.player.grave.push(fox);
  state.player.pp = 5;
  assert.equal(cardActions(state, fox.uid, "grave").find((action) => action.id === "sekkaIgnite")?.enabled, false, "needs two other grave cards");
  state.player.grave.push(
    __testing.makeInstance(state, "owlman", "player", "grave"),
    __testing.makeInstance(state, "owlman", "player", "grave"),
  );
  state.player.pp = 2;
  assert.equal(cardActions(state, fox.uid, "grave").find((action) => action.id === "sekkaIgnite")?.enabled, false, "needs 3PP");
  state.player.pp = 5;
  state.player.evolveRemaining.sekkaAdvance = 0;
  assert.equal(cardActions(state, fox.uid, "grave").find((action) => action.id === "sekkaIgnite")?.enabled, false, "needs an Advance in the evolve zone");
});

test("Castel blocks ability damage for beast followers while combat damage still applies", () => {
  let state = sekkaPlaying();
  state.player.field = [];
  state.ai.field = [];
  const owl = addField(state, "player", "owlman");
  const castel = addField(state, "player", "castel");
  __testing.resolveTask(state, { type: "fanfare", side: "ai", cardId: "greatZelgenea" });
  __testing.runTasks(state);
  assert.equal(state.player.field.find((card) => card.uid === owl.uid)?.damage, 0, "the beast takes zero ability damage");
  assert.equal(state.player.field.some((card) => card.uid === castel.uid), false, "Castel does not protect a same-named follower (itself)");

  addField(state, "player", "castel");
  const blocker = addField(state, "ai", "destructionServant");
  blocker.tapped = true;
  owl.enteredAt = state.globalTurn - 1;
  state = attackCardForSide(state, "player", owl.uid, blocker.uid);
  assert.equal(state.player.field.some((card) => card.uid === owl.uid), false, "combat damage ignores Castel");
  assert.equal(state.ai.field.some((card) => card.uid === blocker.uid), false);
});

test("Sword Rabbit unlocks its zero-cost evolve after a bounce and can field a hand copy when it returns", () => {
  let state = sekkaPlaying();
  state.player.field = [];
  state.player.hand = [];
  const rabbit = addField(state, "player", "swordRabbit");
  assert.equal(cardActions(state, rabbit.uid, "field").find((action) => action.id === "evolve-pp")?.enabled, false, "no bounce yet");

  const carb = addField(state, "player", "babyCarbuncle");
  state.player.deck.push(__testing.makeInstance(state, "owlman", "player", "deck"));
  __testing.resolveTask(state, { type: "fanfare", side: "player", sourceUid: carb.uid, cardId: "babyCarbuncle" });
  assert.equal(state.pending?.effect, "carbuncleBounce");
  const handBefore = state.player.hand.length;
  state = resolveChoice(state, [rabbit.uid]);
  assert.equal(state.pending?.effect, "rabbitReturn", "the returned rabbit triggers its own replacement window");
  assert.equal(state.player.hand.length, handBefore + 2, "the rabbit returned and the carbuncle drew one");
  state = resolveChoice(state, [rabbit.uid]);
  assert.equal(state.player.field.some((card) => card.uid === rabbit.uid), true, "a hand rabbit is fielded again");

  state.player.pp = 0;
  assert.equal(cardActions(state, rabbit.uid, "field").find((action) => action.id === "evolve-pp")?.enabled, true, "zero-cost evolve unlocked by the bounce");
  state = evolveCard(state, rabbit.uid, "pp");
  assert.equal(state.player.field.find((card) => card.uid === rabbit.uid)?.cardId, "swordRabbitEvo");
});

test("Salvia Panther costs three normally and one after another beast follower bounced this turn", () => {
  let state = sekkaPlaying();
  state.player.field = [];
  const salvia = __testing.makeInstance(state, "salviaPanther", "player", "hand");
  state.player.hand = [salvia];
  state.player.pp = 1;
  state.player.maxPP = 4;
  assert.equal(cardActions(state, salvia.uid, "hand").find((action) => action.id === "play")?.enabled, false, "one PP is not enough without a bounce");

  const owl = addField(state, "player", "owlman");
  const carb = addField(state, "player", "babyCarbuncle");
  state.player.deck.push(__testing.makeInstance(state, "castel", "player", "deck"));
  __testing.resolveTask(state, { type: "fanfare", side: "player", sourceUid: carb.uid, cardId: "babyCarbuncle" });
  state = resolveChoice(state, [owl.uid]);
  assert.deepEqual(state.bouncedThisTurn.player, ["owlman"]);
  const salviaInHand = state.player.hand.find((card) => card.uid === salvia.uid)!;
  assert.equal(cardActions(state, salviaInHand.uid, "hand").find((action) => action.id === "play")?.enabled, true, "the discount enables the play at 1PP");
  state = playCard(state, salviaInHand.uid, "hand");
  assert.equal(state.player.field.some((card) => card.cardId === "salviaPanther"), true);
  assert.equal(state.player.pp, 0, "the discounted cost is one PP");
});

test("Tiger Girl may banish three grave beasts after declaring an attack, then combat resolves either way", () => {
  const setup = () => {
    const state = sekkaPlaying();
    state.player.field = [];
    state.ai.field = [];
    state.player.grave = [];
    state.player.banished = [];
    const tiger = addField(state, "player", "tigerGirl");
    tiger.enteredAt = state.globalTurn - 1;
    const target = addField(state, "ai", "destructionServant");
    target.tapped = true;
    const other = addField(state, "ai", "destructionFanatic");
    for (let index = 0; index < 3; index += 1) {
      state.player.grave.push(__testing.makeInstance(state, "owlman", "player", "grave"));
    }
    return { state, tiger, target, other };
  };

  // 發動線：先射掉另一體從者，戰鬥照常結算。
  {
    const { state, tiger, target, other } = setup();
    let next = attackCardForSide(state, "player", tiger.uid, target.uid);
    assert.equal(next.pending?.effect, "tigerGirlStrike");
    next = resolveChoice(next, [other.uid]);
    assert.equal(next.player.banished.length, 3, "three grave beasts are banished");
    assert.equal(next.ai.field.some((card) => card.uid === other.uid), false, "the strike killed the 3/4");
    assert.equal(next.ai.field.some((card) => card.uid === target.uid), false, "combat still killed the attack target");
    const still = next.player.field.find((card) => card.uid === tiger.uid);
    assert.equal(still?.damage, 2, "the tiger took combat damage back");
    assert.equal(still?.tapped, true);
  }

  // 不發動線：pass 後戰鬥照常，墓場不動。
  {
    const { state, tiger, target, other } = setup();
    let next = attackCardForSide(state, "player", tiger.uid, target.uid);
    assert.equal(next.pending?.effect, "tigerGirlStrike");
    next = resolveChoice(next, ["pass"]);
    assert.equal(next.player.banished.length, 0);
    assert.equal(next.player.grave.length, 3);
    assert.equal(next.ai.field.some((card) => card.uid === other.uid), true, "the other follower survives");
    assert.equal(next.ai.field.some((card) => card.uid === target.uid), false, "combat resolved normally");
    assert.equal(next.player.field.find((card) => card.uid === tiger.uid)?.damage, 2);
  }
});

test("Cetus refunds one PP per copy per turn when a beast card is played, and the flag resets next turn", () => {
  let state = sekkaPlaying();
  state.player.field = [];
  addField(state, "player", "cetusSun");
  state.player.pp = 5;
  state.player.maxPP = 5;
  const owlA = __testing.makeInstance(state, "owlman", "player", "hand");
  const owlB = __testing.makeInstance(state, "owlman", "player", "hand");
  state.player.hand = [owlA, owlB];

  state = autoResolve(playCard(state, owlA.uid, "hand"));
  assert.equal(state.player.pp, 5, "1PP paid and 1PP refunded (capped at maxPP)");
  const cetus = state.player.field.find((card) => card.cardId === "cetusSun")!;
  assert.equal(cetus.flags.cetusPpUsed, true);

  state = autoResolve(playCard(state, owlB.uid, "hand"));
  assert.equal(state.player.pp, 4, "the second beast play this turn is not refunded");

  __testing.beginTurnMutable(state, "player");
  assert.equal(state.player.field.find((card) => card.cardId === "cetusSun")?.flags.cetusPpUsed, undefined, "the per-turn flag resets");
});

function isFollowerInstance(card: CardInstance): boolean {
  return CARDS[card.cardId].kind === "follower";
}

test("Green Manifest enforces the combined-cost-five limit in both enumeration and resolution", () => {
  let state = sekkaPlaying();
  state.player.field = [];
  state.player.ex = [];
  state.player.pp = 5;
  state.player.maxPP = 5;
  __testing.addExDirect(state, "player", ["greenManifest"]);
  const green = state.player.ex[0];
  const top = ["owlman", "castel", "cuttingCat", "cetusSun", "tigerGirl", "heroResolve", "ninetailResolve"]
    .map((cardId) => __testing.makeInstance(state, cardId, "player", "deck"));
  state.player.deck.push(...[...top].reverse());

  state = playCard(state, green.uid, "ex");
  assert.equal(state.pending?.effect, "greenManifestPick");
  const optionUid = (cardId: string) => state.pending!.options.find((option) => option.cardId === cardId)!.uid;
  const cetusUid = optionUid("cetusSun");
  const cuttingUid = optionUid("cuttingCat");
  const owlUid = optionUid("owlman");
  const castelUid = optionUid("castel");

  const keys = trainingLegalActions(state).map((action) => action.key);
  assert.ok(!keys.some((key) => key.includes(cetusUid) && key.includes(cuttingUid)), "a 6-cost combination must not be enumerated");
  assert.ok(keys.includes(`choice:${owlUid},${castelUid},${cuttingUid}`), "a 4-cost triple stays legal");
  assert.ok(keys.includes(`choice:${cetusUid}`), "the 4-cost single stays legal");

  const rejected = resolveChoice(state, [cetusUid, cuttingUid]);
  assert.equal(rejected.pending?.effect, "greenManifestPick", "an over-cost selection is rejected and the choice stays open");

  state = resolveChoice(state, [owlUid, cuttingUid]);
  assert.equal(state.player.field.filter(isFollowerInstance).length, 2, "both picks entered the field");
  assert.equal(state.pending?.effect, "bottomOrder");
  state = autoResolve(state);
  for (const card of state.player.field) {
    assert.ok(card.attackBuff >= 1, `${card.cardId} should carry the +1/+1 from Green Manifest`);
  }
});

test("fifty random Sekka-versus-Levin pair games advance without a stall or a throw", () => {
  runRandomPairLeague("sekka", "levin");
});

test("fifty random Levin-versus-Sekka pair games advance without a stall or a throw", () => {
  runRandomPairLeague("levin", "sekka");
});
