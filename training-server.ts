/** Persistent JSON-lines bridge between the TypeScript rules engine and PyTorch. */
import { createInterface } from "node:readline";
import { createGame, type AiDeckId, type GameState } from "./src/game/engine.ts";
import {
  applyTrainingAction,
  createTrainingReplay,
  finalizeTrainingReplay,
  recordTrainingDecision,
  trainingActor,
  trainingLegalActions,
  trainingReward,
  trainingRewardForSide,
  type TrainingDecisionAudit,
  type TrainingReplay,
} from "./src/game/training.ts";
import {
  encodeTrainingAction,
  encodeTrainingState,
  TRAINING_ENCODING_METADATA,
} from "./src/game/training-encoding.ts";
import type { PlayerDeckId } from "./src/game/cards.ts";

type AuditInput = {
  value?: number;
  probabilities?: number[];
  note?: string;
};

type InitRequest = {
  cmd: "init";
  envs: number;
  deck: PlayerDeckId;
  /** ai 槽牌組；預設 "destruction"。非 destruction 時必須 selfPlay=true。 */
  aiDeck?: AiDeckId;
  baseSeed?: number;
  record?: boolean;
  fixedFirst?: boolean;
  selfPlay?: boolean;
};

type StepRequest = {
  cmd: "step";
  actions: number[];
  audits?: AuditInput[];
};

type InspectRequest = { cmd: "inspect" };
type CloseRequest = { cmd: "close" };
type Request = InitRequest | StepRequest | InspectRequest | CloseRequest;

type EnvironmentSlot = {
  id: number;
  episode: number;
  seed: number;
  playerFirst: boolean;
  decisions: number;
  state: GameState;
  recentActions: { key: string; label: string }[];
  replay?: TrainingReplay;
};

let slots: EnvironmentSlot[] = [];
let deck: PlayerDeckId = "fairy";
let aiDeck: AiDeckId = "destruction";
let baseSeed = 1;
let record = false;
let fixedFirst: boolean | undefined;
let selfPlay = false;

function seedFor(id: number, episode: number): number {
  return (baseSeed + id + episode * Math.max(1, slots.length)) >>> 0;
}

function createSlot(id: number, episode: number): EnvironmentSlot {
  const seed = seedFor(id, episode);
  const playerFirst = fixedFirst ?? (seed % 2 === 0);
  const state = createGame(playerFirst, seed, deck, { aiControl: selfPlay ? "manual" : "scripted", aiDeck });
  return {
    id,
    episode,
    seed,
    playerFirst,
    decisions: 0,
    state,
    recentActions: [],
    replay: record ? createTrainingReplay(state) : undefined,
  };
}

function observation(slot: EnvironmentSlot) {
  const legal = trainingLegalActions(slot.state);
  const actor = trainingActor(slot.state);
  if (slot.state.status !== "gameover" && !legal.length) {
    const state = slot.state;
    const debug = {
      status: state.status,
      turnSide: state.turnSide,
      phase: state.phase,
      globalTurn: state.globalTurn,
      decisions: slot.decisions,
      pending: state.pending ? {
        kind: state.pending.kind,
        effect: state.pending.effect,
        title: state.pending.title,
        min: state.pending.min,
        max: state.pending.max,
        options: state.pending.options.map((option) => ({
          uid: option.uid,
          label: option.label,
          cardId: option.cardId,
          description: option.description,
        })),
      } : undefined,
      player: {
        hp: state.player.hp,
        pp: state.player.pp,
        hand: state.player.hand.map((card) => card.cardId),
        field: state.player.field.map((card) => card.cardId),
        ex: state.player.ex.map((card) => card.cardId),
      },
      ai: {
        hp: state.ai.hp,
        pp: state.ai.pp,
        hand: state.ai.hand.map((card) => card.cardId),
        field: state.ai.field.map((card) => ({ cardId: card.cardId, tapped: card.tapped, enteredAt: card.enteredAt })),
        ex: state.ai.ex.map((card) => card.cardId),
      },
      recentEvents: state.events.slice(-8).map((event) => ({ type: event.type, cardId: event.cardId, detail: event.detail })),
      recentLog: state.log.slice(-12),
      lastAction: state.lastAction,
      tasks: state.tasks,
      recentDecisions: slot.replay?.decisions.slice(-8).map((decision) => ({
        index: decision.index,
        turn: decision.globalTurn,
        action: decision.actionKey,
        label: decision.actionLabel,
      })) ?? slot.recentActions.slice(-8),
    };
    throw new Error(`Environment ${slot.id} has no legal action at seed ${slot.seed}: ${JSON.stringify(debug)}`);
  }
  return {
    id: slot.id,
    actor,
    state: encodeTrainingState(slot.state, actor ?? "player"),
    actions: legal.map((action) => encodeTrainingAction(slot.state, action)),
  };
}

function auditFor(slot: EnvironmentSlot, input: AuditInput | undefined): TrainingDecisionAudit | undefined {
  if (!input) return undefined;
  const legal = trainingLegalActions(slot.state);
  return {
    value: input.value,
    note: input.note,
    policy: input.probabilities?.map((probability, index) => ({
      actionKey: legal[index]?.key ?? `invalid:${index}`,
      probability,
    })),
  };
}

function initialize(request: InitRequest) {
  if (!Number.isInteger(request.envs) || request.envs < 1 || request.envs > 256) throw new Error("envs must be between 1 and 256");
  if ((request.aiDeck ?? "destruction") !== "destruction" && !request.selfPlay) {
    throw new Error(`aiDeck "${request.aiDeck}" requires selfPlay=true: the scripted rules AI only plays the Destruction deck`);
  }
  deck = request.deck;
  aiDeck = request.aiDeck ?? "destruction";
  baseSeed = (request.baseSeed ?? 1) >>> 0;
  record = Boolean(request.record);
  fixedFirst = request.fixedFirst;
  selfPlay = Boolean(request.selfPlay);
  slots = Array.from({ length: request.envs }, (_, id) => ({ id } as EnvironmentSlot));
  slots = slots.map((_, id) => createSlot(id, 0));
  return {
    ok: true,
    metadata: TRAINING_ENCODING_METADATA,
    engineVersion: slots[0].state.version,
    deck,
    aiDeck,
    selfPlay,
    observations: slots.map(observation),
  };
}

function step(request: StepRequest) {
  if (request.actions.length !== slots.length) throw new Error(`Expected ${slots.length} actions, received ${request.actions.length}`);
  const items = slots.map((slot, id) => {
    const actor = trainingActor(slot.state);
    if (!actor) throw new Error(`Environment ${id} has no active actor`);
    const legal = trainingLegalActions(slot.state);
    const selected = legal[request.actions[id]];
    if (!selected) throw new Error(`Illegal action index ${request.actions[id]} for environment ${id} (${legal.length} legal)`);
    slot.recentActions.push({ key: selected.key, label: selected.label });
    if (slot.recentActions.length > 16) slot.recentActions.shift();
    if (slot.replay) slot.replay = recordTrainingDecision(slot.replay, slot.state, selected, auditFor(slot, request.audits?.[id]));
    slot.state = applyTrainingAction(slot.state, selected);
    slot.decisions += 1;
    const truncated = slot.state.status !== "gameover" && slot.decisions >= 500;
    const done = slot.state.status === "gameover" || truncated;
    const rawReward = trainingReward(slot.state);
    const actorReward = trainingRewardForSide(slot.state, actor);
    let result;
    let completedReplay: TrainingReplay | undefined;
    if (done) {
      result = {
        seed: slot.seed,
        playerFirst: slot.playerFirst,
        winner: slot.state.winner,
        rawReward,
        globalTurn: slot.state.globalTurn,
        decisions: slot.decisions,
        playerHp: slot.state.player.hp,
        aiHp: slot.state.ai.hp,
        truncated,
      };
      if (slot.replay) completedReplay = finalizeTrainingReplay(slot.replay, slot.state);
      slots[id] = createSlot(id, slot.episode + 1);
    }
    return {
      id,
      actor,
      done,
      rawReward,
      actorReward,
      truncated,
      result,
      replay: completedReplay,
      observation: observation(slots[id]),
    };
  });
  return { ok: true, items };
}

function inspect() {
  return {
    ok: true,
    deck,
    slots: slots.map((slot) => ({ id: slot.id, episode: slot.episode, seed: slot.seed, decisions: slot.decisions, status: slot.state.status })),
  };
}

function handle(request: Request) {
  switch (request.cmd) {
    case "init": return initialize(request);
    case "step": return step(request);
    case "inspect": return inspect();
    case "close": return { ok: true, closing: true };
  }
}

const input = createInterface({ input: process.stdin, crlfDelay: Infinity });
input.on("line", (line) => {
  try {
    const request = JSON.parse(line) as Request;
    const response = handle(request);
    process.stdout.write(`${JSON.stringify(response)}\n`);
    if (request.cmd === "close") input.close();
  } catch (reason) {
    const message = reason instanceof Error ? reason.stack ?? reason.message : String(reason);
    process.stdout.write(`${JSON.stringify({ ok: false, error: message })}\n`);
  }
});
