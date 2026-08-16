import { readFile } from "node:fs/promises";
import { spawnSync } from "node:child_process";
import * as ort from "onnxruntime-web/wasm";
import { buildBrowserPolicyInputs } from "./src/game/browser-policy-input.ts";
import { createGame, type GameState } from "./src/game/engine.ts";
import { encodeTrainingAction, encodeTrainingState } from "./src/game/training-encoding.ts";
import {
  applyTrainingAction,
  trainingActor,
  trainingLegalActions,
  type TrainingAction,
} from "./src/game/training.ts";

type ReferenceObservation = {
  state: ReturnType<typeof encodeTrainingState>;
  actions: ReturnType<typeof encodeTrainingAction>[];
};

type ReferenceResult = { logits: number[]; value: number };

const modelPath = "public/models/destruction-cycle13.onnx";
const checkpointPath = "training-output/league-v1/destruction/current.pt";

function playerAction(actions: TrainingAction[]): TrainingAction {
  const mulligan = actions.find((action) => action.kind === "mulligan" && !action.redraw);
  if (mulligan) return mulligan;
  return actions.find((action) => action.kind === "choice" && action.selected.length > 0)
    ?? actions.find((action) => action.kind === "attack")
    ?? actions.find((action) => action.kind === "play")
    ?? actions.find((action) => action.kind === "activate")
    ?? actions.find((action) => action.kind === "evolve")
    ?? actions.at(-1)!;
}

async function onnxDecision(session: ort.InferenceSession, state: GameState) {
  const actions = trainingLegalActions(state);
  const raw = buildBrowserPolicyInputs(state, actions);
  const feeds: Record<string, ort.Tensor> = {};
  for (const [name, input] of Object.entries(raw)) feeds[name] = new ort.Tensor(input.type, input.data, input.dims);
  const started = performance.now();
  const output = await session.run(feeds);
  const inferenceMs = performance.now() - started;
  const logits = Array.from(output.logits.data as Float32Array);
  const value = Number((output.value.data as Float32Array)[0]);
  let best = 0;
  for (let index = 1; index < logits.length; index += 1) if (logits[index] > logits[best]) best = index;
  return { action: actions[best], actions, logits, value, inferenceMs };
}

ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;
const session = await ort.InferenceSession.create(await readFile(modelPath), {
  executionProviders: ["wasm"],
  graphOptimizationLevel: "all",
});

const observations: ReferenceObservation[] = [];
const actual: ReferenceResult[] = [];
let aiDecisions = 0;
let completedGames = 0;
let maxInferenceMs = 0;

for (const [index, deck] of (["fairy", "levin"] as const).entries()) {
  let state = createGame(index === 0, 84_000 + index, deck, { aiControl: "manual" });
  let steps = 0;
  while (state.status !== "gameover" && steps < 500) {
    const actor = trainingActor(state);
    if (!actor) throw new Error(`no actor at step ${steps}, phase=${state.phase}, status=${state.status}`);
    const actions = trainingLegalActions(state);
    if (!actions.length) throw new Error(`no legal actions for ${actor} at step ${steps}`);
    if (actor === "ai") {
      const decision = await onnxDecision(session, state);
      aiDecisions += 1;
      maxInferenceMs = Math.max(maxInferenceMs, decision.inferenceMs);
      if (observations.length < 48) {
        observations.push({
          state: encodeTrainingState(state, "ai"),
          actions: decision.actions.map((action) => encodeTrainingAction(state, action)),
        });
        actual.push({ logits: decision.logits, value: decision.value });
      }
      state = applyTrainingAction(state, decision.action);
    } else {
      state = applyTrainingAction(state, playerAction(actions));
    }
    steps += 1;
  }
  if (state.status === "gameover") completedGames += 1;
  if (steps >= 500) throw new Error(`${deck} parity game exceeded 500 decisions`);
}

const referenceProcess = spawnSync(
  "python",
  ["training/browser_parity_reference.py", "--checkpoint", checkpointPath],
  { input: JSON.stringify(observations), encoding: "utf8", maxBuffer: 16 * 1024 * 1024 },
);
if (referenceProcess.status !== 0) throw new Error(referenceProcess.stderr || `Python parity exited ${referenceProcess.status}`);
const expected = JSON.parse(referenceProcess.stdout) as ReferenceResult[];
let maxLogitError = 0;
let maxValueError = 0;
for (let caseIndex = 0; caseIndex < expected.length; caseIndex += 1) {
  if (expected[caseIndex].logits.length !== actual[caseIndex].logits.length) throw new Error(`action count mismatch in case ${caseIndex}`);
  for (let actionIndex = 0; actionIndex < expected[caseIndex].logits.length; actionIndex += 1) {
    maxLogitError = Math.max(maxLogitError, Math.abs(expected[caseIndex].logits[actionIndex] - actual[caseIndex].logits[actionIndex]));
  }
  maxValueError = Math.max(maxValueError, Math.abs(expected[caseIndex].value - actual[caseIndex].value));
}
if (maxLogitError > 2e-4 || maxValueError > 2e-4) {
  throw new Error(`browser/PyTorch parity failed: logits=${maxLogitError}, value=${maxValueError}`);
}
console.log(JSON.stringify({ completedGames, aiDecisions, parityCases: observations.length, maxLogitError, maxValueError, maxInferenceMs }));
