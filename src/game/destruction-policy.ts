import * as ort from "onnxruntime-web/wasm";
import { TRAINING_CARD_IDS, TRAINING_ENCODING_METADATA } from "./training-encoding";
import { buildBrowserPolicyInputs } from "./browser-policy-input";
import { GAME_ENGINE_VERSION, type GameState } from "./engine";
import { trainingActor, trainingLegalActions, type TrainingAction } from "./training";

type PolicyManifest = {
  format: "shadowverse-pt-browser-policy";
  formatVersion: number;
  policy: string;
  cycle: number;
  engineVersion: number;
  selfPlay: boolean;
  model: string;
  sha256: string;
  bytes: number;
  cardIds: string[];
  metadata: {
    cardVocabularySize: number;
    zoneCount: number;
    scalarSize: number;
    fieldSlots: number;
    fieldNumberSize: number;
    recentEventSlots: number;
    recentEventNumberSize: number;
    actionKindCount: number;
    abilityBucketCount: number;
    selectionSlots: number;
    actionNumberSize: number;
  };
};

export type DestructionPolicyDecision = {
  action: TrainingAction;
  value: number;
  logits: number[];
  inferenceMs: number;
};

export type DestructionPolicy = {
  manifest: PolicyManifest;
  choose(state: GameState): Promise<DestructionPolicyDecision>;
};

const MANIFEST_PATH = "models/destruction-cycle13.json";
let policyPromise: Promise<DestructionPolicy> | undefined;

function sameArray(left: readonly string[], right: readonly string[]): boolean {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

function assertManifest(manifest: PolicyManifest): void {
  const metadata = TRAINING_ENCODING_METADATA;
  const expected = {
    cardVocabularySize: metadata.cardVocabularySize,
    zoneCount: metadata.zoneNames.length,
    scalarSize: metadata.scalarSize,
    fieldSlots: metadata.fieldSlots,
    fieldNumberSize: metadata.fieldNumberSize,
    recentEventSlots: metadata.recentEventSlots,
    recentEventNumberSize: metadata.recentEventNumberSize,
    actionKindCount: metadata.actionKindCount,
    abilityBucketCount: metadata.abilityBucketCount,
    selectionSlots: metadata.actionSelectionSlots,
    actionNumberSize: metadata.actionNumberSize,
  };
  if (manifest.format !== "shadowverse-pt-browser-policy" || manifest.formatVersion !== 1) {
    throw new Error("不支援的模型 manifest 格式");
  }
  if (manifest.policy !== "destruction" || manifest.cycle !== 13 || !manifest.selfPlay) {
    throw new Error("部署的不是 cycle 13 自我對抗破壞模型");
  }
  if (manifest.engineVersion !== GAME_ENGINE_VERSION) {
    throw new Error(`模型只支援引擎 v${manifest.engineVersion}，目前是 v${GAME_ENGINE_VERSION}`);
  }
  if (!sameArray(manifest.cardIds, TRAINING_CARD_IDS)) throw new Error("模型卡牌字典與網頁不一致");
  for (const [key, value] of Object.entries(expected)) {
    if (manifest.metadata[key as keyof typeof expected] !== value) throw new Error(`模型編碼欄位不一致：${key}`);
  }
}

async function sha256(buffer: ArrayBuffer): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", buffer);
  return Array.from(new Uint8Array(digest), (value) => value.toString(16).padStart(2, "0")).join("");
}

async function createPolicy(): Promise<DestructionPolicy> {
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy = false;
  const manifestUrl = new URL(`${import.meta.env.BASE_URL}${MANIFEST_PATH}`, window.location.origin);
  const manifestResponse = await fetch(manifestUrl);
  if (!manifestResponse.ok) throw new Error(`模型資訊讀取失敗：HTTP ${manifestResponse.status}`);
  const manifest = await manifestResponse.json() as PolicyManifest;
  assertManifest(manifest);

  const modelUrl = new URL(manifest.model, manifestUrl);
  const modelResponse = await fetch(modelUrl);
  if (!modelResponse.ok) throw new Error(`模型讀取失敗：HTTP ${modelResponse.status}`);
  const modelBuffer = await modelResponse.arrayBuffer();
  if (modelBuffer.byteLength !== manifest.bytes) throw new Error("模型檔案長度與 manifest 不符");
  if (await sha256(modelBuffer) !== manifest.sha256) throw new Error("模型 SHA-256 驗證失敗");

  const session = await ort.InferenceSession.create(new Uint8Array(modelBuffer), {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
  return {
    manifest,
    async choose(state: GameState): Promise<DestructionPolicyDecision> {
      if (trainingActor(state) !== "ai") throw new Error("目前不是破壞模型的決策時點");
      const actions = trainingLegalActions(state);
      if (!actions.length) throw new Error("破壞模型沒有合法動作");
      const inputs = buildBrowserPolicyInputs(state, actions);
      const feeds: Record<string, ort.Tensor> = {};
      for (const [name, input] of Object.entries(inputs)) {
        feeds[name] = new ort.Tensor(input.type, input.data, input.dims);
      }
      const started = performance.now();
      const output = await session.run(feeds);
      const inferenceMs = performance.now() - started;
      const logits = Array.from(output.logits.data as Float32Array);
      const value = Number((output.value.data as Float32Array)[0]);
      let bestIndex = 0;
      for (let index = 1; index < logits.length; index += 1) {
        if (logits[index] > logits[bestIndex]) bestIndex = index;
      }
      return { action: actions[bestIndex], value, logits, inferenceMs };
    },
  };
}

export function loadDestructionPolicy(): Promise<DestructionPolicy> {
  policyPromise ??= createPolicy().catch((reason: unknown) => {
    policyPromise = undefined;
    throw reason;
  });
  return policyPromise;
}
