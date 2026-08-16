/**
 * 訓練環境基準：貪心玩家策略 vs 內建破壞 AI。
 *
 * 用法：
 *   npm run bench:training -- [games] [replay-output.json] [fairy|levin] [base-seed] [scripted|manual]
 *
 * 輸出的回放包會保留第一局、第一場妖精勝利與第一場破壞勝利，方便人工檢查雙方操作。
 */
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { performance } from "node:perf_hooks";
import { createGame, definition, type GameState } from "./src/game/engine.ts";
import {
  applyTrainingAction,
  createTrainingReplay,
  finalizeTrainingReplay,
  recordTrainingDecision,
  replayTrainingGame,
  trainingLegalActions,
  type TrainingAction,
  type TrainingReplay,
} from "./src/game/training.ts";

function greedyPlayerAction(state: GameState, actions: TrainingAction[]): TrainingAction {
  const choices = actions.filter((action): action is Extract<TrainingAction, { kind: "choice" }> => action.kind === "choice");
  if (choices.length) {
    if (state.pending?.effect === "attackTarget") {
      const face = choices.find((action) => action.selected.includes("ai-leader"));
      if (face) return face;
    }
    return choices
      .filter((action) => action.selected.length > 0)
      .sort((a, b) => b.selected.length - a.selected.length)[0] ?? choices[0];
  }

  const play = actions
    .filter((action): action is Extract<TrainingAction, { kind: "play" }> => action.kind === "play")
    .sort((a, b) => definition(b.cardId).cost - definition(a.cardId).cost)[0];
  if (play) return play;
  const evolve = actions.find((action) => action.kind === "evolve");
  if (evolve) return evolve;
  const attack = actions.find((action) => action.kind === "attack" && action.targetUid.endsWith("-leader"))
    ?? actions.find((action) => action.kind === "attack");
  if (attack) return attack;
  const activate = actions.find((action) => action.kind === "activate");
  if (activate) return activate;
  const keep = actions.find((action) => action.kind === "mulligan" && !action.redraw);
  if (keep) return keep;
  const end = actions.find((action) => action.kind === "end");
  if (end) return end;
  throw new Error("No legal action returned by the training environment");
}

const games = Math.max(1, Number(process.argv[2] ?? 500));
const replayOutput = process.argv[3] && process.argv[3] !== "-" ? process.argv[3] : undefined;
const deck = process.argv[4] === "levin" ? "levin" : "fairy";
const baseSeed = Math.max(0, Number(process.argv[5] ?? 1));
const aiControl = process.argv[6] === "manual" ? "manual" : "scripted";
let playerWins = 0;
let aiWins = 0;
let draws = 0;
let unfinished = 0;
let totalDecisions = 0;
let totalGlobalTurns = 0;
let firstReplay: TrainingReplay | undefined;
let firstPlayerWin: TrainingReplay | undefined;
let firstAiWin: TrainingReplay | undefined;
const startedAt = performance.now();

for (let index = 0; index < games; index += 1) {
  const seed = baseSeed + index;
  let state = createGame(seed % 2 === 0, seed, deck, { aiControl });
  let replay = createTrainingReplay(state);
  let decisions = 0;
  while (state.status !== "gameover" && decisions < 500) {
    const legal = trainingLegalActions(state);
    const action = greedyPlayerAction(state, legal);
    replay = recordTrainingDecision(replay, state, action, { note: `greedy-${deck}-baseline-v1` });
    state = applyTrainingAction(state, action);
    decisions += 1;
  }
  replay = finalizeTrainingReplay(replay, state);
  firstReplay ??= replay;
  if (state.status !== "gameover") unfinished += 1;
  else if (state.winner === "player") {
    playerWins += 1;
    firstPlayerWin ??= replay;
  } else if (state.winner === "ai") {
    aiWins += 1;
    firstAiWin ??= replay;
  } else draws += 1;
  totalDecisions += decisions;
  totalGlobalTurns += state.globalTurn;
}

const elapsedSeconds = (performance.now() - startedAt) / 1000;
const summary = {
  deck,
  aiControl,
  seedRange: [baseSeed, baseSeed + games - 1],
  games,
  playerWins,
  aiWins,
  draws,
  unfinished,
  playerWinRate: `${(playerWins / games * 100).toFixed(1)}%`,
  aiWinRate: `${(aiWins / games * 100).toFixed(1)}%`,
  totalDecisions,
  averageGlobalTurns: Number((totalGlobalTurns / games).toFixed(2)),
  elapsedSeconds: Number(elapsedSeconds.toFixed(3)),
  gamesPerSecond: Number((games / elapsedSeconds).toFixed(2)),
  decisionsPerSecond: Number((totalDecisions / elapsedSeconds).toFixed(2)),
};

console.log(JSON.stringify(summary, null, 2));

if (replayOutput) {
  const retainedReplays = { first: firstReplay, firstPlayerWin, firstAiWin };
  const verifiedReplays = Object.entries(retainedReplays)
    .filter((entry): entry is [string, TrainingReplay] => Boolean(entry[1]))
    .map(([name, replay]) => {
      replayTrainingGame(replay);
      return name;
    });
  const outputPath = resolve(replayOutput);
  mkdirSync(dirname(outputPath), { recursive: true });
  writeFileSync(outputPath, `${JSON.stringify({
    format: "shadowverse-pt-training-replay-bundle",
    replayVersion: 1,
    policy: `greedy-${deck}-baseline-v1`,
    summary,
    verifiedReplays,
    replays: retainedReplays,
  }, null, 2)}\n`, "utf8");
  console.log(`Replay bundle: ${outputPath}`);
}
