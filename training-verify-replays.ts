import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { replayTrainingGame, type TrainingReplay } from "./src/game/training.ts";

type ReplayBundle = {
  format: string;
  replays?: Record<string, TrainingReplay>;
};

for (const input of process.argv.slice(2)) {
  const path = resolve(input);
  const bundle = JSON.parse(readFileSync(path, "utf8")) as ReplayBundle;
  if (bundle.format !== "shadowverse-pt-training-replay-bundle" || !bundle.replays) {
    throw new Error(`${path} is not a replay bundle`);
  }
  let decisions = 0;
  for (const [name, replay] of Object.entries(bundle.replays)) {
    const reconstructed = replayTrainingGame(replay);
    const finalState = reconstructed.states.at(-1);
    if (!finalState || finalState.status !== "gameover" || finalState.winner !== replay.result?.winner) {
      throw new Error(`${path}:${name} reconstructed to a different result`);
    }
    decisions += replay.decisions.length;
  }
  process.stdout.write(`${JSON.stringify({ path, replays: Object.keys(bundle.replays).length, decisions, verified: true })}\n`);
}
