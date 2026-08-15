/**
 * 無頭基準測試：腳本化玩家 vs 破壞巫AI，量化AI強度。
 * 用法：npx tsx bench.ts [games] [fairy|levin]
 */
import {
  attackTargets,
  cardActions,
  createGame,
  definition,
  endTurn,
  evolveCard,
  finishMulligan,
  activateFieldCard,
  playCard,
  resolveChoice,
  type GameState,
  type Zone,
} from "./src/game/engine.ts";

function autoResolve(state: GameState, preferFace = true): GameState {
  let current = state;
  let safety = 0;
  while (current.pending && safety < 60) {
    const pending = current.pending;
    const eligible = pending.options.filter((option) => !option.description?.includes("不符合"));
    if (pending.effect === "attackTarget" && preferFace) {
      const face = eligible.find((option) => option.uid === "ai-leader");
      current = resolveChoice(current, [face?.uid ?? eligible[0]?.uid].filter(Boolean) as string[]);
    } else {
      const count = pending.kind === "order" ? pending.max : Math.max(pending.min, Math.min(1, eligible.length));
      current = resolveChoice(current, eligible.slice(0, count).map((option) => option.uid));
    }
    safety += 1;
  }
  return current;
}

/** 中等強度的妖精腳本玩家：鋪場→進化→全員攻擊（可擊殺就換，否則打臉）。 */
function playerTurn(input: GameState): GameState {
  let state = autoResolve(input);
  let actions = 0;
  while (state.status === "playing" && state.turnSide === "player" && state.phase === "main" && actions < 40) {
    actions += 1;
    // 1) 出牌：可出的裡面選費用最高的
    const playable = [
      ...state.player.hand.map((card) => ({ card, zone: "hand" as Zone })),
      ...state.player.ex.map((card) => ({ card, zone: "ex" as Zone })),
    ]
      .filter(({ card, zone }) => cardActions(state, card.uid, zone).some((action) => action.id === "play" && action.enabled))
      .sort((a, b) => definition(b.card).cost - definition(a.card).cost)[0];
    if (playable) {
      state = autoResolve(playCard(state, playable.card.uid, playable.zone));
      continue;
    }
    // 2) 進化：第一個可進化的
    const evolvable = state.player.field.find((card) =>
      cardActions(state, card.uid, "field").some((action) => action.id === "evolve-ep" && action.enabled));
    if (evolvable) {
      state = autoResolve(evolveCard(state, evolvable.uid, "ep", false));
      continue;
    }
    // 3) 攻擊：所有能攻擊的
    const attacker = state.player.field.find((card) => attackTargets(state, card).length > 0);
    if (attacker) {
      state = autoResolve(activateFieldCard(state, attacker.uid, "attack"));
      continue;
    }
    break;
  }
  if (state.status === "playing" && state.turnSide === "player") state = autoResolve(endTurn(state));
  return state;
}

const games = Number(process.argv[2] ?? 300);
const deck = (process.argv[3] === "levin" ? "levin" : "fairy") as "fairy" | "levin";
let aiWins = 0;
let playerWins = 0;
let draws = 0;
let unfinished = 0;
let totalTurns = 0;
let aiWinTurnSum = 0;
let aiHpSum = 0;

for (let seed = 1; seed <= games; seed += 1) {
  let state = autoResolve(finishMulligan(createGame(seed % 2 === 0, seed, deck), seed % 3 === 0));
  let guard = 0;
  while (state.status === "playing" && guard < 200) {
    guard += 1;
    if (state.turnSide === "player") state = playerTurn(state);
    else state = autoResolve(state);
  }
  totalTurns += state.globalTurn;
  if (state.status !== "gameover") unfinished += 1;
  else if (state.winner === "ai") { aiWins += 1; aiWinTurnSum += state.globalTurn; aiHpSum += state.ai.hp; }
  else if (state.winner === "player") playerWins += 1;
  else draws += 1;
}

console.log(JSON.stringify({
  games,
  deck,
  aiWins,
  playerWins,
  draws,
  unfinished,
  aiWinRate: (aiWins / games * 100).toFixed(1) + "%",
  avgGameTurns: (totalTurns / games).toFixed(1),
  avgAiWinTurn: aiWins ? (aiWinTurnSum / aiWins).toFixed(1) : "-",
  avgAiWinHp: aiWins ? (aiHpSum / aiWins).toFixed(1) : "-",
}, null, 2));
