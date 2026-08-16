import { useEffect, useMemo, useRef, useState } from "react";
import { PLAYER_DECKS, cardName, type Side } from "../game/cards";
import {
  attackOf,
  definition,
  maxHealthOf,
  remainingHealthOf,
  type CardInstance,
  type GameState,
  type MatchEvent,
} from "../game/engine";
import {
  replayTrainingGame,
  trainingLegalActions,
  type TrainingReplay,
  type TrainingReplayDecision,
} from "../game/training";

type ReplayPerspective = "fairy" | "destruction" | "omniscient";

export type LoadedReplaySet = {
  fileName: string;
  policy?: string;
  summary?: Record<string, unknown>;
  entries: { key: string; label: string; replay: TrainingReplay }[];
};

type ReplayBundle = {
  format: "shadowverse-pt-training-replay-bundle";
  policy?: string;
  summary?: Record<string, unknown>;
  replays?: Record<string, TrainingReplay | undefined>;
};

const SAMPLE_LABELS: Record<string, string> = {
  first: "第一局",
  firstPlayerWin: "第一場玩家方勝",
  firstAiWin: "第一場破壞勝",
  showcaseWin: "長局代表勝",
  closeWin: "極限殘血勝",
  cleanWin: "優勢勝",
  firstWin: "先攻代表勝",
  secondWin: "後攻代表勝",
  typicalLoss: "典型敗局",
  randomAudit1: "盲抽稽核 1",
  randomAudit2: "盲抽稽核 2",
  randomAudit3: "盲抽稽核 3",
};

function isTrainingReplay(value: unknown): value is TrainingReplay {
  if (!value || typeof value !== "object") return false;
  const replay = value as Partial<TrainingReplay>;
  return replay.format === "shadowverse-pt-training-replay"
    && replay.replayVersion === 1
    && typeof replay.seed === "number"
    && Array.isArray(replay.decisions);
}

export function parseReplayFile(value: unknown, fileName: string): LoadedReplaySet {
  if (isTrainingReplay(value)) {
    replayTrainingGame(value);
    return { fileName, entries: [{ key: "replay", label: "對局回放", replay: value }] };
  }
  const bundle = value as Partial<ReplayBundle>;
  if (!bundle || bundle.format !== "shadowverse-pt-training-replay-bundle" || !bundle.replays) {
    throw new Error("不是支援的 Shadowverse 訓練回放 JSON");
  }
  const entries = Object.entries(bundle.replays)
    .filter((entry): entry is [string, TrainingReplay] => isTrainingReplay(entry[1]))
    .map(([key, replay]) => {
      replayTrainingGame(replay);
      return { key, label: SAMPLE_LABELS[key] ?? key, replay };
    });
  if (!entries.length) throw new Error("回放包內沒有可播放的對局");
  return { fileName, policy: bundle.policy, summary: bundle.summary, entries };
}

export function ReplayLoadButton({ onLoad }: { onLoad: (source: LoadedReplaySet) => void }) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [error, setError] = useState<string>();
  const openFile = async (file?: File) => {
    if (!file) return;
    try {
      const source = parseReplayFile(JSON.parse(await file.text()), file.name);
      setError(undefined);
      onLoad(source);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "無法讀取回放");
    } finally {
      if (inputRef.current) inputRef.current.value = "";
    }
  };
  return (
    <div className="replay-load-control">
      <button type="button" className="restore-button" onClick={() => inputRef.current?.click()}>載入 AI 對局回放</button>
      <input
        ref={inputRef}
        className="visually-hidden"
        type="file"
        accept="application/json,.json"
        onChange={(event) => void openFile(event.target.files?.[0])}
      />
      {error ? <p className="replay-load-error" role="alert">{error}</p> : null}
    </div>
  );
}

function ReplayCard({ card, compact = false }: { card: CardInstance; compact?: boolean }) {
  const def = definition(card);
  const follower = def.kind === "follower";
  return (
    <article className={`card-tile replay-card ${compact ? "replay-card--compact" : ""} ${card.tapped ? "card-tile--tapped" : ""}`} title={def.text}>
      <span className="card-tile__frame">
        <img src={def.image} alt={def.name} className="card-tile__image" width={240} height={336} />
        {card.baseCardId ? <span className="card-tile__evolved">進化</span> : null}
        {card.tapped ? <span className="card-tile__act">橫置</span> : null}
        {follower ? (
          <span className="card-tile__combat">
            <b>{attackOf(card)}</b>
            <b className={remainingHealthOf(card) < maxHealthOf(card) ? "is-damaged" : ""}>{remainingHealthOf(card)}</b>
          </span>
        ) : null}
      </span>
      <span className="card-tile__name">{def.name}</span>
    </article>
  );
}

function HiddenHand({ count, label }: { count: number; label: string }) {
  return (
    <div className="replay-hidden-hand" aria-label={`${label}${count}張`}>
      <div>{Array.from({ length: Math.min(count, 10) }, (_, index) => <span key={index} className="card-back" />)}</div>
      <b>隱藏手牌 · {count}張</b>
    </div>
  );
}

function ReplayLeaderSummary({ state, side }: { state: GameState; side: Side }) {
  const owner = state[side];
  const evolveCount = Object.values(owner.evolveRemaining).reduce((sum, count) => sum + count, 0);
  return (
    <div className="replay-leader-summary">
      <strong>{side === "ai" ? "破壊ウィッチ" : PLAYER_DECKS[state.playerDeck].label}</strong>
      <span className={owner.hp <= 7 ? "is-danger" : ""}>HP <b>{owner.hp}</b></span>
      <span>PP <b>{owner.pp}/{owner.maxPP}</b></span>
      <span>EP <b>{owner.ep}</b></span>
      <span>SEP <b>{owner.sep}</b></span>
      <span>牌庫 <b>{owner.deck.length}</b></span>
      <span>墓場 <b>{owner.grave.length}</b></span>
      <span>消失 <b>{owner.banished.length}</b></span>
      <span>進化 <b>{evolveCount}</b></span>
    </div>
  );
}

function ReplayCompactZone({ label, cards, hidden, side, state, leader = false }: { label: string; cards: CardInstance[]; hidden?: boolean; side: Side; state: GameState; leader?: boolean }) {
  return (
    <section className={`replay-compact-zone replay-compact-zone--${side}`}>
      <div className="replay-zone-heading">
        <strong>{label}</strong>
        <small>{hidden ? `${cards.length}張 · 隱藏` : `${cards.length}張`}</small>
      </div>
      <div className="replay-zone-cards">
        {hidden
          ? <HiddenHand count={cards.length} label={label} />
          : cards.length ? cards.map((card) => <ReplayCard key={card.uid} card={card} compact />) : <span className="replay-zone-empty">空</span>}
      </div>
      {leader ? <ReplayLeaderSummary state={state} side={side} /> : null}
    </section>
  );
}

function ReplayBattleBoard({ state, revealPlayer, revealAi }: { state: GameState; revealPlayer: boolean; revealAi: boolean }) {
  const playerLabel = state.playerDeck === "levin" ? "雷維翁" : state.playerDeck === "sekka" ? "雪華獸" : "妖精";
  return (
    <div className="replay-battle-board">
      <ReplayCompactZone label="破壞手牌" cards={state.ai.hand} hidden={!revealAi} side="ai" state={state} leader />
      <ReplayCompactZone label="破壞 EX" cards={state.ai.ex} side="ai" state={state} />
      <ReplayCompactZone label="破壞場上" cards={state.ai.field} side="ai" state={state} />
      <div className="replay-center-line"><span /><strong>{state.lastAction ?? "等待行動"}</strong><span /></div>
      <ReplayCompactZone label={`${playerLabel}場上`} cards={state.player.field} side="player" state={state} />
      <ReplayCompactZone label={`${playerLabel} EX`} cards={state.player.ex} side="player" state={state} />
      <ReplayCompactZone label={`${playerLabel}手牌`} cards={state.player.hand} hidden={!revealPlayer} side="player" state={state} leader />
    </div>
  );
}

function eventLabel(event: MatchEvent, perspective: ReplayPerspective, playerLabel: string): string {
  const owner = event.side === "player" ? playerLabel : event.side === "ai" ? "破壞" : "系統";
  const card = event.cardId ? ` · ${cardName(event.cardId)}` : "";
  const detail = event.side === "ai" && perspective === "fairy"
    ? event.detail?.replace(/\s+cards=.*/, "")
    : event.detail;
  return `${owner}｜${event.type}${card}${detail ? `｜${detail}` : ""}`;
}

function DecisionPanel({ state, decision, events, perspective }: { state: GameState; decision?: TrainingReplayDecision; events: MatchEvent[]; perspective: ReplayPerspective }) {
  const viewerSide = perspective === "fairy" ? "player" : perspective === "destruction" ? "ai" : undefined;
  const canInspectDecision = !decision || viewerSide === undefined || decision.actor === viewerSide;
  const actions = canInspectDecision ? trainingLegalActions(state) : [];
  const probabilities = new Map(decision?.audit?.policy?.map((item) => [item.actionKey, item.probability]));
  return (
    <aside className="replay-audit-panel">
      <section>
        <p className="eyebrow">本格事件</p>
        <h2>雙方操作</h2>
        {events.length ? <ol>{events.map((event) => <li key={event.seq}><span>{event.seq}</span><p>{eventLabel(event, perspective, state.playerDeck === "levin" ? "雷維翁" : state.playerDeck === "sekka" ? "雪華獸" : "妖精")}</p></li>)}</ol> : <p className="replay-muted">本格沒有新增事件。</p>}
      </section>
      <section>
        <p className="eyebrow">下一個決策</p>
        <h2>{decision && !canInspectDecision ? "對手正在決策" : decision?.actionLabel ?? (state.status === "gameover" ? "對局已結束" : "沒有待執行決策")}</h2>
        {decision && !canInspectDecision ? <p className="replay-muted">此視角不能查看對手的合法操作、策略機率或 Value；切換裁判全知視角可檢查。</p> : null}
        {canInspectDecision && decision?.audit?.value !== undefined ? <p className="replay-value">模型 Value：{decision.audit.value.toFixed(4)}</p> : null}
        {canInspectDecision && decision?.audit?.note ? <p className="replay-muted">策略：{decision.audit.note}</p> : null}
        {actions.length ? (
          <ul className="replay-legal-actions">
            {actions.map((action) => {
              const selected = action.key === decision?.actionKey;
              const probability = probabilities.get(action.key);
              return <li key={action.key} className={selected ? "is-selected" : ""}><span>{selected ? "選擇" : "合法"}</span><p>{action.label}</p>{probability !== undefined ? <b>{(probability * 100).toFixed(1)}%</b> : null}</li>;
            })}
          </ul>
        ) : null}
      </section>
    </aside>
  );
}

export default function ReplayViewer({ source, onClose }: { source: LoadedReplaySet; onClose: () => void }) {
  const [entryKey, setEntryKey] = useState(source.entries[0].key);
  const entry = source.entries.find((item) => item.key === entryKey) ?? source.entries[0];
  const reconstructed = useMemo(() => replayTrainingGame(entry.replay), [entry]);
  const [step, setStep] = useState(0);
  const [perspective, setPerspective] = useState<ReplayPerspective>("fairy");
  const [playing, setPlaying] = useState(false);
  const maxStep = reconstructed.states.length - 1;
  const state = reconstructed.states[Math.min(step, maxStep)];
  const previousState = step > 0 ? reconstructed.states[step - 1] : undefined;
  const stepEvents = state.events.slice(previousState?.events.length ?? 0);
  const nextDecision = entry.replay.decisions[step];

  useEffect(() => {
    setStep(0);
    setPlaying(false);
  }, [entryKey]);

  useEffect(() => {
    if (!playing) return;
    const timer = window.setInterval(() => {
      setStep((current) => {
        if (current >= maxStep) {
          setPlaying(false);
          return current;
        }
        return current + 1;
      });
    }, 900);
    return () => window.clearInterval(timer);
  }, [playing, maxStep]);

  useEffect(() => {
    const handleKey = (event: KeyboardEvent) => {
      if (event.key === "ArrowLeft") setStep((current) => Math.max(0, current - 1));
      if (event.key === "ArrowRight") setStep((current) => Math.min(maxStep, current + 1));
      if (event.key === " ") {
        event.preventDefault();
        setPlaying((current) => !current);
      }
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [maxStep]);

  const revealPlayer = perspective === "fairy" || perspective === "omniscient";
  const revealAi = perspective === "destruction" || perspective === "omniscient";
  const lastAction = step > 0 ? entry.replay.decisions[step - 1] : undefined;
  const playerLabel = state.playerDeck === "levin" ? "雷維翁" : state.playerDeck === "sekka" ? "雪華獸" : "妖精";
  const resultLabel = state.status === "mulligan" ? "換牌階段" : state.status === "gameover"
    ? state.winner === "player" ? `${playerLabel}勝利` : state.winner === "ai" ? "破壞勝利" : "平手"
    : `${state.turnSide === "player" ? playerLabel : "破壞"}第${state[state.turnSide].ownTurn}回合`;

  return (
    <main className="replay-shell">
      <header className="replay-header">
        <div><p className="eyebrow">AI 對局回放 · seed {entry.replay.seed}</p><h1>{entry.label}｜{resultLabel}</h1></div>
        <div className="replay-header-controls">
          {source.entries.length > 1 ? <label>樣本<select value={entryKey} onChange={(event) => setEntryKey(event.target.value)}>{source.entries.map((item) => <option key={item.key} value={item.key}>{item.label}</option>)}</select></label> : null}
          <label>視角<select value={perspective} onChange={(event) => setPerspective(event.target.value as ReplayPerspective)}><option value="fairy">{playerLabel}視角</option><option value="destruction">破壞視角</option><option value="omniscient">裁判全知</option></select></label>
          <button type="button" onClick={onClose}>離開回放</button>
        </div>
      </header>

      <section className="replay-transport" aria-label="回放控制">
        <button type="button" onClick={() => setStep(0)} disabled={step === 0}>第一格</button>
        <button type="button" onClick={() => setStep((current) => Math.max(0, current - 1))} disabled={step === 0}>上一步</button>
        <button type="button" className="replay-play-button" onClick={() => setPlaying((current) => !current)}>{playing ? "暫停" : "播放"}</button>
        <button type="button" onClick={() => setStep((current) => Math.min(maxStep, current + 1))} disabled={step === maxStep}>下一步</button>
        <button type="button" onClick={() => setStep(maxStep)} disabled={step === maxStep}>最後一格</button>
        <label><span>{step} / {maxStep}</span><input type="range" min={0} max={maxStep} value={step} onChange={(event) => { setPlaying(false); setStep(Number(event.target.value)); }} /></label>
      </section>

      <div className="replay-notice">
        <strong>{lastAction ? `剛執行：${lastAction.actionLabel}` : "起始換牌局面"}</strong>
        <span>雙方模型的留牌、出牌、攻擊、進化、啟動、Quick 與效果選擇都會逐手呈現；切換裁判全知視角可核對完整機率。</span>
      </div>

      <section className="replay-layout">
        <div className="replay-board-column">
          <ReplayBattleBoard state={state} revealAi={revealAi} revealPlayer={revealPlayer} />
        </div>
        <DecisionPanel state={state} decision={nextDecision} events={stepEvents} perspective={perspective} />
      </section>
    </main>
  );
}
