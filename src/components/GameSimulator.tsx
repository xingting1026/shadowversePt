import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { PLAYER_DECKS, type PlayerDeckId, type Side } from "../game/cards";
import {
  activateFieldCard,
  attackOf,
  cardActions,
  createGame,
  definition,
  endTurn,
  evolveCard,
  finishManualMulligan,
  maxHealthOf,
  playCard,
  remainingHealthOf,
  resolveChoice,
  restartWithSameSeed,
  type CardInstance,
  type GameState,
  type PendingChoice,
  type Zone,
} from "../game/engine";
import { loadDestructionPolicy, type DestructionPolicy } from "../game/destruction-policy";
import { applyTrainingAction, trainingActor } from "../game/training";
import ReplayViewer, { parseReplayFile, ReplayLoadButton, type LoadedReplaySet } from "./ReplayViewer";

type SelectedCard = { uid: string; zone: Zone; side: Side };
type ZoneDrawer = { title: string; cards: CardInstance[]; side: Side; zone: Zone; note?: string };

const KIND_LABEL = { follower: "從者", spell: "法術", amulet: "護符", token: "衍生卡" } as const;
const ZONE_LABEL: Record<Zone, string> = { deck: "牌庫", hand: "手牌", field: "場上", ex: "EX區", grave: "墓場", banished: "消失領域" };

function locateCard(state: GameState, selection: SelectedCard): CardInstance | undefined {
  return state[selection.side][selection.zone].find((item) => item.uid === selection.uid);
}

function StatPill({ children, tone = "neutral" }: { children: ReactNode; tone?: "neutral" | "danger" | "good" | "gold" }) {
  return <span className={`stat-pill stat-pill--${tone}`}>{children}</span>;
}

function MiniCard({
  card,
  onClick,
  compact = false,
  selected = false,
  order,
  disabled = false,
}: {
  card: CardInstance;
  onClick?: () => void;
  compact?: boolean;
  selected?: boolean;
  order?: number;
  disabled?: boolean;
}) {
  const def = definition(card);
  const follower = def.kind === "follower";
  return (
    <button
      type="button"
      className={`card-tile ${compact ? "card-tile--compact" : ""} ${card.tapped ? "card-tile--tapped" : ""} ${selected ? "card-tile--selected" : ""} ${disabled ? "card-tile--disabled" : ""}`}
      onClick={onClick}
      disabled={disabled}
      aria-label={`查看${def.name}`}
    >
      <span className="card-tile__frame">
        {/* Card artwork is locally cached from the official card list. */}
        <img src={def.image} alt={def.name} className="card-tile__image" width={240} height={336} />
        {order ? <span className="card-tile__order">{order}</span> : null}
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
    </button>
  );
}

function CardRow({
  cards,
  side,
  zone,
  onSelect,
  empty,
}: {
  cards: CardInstance[];
  side: Side;
  zone: Zone;
  onSelect: (selection: SelectedCard) => void;
  empty: string;
}) {
  if (!cards.length) return <div className="empty-zone">{empty}</div>;
  return (
    <div className="card-row">
      {cards.map((card) => (
        <MiniCard key={card.uid} card={card} onClick={() => onSelect({ uid: card.uid, side, zone })} />
      ))}
    </div>
  );
}

function PlayerVitals({ state, side }: { state: GameState; side: Side }) {
  const player = state[side];
  return (
    <div className={`vitals ${side === "ai" ? "vitals--ai" : "vitals--player"}`}>
      <div className="vitals__identity">
        <span className="vitals__eyebrow">{side === "ai" ? "cycle 13 對抗模型" : `玩家 · ${state.playerDeck === "levin" ? "1KUUZE" : state.playerDeck === "sekka" ? "D9Q1A" : "7P9XP"}`}</span>
        <strong>{side === "ai" ? "破壊ウィッチ" : PLAYER_DECKS[state.playerDeck].label}</strong>
      </div>
      <div className="vitals__stats">
        <StatPill tone={player.hp <= 7 ? "danger" : "good"}>HP {player.hp}</StatPill>
        <StatPill tone="gold">PP {player.pp}/{player.maxPP}</StatPill>
        <StatPill>EP {player.ep}</StatPill>
        <StatPill>SEP {player.sep}</StatPill>
      </div>
    </div>
  );
}

function ZoneButtons({
  state,
  side,
  openDrawer,
}: {
  state: GameState;
  side: Side;
  openDrawer: (drawer: ZoneDrawer) => void;
}) {
  const player = state[side];
  const evolveCards = Object.entries(player.evolveRemaining).flatMap(([cardId, count]) =>
    Array.from({ length: count }, (_, index) => ({
      uid: `evolve-${side}-${cardId}-${index}`,
      cardId,
      owner: side,
      zone: "ex" as Zone,
      tapped: false,
      damage: 0,
      attackBuff: 0,
      healthBuff: 0,
      enteredAt: -1,
      evolvedThisTurn: false,
      tempStorm: false,
      tempRush: false,
      tempDesignated: false,
      flags: {},
    })),
  );
  return (
    <div className="zone-buttons">
      <button type="button" className="zone-chip zone-chip--deck" disabled>
        <span>牌庫</span><b>{player.deck.length}</b>
      </button>
      <button type="button" className="zone-chip" onClick={() => openDrawer({ title: `${side === "ai" ? "破壞巫" : "你的"}墓場`, cards: player.grave, side, zone: "grave", note: "衍生卡離場後不會留在墓場。" })}>
        <span>墓場</span><b>{player.grave.length}</b>
      </button>
      <button type="button" className="zone-chip" onClick={() => openDrawer({ title: `${side === "ai" ? "破壞巫" : "你的"}消失領域`, cards: player.banished, side, zone: "banished" })}>
        <span>消失</span><b>{player.banished.length}</b>
      </button>
      <button type="button" className="zone-chip" onClick={() => openDrawer({ title: `${side === "ai" ? "破壞巫" : "你的"}進化區（可用）`, cards: evolveCards, side, zone: "ex", note: `已使用 ${Object.values(player.evolveUsed).reduce((a, b) => a + b, 0)} 張` })}>
        <span>進化區</span><b>{evolveCards.length}</b>
      </button>
    </div>
  );
}

function HandBacks({ count }: { count: number }) {
  return (
    <div className="opponent-hand" aria-label={`破壞巫手牌${count}張`}>
      {Array.from({ length: Math.min(count, 9) }, (_, index) => <span key={index} className="card-back" style={{ transform: `translateX(${index * -7}px) rotate(${(index - count / 2) * 1.4}deg)` }} />)}
      <b>{count}張</b>
    </div>
  );
}

function SetupScreen({
  onStart,
  onReplay,
  policy,
  policyError,
  onRetry,
}: {
  onStart: (first: boolean, seed: number, deck: PlayerDeckId) => void;
  onReplay: (source: LoadedReplaySet) => void;
  policy: DestructionPolicy | null;
  policyError: string | null;
  onRetry: () => void;
}) {
  const [seed, setSeed] = useState(() => Math.floor(Date.now() % 2_147_483_647));
  const [deck, setDeck] = useState<PlayerDeckId>("fairy");
  return (
    <main className="setup-shell">
      <section className="setup-card">
        <div className="setup-card__mark">{deck === "levin" ? "L / W" : deck === "sekka" ? "S / W" : "F / W"}</div>
        <p className="eyebrow">SHADOWVERSE EVOLVE · 對局練習室</p>
        <h1>{deck === "levin" ? "雷維翁 vs. 破壊" : deck === "sekka" ? "雪華獸 vs. 破壊" : "妖精 vs. 破壊"}</h1>
        <p className="setup-card__lead">
          {deck === "levin"
            ? "1KUUZE雷維翁皇家牌組：靠棄牌與磨牌把雷維翁堆進墓場，5張開啟全隊強化，アルベール一回合多段疾走收尾。對手是52人賽冠軍破壞巫。"
            : deck === "sekka"
              ? "D9Q1A雪華獸エルフ牌組：回手與再展開累積資源，狐火從墓場點燃九火石炎・セッカ，配合セタス與緑の顕現一波鋪場。對手是52人賽冠軍破壞巫。"
              : "固定7P9XP妖精牌組，對戰52人賽冠軍5JK33破壞巫。你操作妖精，對手由妖精與雷維翁共同對抗訓練的破壞模型操作。"}
        </p>
        <div className={`model-status ${policyError ? "is-error" : policy ? "is-ready" : "is-loading"}`} role="status">
          <span className="model-status__dot" />
          <div>
            <strong>{policyError ? "模型載入失敗" : policy ? `破壞模型 cycle ${policy.manifest.cycle} 已就緒` : "正在載入破壞模型…"}</strong>
            <small>{policyError ?? (policy ? "engine v3 · 自我對抗聯賽 · 瀏覽器本機推論" : "首次開啟需下載約 4 MB 模型")}</small>
          </div>
          {policyError ? <button type="button" onClick={onRetry}>重試</button> : null}
        </div>
        <div className="setup-actions" role="radiogroup" aria-label="選擇牌組">
          <button
            type="button"
            className={deck === "fairy" ? "primary-button" : "secondary-button"}
            aria-pressed={deck === "fairy"}
            onClick={() => setDeck("fairy")}
          >妖精エルフ</button>
          <button
            type="button"
            className={deck === "levin" ? "primary-button" : "secondary-button"}
            aria-pressed={deck === "levin"}
            onClick={() => setDeck("levin")}
          >レヴィオンロイヤル</button>
          <button
            type="button"
            className={deck === "sekka" ? "primary-button" : "secondary-button"}
            aria-pressed={deck === "sekka"}
            onClick={() => setDeck("sekka")}
          >雪華獸（セッカ）</button>
        </div>
        <label className="seed-field">
          <span>亂數種子</span>
          <input type="number" value={seed} onChange={(event) => setSeed(Number(event.target.value) || 1)} />
        </label>
        <div className="setup-actions">
          <button type="button" className="primary-button" disabled={!policy} onClick={() => onStart(true, seed, deck)}>我方先攻</button>
          <button type="button" className="secondary-button" disabled={!policy} onClick={() => onStart(false, seed, deck)}>我方後攻</button>
        </div>
        {import.meta.env.DEV ? <ReplayLoadButton onLoad={onReplay} /> : null}
        <div className="setup-card__facts">
          <span>完整卡圖與效果詳情</span><span>cycle 13 模型本機推論</span><span>相同亂數種子可驗證</span>
        </div>
      </section>
    </main>
  );
}

function MulliganScreen({ state, onKeep, onRedraw, onInspect }: { state: GameState; onKeep: () => void; onRedraw: () => void; onInspect: (selection: SelectedCard) => void }) {
  return (
    <main className="mulligan-shell">
      <section className="mulligan-panel">
        <p className="eyebrow">起手 · {state.playerFirst ? "先攻" : "後攻"}</p>
        <h1>起手要保留嗎？</h1>
        <p>只能選擇整手保留，或將4張全部放到牌庫底後重抽一次。點卡片可先查看完整效果。</p>
        <div className="mulligan-cards">
          {state.player.hand.map((card) => <MiniCard key={card.uid} card={card} onClick={() => onInspect({ uid: card.uid, side: "player", zone: "hand" })} />)}
        </div>
        <div className="setup-actions">
          <button type="button" className="primary-button" onClick={onKeep}>保留起手</button>
          <button type="button" className="secondary-button" onClick={onRedraw}>全部重抽</button>
        </div>
      </section>
    </main>
  );
}

function AiMulliganScreen({ error }: { error: string | null }) {
  return (
    <main className="mulligan-shell">
      <section className="mulligan-panel mulligan-panel--thinking" role="status">
        <p className="eyebrow">破壞模型 · 起手決策</p>
        <h1>{error ? "模型暫停" : "破壞巫正在判斷留牌…"}</h1>
        <p>{error ?? "模型只看到自己的手牌與公開資訊，不會讀取你的手牌。"}</p>
        {!error ? <span className="thinking-pulse" /> : null}
      </section>
    </main>
  );
}

function PendingModal({ pending, onResolve }: { pending: PendingChoice; onResolve: (uids: string[]) => void }) {
  const [selected, setSelected] = useState<string[]>([]);
  const [inspected, setInspected] = useState<string | null>(null);
  const inspectedOption = pending.options.find((option) => option.uid === inspected);
  const inspectedDef = inspectedOption?.cardId ? definition(inspectedOption.cardId) : undefined;
  const toggle = (uid: string, disabled: boolean) => {
    if (disabled) return;
    setSelected((current) => {
      if (current.includes(uid)) return current.filter((item) => item !== uid);
      if (pending.kind === "single" || pending.kind === "yesNo" || pending.kind === "triggerOrder") return [uid];
      if (current.length >= pending.max) return current;
      return [...current, uid];
    });
  };
  const valid = selected.length >= pending.min && selected.length <= pending.max;
  return (
    <div className="modal-backdrop" role="dialog" aria-modal="true" aria-label={pending.title}>
      <section className="choice-modal">
        <div className="modal-heading">
          <p className="eyebrow">效果結算</p>
          <h2>{pending.title}</h2>
          <p>{pending.prompt}</p>
        </div>
        {inspectedOption && inspectedDef ? (
          <section className="choice-inspector" aria-live="polite">
            <div>
              <p className="eyebrow">牌效確認 · {KIND_LABEL[inspectedDef.kind]}</p>
              <h3>{inspectedDef.name}</h3>
              <p>{inspectedDef.text}</p>
            </div>
            <div className="choice-inspector__actions">
              <span>費用 {inspectedDef.cost}{inspectedDef.kind === "follower" ? ` · ${inspectedDef.attack}/${inspectedDef.health}` : ""}</span>
              <button
                type="button"
                disabled={Boolean(inspectedOption.description?.includes("不符合"))}
                onClick={() => toggle(inspectedOption.uid, Boolean(inspectedOption.description?.includes("不符合")))}
              >
                {selected.includes(inspectedOption.uid) ? "取消選擇" : pending.kind === "order" ? "排入下一順位" : "選擇這張"}
              </button>
            </div>
          </section>
        ) : pending.options.some((option) => option.cardId) ? <p className="choice-inspect-hint">先點擊卡片查看完整牌效，再決定是否選擇。</p> : null}
        <div className={`choice-grid ${pending.kind === "order" ? "choice-grid--order" : ""}`}>
          {pending.options.map((option) => {
            const index = selected.indexOf(option.uid);
            const fake: CardInstance | undefined = option.cardId ? {
              uid: option.uid, cardId: option.cardId, owner: "player", zone: "ex", tapped: false,
              damage: 0, attackBuff: 0, healthBuff: 0, enteredAt: -1, evolvedThisTurn: false,
              tempStorm: false, tempRush: false, tempDesignated: false, flags: {},
            } : undefined;
            const disabled = Boolean(option.description?.includes("不符合"));
            return fake ? (
              <div key={option.uid} className={`choice-card-wrap ${disabled ? "is-ineligible" : ""}`}>
                <MiniCard card={fake} compact selected={index >= 0} order={index >= 0 ? index + 1 : undefined} onClick={() => setInspected(option.uid)} />
                {option.description ? <small>{option.description}</small> : null}
              </div>
            ) : (
              <button key={option.uid} type="button" disabled={disabled} className={`text-choice ${index >= 0 ? "is-selected" : ""}`} onClick={() => toggle(option.uid, disabled)}>
                <strong>{option.label ?? option.uid}</strong>
                {option.description ? <span>{option.description}</span> : null}
              </button>
            );
          })}
        </div>
        <div className="modal-actions">
          <span>{pending.kind === "order" ? `已排列 ${selected.length}/${pending.max}` : `已選 ${selected.length} · 需要 ${pending.min}–${pending.max}`}</span>
          <button type="button" className="primary-button" disabled={!valid} onClick={() => { const answer = selected; setSelected([]); onResolve(answer); }}>確認</button>
        </div>
      </section>
    </div>
  );
}

function CardDetail({ state, selection, onClose, setState }: { state: GameState; selection: SelectedCard; onClose: () => void; setState: (state: GameState) => void }) {
  const card = locateCard(state, selection);
  if (!card) return null;
  const def = definition(card);
  const actions = cardActions(state, card.uid, selection.zone, selection.side);
  const visibleActions = actions.filter((action) => action.enabled || action.id !== "play");
  const perform = (id: string) => {
    let next = state;
    if (id === "play") next = playCard(state, card.uid, selection.zone);
    else if (id === "attack" || ["amatsuStorm", "bouquetBounce", "gardenDamage", "wonderDraw", "wingDestroy", "dukePing", "archerSnipe", "albertRestand"].includes(id)) next = activateFieldCard(state, card.uid, id);
    else {
      const action = actions.find((item) => item.id === id);
      if (action?.payment) next = evolveCard(state, card.uid, action.payment, action.superEvolve);
    }
    onClose();
    setState(next);
  };
  return (
    <div className="modal-backdrop" role="dialog" aria-modal="true" aria-label={def.name}>
      <section className="detail-modal">
        <button type="button" className="modal-close" onClick={onClose} aria-label="關閉">×</button>
        <div className="detail-modal__art"><img src={def.image} alt={def.name} width={480} height={672} /></div>
        <div className="detail-modal__content">
          <p className="eyebrow">{KIND_LABEL[def.kind]} · {ZONE_LABEL[selection.zone]}</p>
          <h2>{def.name}</h2>
          <div className="detail-stats">
            <StatPill tone="gold">費用 {def.cost}</StatPill>
            {def.kind === "follower" ? <><StatPill>攻擊 {attackOf(card)}</StatPill><StatPill tone={remainingHealthOf(card) < maxHealthOf(card) ? "danger" : "good"}>體力 {remainingHealthOf(card)}/{maxHealthOf(card)}</StatPill></> : null}
            {card.tapped ? <StatPill tone="danger">橫置</StatPill> : <StatPill tone="good">直立</StatPill>}
          </div>
          <p className="card-rules-text">{def.text}</p>
          <div className="traits">{def.traits.map((trait) => <span key={trait}>{trait}</span>)}</div>
          {selection.side === "player" && visibleActions.length ? (
            <div className="detail-actions">
              {visibleActions.map((action) => (
                <button key={action.id} type="button" disabled={!action.enabled} onClick={() => perform(action.id)} title={action.reason}>
                  <strong>{action.label}</strong>
                  {!action.enabled && action.reason ? <span>{action.reason}</span> : null}
                </button>
              ))}
            </div>
          ) : null}
          {selection.side === "player" && actions.some((action) => action.id === "play" && !action.enabled) ? <p className="no-play-note">目前不能出場：{actions.find((action) => action.id === "play")?.reason}</p> : null}
        </div>
      </section>
    </div>
  );
}

function Drawer({ drawer, onClose, onSelect }: { drawer: ZoneDrawer; onClose: () => void; onSelect: (selection: SelectedCard) => void }) {
  return (
    <div className="modal-backdrop" role="dialog" aria-modal="true" aria-label={drawer.title}>
      <section className="drawer-modal">
        <button type="button" className="modal-close" onClick={onClose}>×</button>
        <p className="eyebrow">區域查看</p>
        <h2>{drawer.title}</h2>
        {drawer.note ? <p>{drawer.note}</p> : null}
        <div className="drawer-grid">
          {drawer.cards.length ? drawer.cards.map((card) => <MiniCard key={card.uid} card={card} compact onClick={card.uid.startsWith("evolve-") ? undefined : () => { onClose(); onSelect({ uid: card.uid, side: drawer.side, zone: drawer.zone }); }} />) : <div className="empty-zone">目前沒有卡片</div>}
        </div>
      </section>
    </div>
  );
}

export default function GameSimulator() {
  const [state, setState] = useState<GameState | null>(null);
  const [replaySource, setReplaySource] = useState<LoadedReplaySet | null>(null);
  const [policy, setPolicy] = useState<DestructionPolicy | null>(null);
  const [policyError, setPolicyError] = useState<string | null>(null);
  const [aiStatus, setAiStatus] = useState("AI 思考中");
  const [selected, setSelected] = useState<SelectedCard | null>(null);
  const [drawer, setDrawer] = useState<ZoneDrawer | null>(null);
  const [showLog, setShowLog] = useState(import.meta.env.DEV);
  const sentRef = useRef({ gameId: "", count: 0 });

  const beginPolicyLoad = () => {
    setPolicyError(null);
    void loadDestructionPolicy()
      .then(setPolicy)
      .catch((reason: unknown) => setPolicyError(reason instanceof Error ? reason.message : String(reason)));
  };

  useEffect(() => {
    beginPolicyLoad();
  }, []);

  useEffect(() => {
    if (!import.meta.env.DEV) return;
    const replayPath = new URLSearchParams(window.location.search).get("replay");
    if (!replayPath) return;
    const target = new URL(replayPath, window.location.href);
    if (target.origin !== window.location.origin || !target.pathname.endsWith(".json")) return;
    const controller = new AbortController();
    void fetch(target, { signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error(`回放讀取失敗：HTTP ${response.status}`);
        return response.json() as Promise<unknown>;
      })
      .then((value) => {
        setReplaySource(parseReplayFile(value, target.pathname.split("/").at(-1) ?? "replay.json"));
      })
      .catch((reason: unknown) => {
        if (reason instanceof DOMException && reason.name === "AbortError") return;
        console.error(reason);
      });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (!state || !import.meta.env.DEV) return;
    if (sentRef.current.gameId !== state.gameId) sentRef.current = { gameId: state.gameId, count: 0 };
    const fresh = state.events.slice(sentRef.current.count);
    if (!fresh.length) return;
    sentRef.current.count = state.events.length;
    void fetch("/__matchlog", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ gameId: state.gameId, lines: fresh }),
    }).catch(() => {});
  }, [state]);

  useEffect(() => {
    if (!state || !policy || trainingActor(state) !== "ai" || policyError) return;
    let cancelled = false;
    const decisionState = state;
    const timer = window.setTimeout(() => {
      setAiStatus(state.status === "mulligan" ? "AI 判斷留牌中" : "AI 計算最佳動作中");
      void policy.choose(decisionState)
        .then((decision) => {
          if (cancelled) return;
          setAiStatus(`${decision.action.label} · ${decision.inferenceMs.toFixed(0)}ms`);
          setState((current) => current === decisionState ? applyTrainingAction(current, decision.action) : current);
        })
        .catch((reason: unknown) => {
          if (cancelled) return;
          setPolicyError(reason instanceof Error ? reason.message : String(reason));
        });
    }, 180);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [state, policy, policyError]);

  const turnLabel = useMemo(() => {
    if (!state) return "";
    if (state.status === "gameover") return state.winner === "player" ? "勝利" : state.winner === "ai" ? "敗北" : "平手";
    return state.turnSide === "player" ? `你的第${state.player.ownTurn}回合` : `破壞巫第${state.ai.ownTurn}回合`;
  }, [state]);

  if (import.meta.env.DEV && replaySource) return <ReplayViewer source={replaySource} onClose={() => setReplaySource(null)} />;
  if (!state) return <SetupScreen onStart={(first, seed, deck) => { setPolicyError(null); setState(createGame(first, seed, deck, { aiControl: "manual" })); }} onReplay={setReplaySource} policy={policy} policyError={policyError} onRetry={beginPolicyLoad} />;
  if (state.status === "mulligan") {
    if (trainingActor(state) === "ai") return <AiMulliganScreen error={policyError} />;
    return <><MulliganScreen state={state} onKeep={() => setState(finishManualMulligan(state, "player", false))} onRedraw={() => setState(finishManualMulligan(state, "player", true))} onInspect={setSelected} />{selected ? <CardDetail state={state} selection={selected} onClose={() => setSelected(null)} setState={setState} /> : null}</>;
  }

  return (
    <main className="game-shell">
      <header className="game-header">
        <div>
          <p className="eyebrow">{PLAYER_DECKS[state.playerDeck].label} × 破壞巫</p>
          <h1>{state.playerDeck === "levin" ? "雷維翁 vs. 破壊" : state.playerDeck === "sekka" ? "雪華獸 vs. 破壊" : "妖精 vs. 破壊"}</h1>
        </div>
        <div className="turn-status">
          <span className={`turn-dot ${state.turnSide === "player" ? "is-player" : "is-ai"}`} />
          <strong>{turnLabel}</strong>
          <small>亂數種子 {state.seed}</small>
        </div>
        <div className="header-actions">
          {import.meta.env.DEV ? <button type="button" onClick={() => setShowLog((value) => !value)}>{showLog ? "隱藏紀錄" : "顯示紀錄"}</button> : null}
          <button type="button" onClick={() => { setState(restartWithSameSeed(state)); setSelected(null); }}>同亂數種子重開</button>
          <button type="button" onClick={() => { setState(null); setSelected(null); }}>新對局</button>
        </div>
      </header>

      {policyError ? <div className="model-error-banner" role="alert"><strong>破壞模型已暫停</strong><span>{policyError}</span></div> : null}
      <section className={`game-layout ${import.meta.env.DEV && showLog ? "has-log" : ""}`}>
        <div className="board-column">
          <section className="player-board player-board--ai">
            <PlayerVitals state={state} side="ai" />
            <div className="board-tools"><ZoneButtons state={state} side="ai" openDrawer={setDrawer} /><HandBacks count={state.ai.hand.length} /></div>
            <div className="ex-zone ex-zone--ai">
              <div className="zone-label"><span>破壞巫EX區</span><small>{state.ai.ex.length}/5 · 公開資訊</small></div>
              <CardRow cards={state.ai.ex} side="ai" zone="ex" onSelect={setSelected} empty="對方EX區目前沒有卡片" />
            </div>
            <div className="zone-label"><span>破壞巫場上</span><small>{state.ai.field.length}/5</small></div>
            <CardRow cards={state.ai.field} side="ai" zone="field" onSelect={setSelected} empty="對方場上沒有卡片" />
          </section>

          <div className="battle-divider">
            <span />
            <strong>{state.lastAction ?? "等待行動"}</strong>
            <span />
          </div>

          <section className="player-board player-board--player">
            <div className="zone-label"><span>我方場上</span><small>{state.player.field.length}/5</small></div>
            <CardRow cards={state.player.field} side="player" zone="field" onSelect={setSelected} empty="點擊手牌，查看效果後選擇出場" />
            <div className="ex-zone">
              <div className="zone-label"><span>EX區</span><small>{state.player.ex.length}/5 · 可直接使用</small></div>
              <CardRow cards={state.player.ex} side="player" zone="ex" onSelect={setSelected} empty="EX區目前沒有卡片" />
            </div>
            <div className="board-tools"><ZoneButtons state={state} side="player" openDrawer={setDrawer} />{trainingActor(state) === "player" && state.turnSide === "player" && state.phase === "main" && !state.pending ? <button type="button" className="end-turn-button" onClick={() => setState(endTurn(state))}>結束回合</button> : <span className="waiting-pill">{trainingActor(state) === "ai" ? aiStatus : "正在處理效果"}</span>}</div>
            <PlayerVitals state={state} side="player" />
          </section>

          <section className="hand-dock">
            <div className="zone-label"><span>我的手牌</span><small>{state.player.hand.length}/7</small></div>
            <CardRow cards={state.player.hand} side="player" zone="hand" onSelect={setSelected} empty="手牌為空" />
          </section>
        </div>

        {import.meta.env.DEV && showLog ? (
          <aside className="log-panel">
            <div className="log-panel__heading"><p className="eyebrow">行動紀錄</p><h2>對局紀錄</h2></div>
            <ol>{state.log.map((entry, index) => <li key={`${index}-${entry}`} className={index === 0 ? "is-latest" : ""}><span>{state.log.length - index}</span><p>{entry}</p></li>)}</ol>
            <div className="rules-note"><strong>規則基準</strong><p>Shadowverse EVOLVE 綜合規則 ver.1.27與官方卡片Q&amp;A。衍生卡不留墓；EX與場各上限5；手牌上限7。</p></div>
          </aside>
        ) : null}
      </section>

      {state.pending && trainingActor(state) === "player" ? <PendingModal key={`${state.pending.effect}-${state.pending.title}-${state.pending.options.map((option) => option.uid).join("|")}`} pending={state.pending} onResolve={(uids) => setState(resolveChoice(state, uids))} /> : null}
      {selected ? <CardDetail state={state} selection={selected} onClose={() => setSelected(null)} setState={setState} /> : null}
      {drawer ? <Drawer drawer={drawer} onClose={() => setDrawer(null)} onSelect={setSelected} /> : null}
      {state.status === "gameover" ? (
        <div className="gameover-banner"><p className="eyebrow">對局結束</p><h2>{state.winner === "player" ? "你擊敗了破壞巫" : state.winner === "ai" ? "破壞巫取得勝利" : "平手"}</h2><div><button type="button" className="primary-button" onClick={() => setState(restartWithSameSeed(state))}>同亂數種子再試一次</button><button type="button" className="secondary-button" onClick={() => setState(null)}>換先後手／亂數種子</button></div></div>
      ) : null}
    </main>
  );
}
