# 妖精／雷維翁 vs. 破壞巫對局模擬器

這是一個固定牌組的 Shadowverse EVOLVE 互動對局模擬器：玩家可在開局選擇妖精（7P9XP）或雷維翁皇家（1KUUZE）牌組。互動模式的對手是 cycle 13 破壞巫 policy；模型由妖精、雷維翁、破壞三方 self-play league 訓練，直接在瀏覽器本機推論。

雷維翁牌組的核心：靠棄牌（ミイム、ルネス）與磨牌（弓使い）把雷維翁卡牌堆進墓場，5張開啟全隊強化（ジェノ疾走、メイム指定攻擊、マイム/弓使い3點傷害、超越者三選三），墓場10張[ロイヤル]從者後アルベール可以3PP重立多段疾走收尾。進化區依實戰只收錄ルネス×3與マイム×3。

線上版本：[GitHub Pages](https://xingting1026.github.io/shadowversePt/)

## 功能

- 開局選擇妖精或雷維翁牌組、先攻或後攻，依亂數種子隨機洗牌。
- 起手可整手保留或全部重抽一次。
- 點擊卡片先看卡圖與中文效果，再決定是否使用或選取。
- 顯示牌庫、手牌、場上、EX區、墓場、消失領域與進化區。
- cycle 13 破壞模型自行決定留牌、出牌、攻擊、進化、ACT、Quick、效果目標與是否選取。
- 同一個破壞模型同時以妖精與雷維翁對局訓練，observation 內含對戰牌組識別，不是兩個互不相干的規則腳本。
- 模型 manifest 會核對 engine 版本、卡牌字典、張量尺寸、檔案長度與 SHA-256，避免錯版模型靜默運行。
- 對局紀錄與 AI 回放只在 dev 模式提供，不會出現在 GitHub Pages 正式版。
- 不保存對局；重新整理頁面後會回到新對局畫面。

## 本機執行

需要 Node.js 22.13 或更新版本。

```bash
npm install
npm run dev
```

dev 模式會把每一局的結構化事件（雙方出牌、進化、攻擊、血量變化）
自動寫入 `match-logs/match-<gameId>.jsonl`，供事後分析。

正式模型、1000 局盲測結果與訓練指令見 `TRAINING.md`。dev 回放可在首頁選擇 JSON，或直接開啟：

```text
http://localhost:5173/shadowversePt/?replay=match-logs/levin-league-final.json
http://localhost:5173/shadowversePt/?replay=match-logs/fairy-league-final.json
```

完整測試與建置：

```bash
npm test
npm run verify:model
```

推送到 `main` 後，GitHub Actions 會自動測試並部署 `dist` 到 GitHub Pages。
