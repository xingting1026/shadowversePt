# 三方 Policy League：訓練、評估與回放

目前訓練環境是 engine version 3。妖精、雷維翁、破壞三方都有自己的神經網路 policy；破壞不再是
固定規則 AI。每個模型都要自行決定整手留／換、出牌、攻擊與目標、進化、ACT、Quick、效果目標、
可選搜索、額外費用、子集合與牌庫底順序。

## 為什麼不是傳統 GAN

這是零和、不完全資訊、序列決策遊戲，不是「生成器製造樣本、判別器判真假」的問題。採用的是多智能體
self-play／alternating best response：一方更新時凍結對手，避免兩個 policy 同時移動造成不穩；每局有
50% 對最新對手、50% 均勻抽歷史 checkpoint，避免只會剋上一版而忘記舊策略。訓練完成後檢查最新
整列與整欄，量測是否仍被歷史版本剝削。

這能提供「在目前規則引擎、牌表、網路容量與歷史對手池內的經驗穩定策略」，不能數學證明全遊戲的
全域 Nash equilibrium。

## 正式版本與結果

- cycle 0–12：三方交替 masked PPO；每 phase 20,000 個目標方決策。
- cycle 13：只校正留牌的獨立 residual head；其他出牌／攻擊／效果 logits 凍結。
- 留牌標籤：同一 seed、同一起手各跑「全留」與「全換」兩條完整未來線，再比較終局。
- 收斂篩檢：妖精的最大歷史剝削差距 0%；雷維翁 4%，低於 5% 門檻。

正式盲測使用未參與訓練與留牌標註的新 seed，決定論推論，先後攻各 500 局：

| 對局 | 勝敗 | 玩家方勝率 | 95% Wilson | 先攻 | 後攻 | 截斷 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 妖精 vs 破壞 | 463–537 | 46.3% | 43.2%–49.4% | 44.6% | 48.0% | 0 |
| 雷維翁 vs 破壞 | 531–469 | 53.1% | 50.0%–56.2% | 59.4% | 46.8% | 0 |

輸出：

- `training-output/league-v1/`：三方 cycle 0–13 checkpoint、矩陣、frontier 與評估摘要；
- `match-logs/fairy-league-final.json`：妖精正式回放包；
- `match-logs/levin-league-final.json`：雷維翁正式回放包；
- `match-logs/replay-qa-1280x720.png`：1280×720 中盤版面 QA。

`training-output/` 與 `match-logs/` 是大型本機產物，已列入 `.gitignore`。

GitHub Pages 只部署破壞巫的 cycle 13 推論模型：

- `public/models/destruction-cycle13.onnx`：瀏覽器 ONNX 模型；
- `public/models/destruction-cycle13.json`：engine、卡牌字典、張量 schema、SHA-256 與匯出 parity；
- 妖精與雷維翁模型只用於訓練／評估，不會取代網頁上的玩家操作。

## 執行

續跑 league：

```powershell
python training\train_league.py --cycles 18 --phase-decisions 20000 --envs 32 --output training-output\league-v1 --resume
```

檢查最新版對歷史 checkpoint：

```powershell
python training\evaluate_frontier.py --league training-output\league-v1 --games 50 --stride 3
```

1000 局正式評估並錄回放：

```powershell
python training\record_league.py --league training-output\league-v1 --deck fairy --games 1000 --output match-logs\fairy-league-final.json
python training\record_league.py --league training-output\league-v1 --deck levin --games 1000 --output match-logs\levin-league-final.json
```

驗證回放可由 seed 與決策序列逐步重建：

```powershell
npx tsx training-verify-replays.ts match-logs\fairy-league-final.json match-logs\levin-league-final.json
```

重新輸出網頁模型，並比對 PyTorch 與 ONNX：

```powershell
python training\export_onnx.py --checkpoint training-output\league-v1\destruction\current.pt --output public\models\destruction-cycle13.onnx --manifest public\models\destruction-cycle13.json
npm run verify:model
```

`verify:model` 會讓 ONNX 破壞模型實際完成妖精、雷維翁各一局，並將真實局面的瀏覽器輸入與原
PyTorch checkpoint 逐值比對。

## dev-only 網頁回放

```powershell
npm run dev
```

只有 `npm run dev` 會顯示「載入 AI 對局回放」並接受 `?replay=`；production build 會移除回放與
match log UI。dev 模式可在首頁選 JSON，也可直接在終端顯示的網址後加：

```text
?replay=match-logs/levin-league-final.json
?replay=match-logs/fairy-league-final.json
```

例如 `http://localhost:5173/shadowversePt/?replay=match-logs/levin-league-final.json`。播放器固定由上至下顯示
破壞手牌、破壞 EX、破壞場上、玩家場上、玩家 EX、玩家手牌；可切換玩家、破壞與裁判全知視角，
逐手查看合法動作、選擇機率和 value。

## 可信度邊界

18 份正式回放共 1,872 個決策已全部重建驗證。回放與勝率仍只反映目前
三副固定牌表及已實作規則。若卡牌效果、牌表、動作枚舉或 observation 改變，舊 checkpoint 必須重新
驗證或重訓，不能沿用舊勝率宣稱新版本同樣穩定。
