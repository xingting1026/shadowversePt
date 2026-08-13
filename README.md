# 妖精 vs. 破壞巫對局模擬器

這是一個固定牌組的 Shadowverse EVOLVE 互動對局模擬器：玩家操作妖精牌組，對手則是依局面評分、採取規則型決策的破壞巫 AI。

線上版本：[GitHub Pages](https://xingting1026.github.io/shadowversePt/)

## 功能

- 自選先攻或後攻，依亂數種子隨機洗牌。
- 起手可整手保留或全部重抽一次。
- 點擊卡片先看卡圖與中文效果，再決定是否使用或選取。
- 顯示牌庫、手牌、場上、EX區、墓場、消失領域與進化區。
- 破壞巫 AI 會優先建立蛋的資源循環，並保留可進化從者到能進化的時點。
- 不保存對局；重新整理頁面後會回到新對局畫面。

## 本機執行

需要 Node.js 22.13 或更新版本。

```bash
npm install
npm run dev
```

完整測試與建置：

```bash
npm test
```

推送到 `main` 後，GitHub Actions 會自動測試並部署 `dist` 到 GitHub Pages。
