# 📈 SMC × DRL Trading Platform

DEMO SITE:[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://114-2-drl-final.streamlit.app/)

此專案結合了**深度強化學習 (Deep Reinforcement Learning, DRL)** 與 **聰明錢概念 (Smart Money Concepts, SMC)**，建立一個具有高度互動性圖形介面與即時訓練追蹤的自動化量化交易輔助系統。

透過擷取多時間級別（Multi-Timeframe: W1, D1, H4, H1）的市場微觀結構特徵，系統輔助 Deep Q-Network (DQN) 代理人學習最佳資金部位管理策略，並自動推算最佳風險報酬比 (Risk-Reward Ratio, RRR) 交易計畫。

---

## ✨ 核心特色 (Core Features)

### 1. 🔍 聰明錢概念 (Smart Money Concepts, SMC) 分析
系統使用 `smartmoneyconcepts` 技術將傳統 K 線轉化為機構級交易員關注的市場結構特徵，包含：
* **流動性掃蕩 (Liquidity Sweeps)**
* **前期高低點 (Old Highs / Old Lows / BSL / SSL)**
* **合理價值缺口 (Fair Value Gaps, FVG)**：Bullish / Bearish Gaps
* **訂單塊 (Order Blocks, OB)**：供需失衡的機構建倉區塊
* **溢價/折價區 (Premium / Discount Zones)**

### 2. 🧠 多時間級別 (MTF) DQN 決策代理人
突破單一時間框架的限制，模型架構融合了**大局觀**與**微觀進場點**：
* **狀態維度 (State Dimension)**：結合了技術指標與 W1, D1, H4, H1 四個時間級別的 SMC Bias（偏誤方向）。
* **動作空間 (Action Space)**：採用資金部位管理模式，動態調整持倉比例 (`0%`, `25%`, `50%`, `100%`)。
* **獎勵塑形 (Reward Shaping)**：
  * 加入 **MTF Bonus**：順應大時區趨勢時給予額外獎勵。
  * 加入 **Conflict Penalty**：逆勢操作時給予懲罰。
  * 結合最大回撤懲罰 (Drawdown Penalty) 與交易摩擦成本 (Trade Penalty)。

### 3. 📊 專業級互動式視覺化儀表板 (Streamlit Dashboard)
提供無縫的資料獲取、模型訓練與圖表回測體驗：
* **SMC K線圖**：透過 Plotly 渲染高互動性 K 線圖，支援動態顯示/隱藏 FVG、OB 及流動性標記。
* **回測交易點位可視化**：在圖表上精準標註測試集的 `BUY` / `SELL` 動作。
* **最佳/最差交易分析**：自動配對買賣點，計算每一筆交易的報酬率，並以視覺化標示出 **Best RRR** (金色) 與 **Worst RRR** (紅色) 的交易區間。
* **純淨的 UI/UX 設計**：採用 Inter 現代英文字體、無表情符號的極簡專業化介面設計。

### 4. 🚀 即時訓練追蹤與推薦報告
* **Terminal-like Training Log**：於網頁端即時串流 DQN 訓練過程 (Episodes, Reward, Loss, Return)，支援自動向下滾動。
* **DRL × SMC 推薦戰情室**：
  * **Recommendation**: 系統給予最新的建倉建議與持倉比例。
  * **MTF SMC**: 各級別的方向性偏誤 (Bias) 與共識分數 (Confluence Score)。
  * **Backtest**: 顯示測試集之 Sharpe Ratio、Max Drawdown 與 Profit Factor。
  * **RRR**: 基於當前市場結構，自動計算建議的入場點 (Entry)、止損點 (Stop Loss) 與止盈點 (Take Profit)。

---

## 📁 專案架構 (Project Structure)

```text
DRL-Final/
├── app.py                      # Streamlit 網頁主程式（SMC儀表板、互動繪圖、訓練追蹤）
├── train.py                    # DRL 模型訓練邏輯、資料切分與訓練管線
├── config.py                   # 系統超參數、環境配置與部位設定檔
├── recommend.py                # 交易策略推理與 RRR 計畫生成
├── requirements.txt            # 依賴套件列表
├── agent/
│   └── dqn_agent.py            # DQN 模型與代理人實作、經驗回放池
├── env/
│   └── trading_env.py          # 支援 MTF SMC 狀態的 Gym-like 交易環境
├── model/
│   └── network.py              # 神經網路結構實作
└── utils/
    ├── data_utils.py           # yfinance 資料爬取、MTF 合併與 SMC 特徵前處理
    └── metrics.py              # 回測成效與夏普值等 KPI 計算
```

---

## 🛠️ 安裝與啟動 (Installation & Quick Start)

### 1. 建立虛擬環境與安裝依賴套件

為避免套件衝突，強烈建議使用虛擬環境 (Virtual Environment)：

```bash
# 建立並啟動虛擬環境 (Mac/Linux)
python3 -m venv venv
source venv/bin/activate

# 若為 Windows 用戶，請使用：
# python -m venv venv
# .\venv\Scripts\activate

# 安裝所需套件
pip install -r requirements.txt

# 安裝 SMC 指標分析庫 (若 requirements.txt 未包含)
pip install smartmoneyconcepts
```

### 2. 啟動 Streamlit 服務

在專案根目錄下的終端機執行以下指令啟動儀表板：

```bash
streamlit run app.py
```

---

## 📖 操作指南 (User Guide)

1. **參數設定**：於瀏覽器開啟網頁後，在頂部輸入目標**股票代號 / Ticker**（例如 `2330.TW`、`AAPL` 或加密貨幣 `BTC-USD`），並選擇**開始日期**與**結束日期**。
2. **抓取資料**：點擊 **`Fetch & Analyze`**，系統會自動透過 `yfinance` 抓取並運算 SMC 特徵，渲染高互動性的 SMC K線圖表。
3. **多時間框架切換**：在圖表上方的下拉選單可以即時切換 `1h`, `4h`, `1d`, `1wk` 級別的視角，檢視不同維度的市場結構。
4. **啟動訓練**：點擊圖表下方的 **`DQN + SMC + MTF + RRR (...)`** 按鈕，系統便會自動在背景執行 DRL 訓練，並在畫面上即時滾動輸出訓練進度與報酬率。
5. **檢視策略報告**：訓練完成後，圖表上會疊加 DRL 代理人在測試集(Test Set) 的交易紀錄（綠上箭頭/紅下箭頭），並突顯表現最好與最差的波段。同時下方會呈現四欄式的**分析建議報告 (DRL × SMC Report)**。

---

## ⚙️ 訓練超參數設定 (Hyperparameters)
如需微調 DRL 代理人或環境設定，請編輯 `config.py`，重點參數如下：

* **部位比例**：`ACTION_POSITION_RATIOS = [0.0, 0.25, 0.50, 1.0]`
* **訓練/驗證/測試 切分**：`train_ratio=0.7`, `val_ratio=0.15`, `test_ratio=0.15`
* **交易成本**：手續費率 `0.001425`，交易稅 `0.003` (依台股預設)
* **DQN 網路**：`gamma=0.95`, `lr=1e-4`, `batch_size=64`, `episodes=25`

---

> **⚠️ 免責聲明 (Disclaimer)**：本專案為學術研究與程式開發練習，模型輸出的「買賣建議」、「最佳進出場點」與「回測績效」僅供參考。實際金融市場具高度風險，本系統**不構成任何實質投資建議**，使用者應自行承擔投資風險。

---

## 更新紀錄 (Changelog)

### [2026-05-13] V2 DQN 模型整合與 MTF Risk-Reward Ratio (RRR) 視覺化
為提升多時區風險報酬分析能力，本次更新將實驗性質的 `DQN + SMC + MTF + RRR` 模型完整實作進系統中，並採取平行架構（V2）以確保不破壞原有的穩定版本。

**核心更新項目：**
1. **建立 V2 雙軌訓練與環境架構**
   - 擴充 `utils/data_utils_v2.py`：將原有的 62 個特徵擴增至 82 個，完美整合 Jupyter Notebook 中實作的 `w1`, `d1`, `h4`, `h1` 四個時區的 Risk-Reward Ratio (RRR) 相關特徵。
   - 新增 `env/trading_env_v2.py` 與 `train_v2.py`：獨立訓練管線，訓練後模型將獨立儲存為 `mtf_dqn_model_v2.pth`。
2. **推論模組升級 (`recommend_v2.py`)**
   - 在產生策略推薦時，不僅依據 DQN 輸出提供倉位建議，更一併將多時區 (MTF) 的 RRR 狀態（含進場價、停損價、停利價、風報比與判斷基準）捕捉並記錄為 snapshot。
3. **Streamlit UI 強化與視覺化整合 (`app.py`)**
   - **雙模型自由切換**：在介面上新增平行的 `V2: DQN + SMC + MTF + RRR (Advanced)` 訓練按鈕，保留舊版與新版的靈活切換。
   - **四欄位 RRR 戰情室**：將 W1, D1, H4, H1 四個時區的風險報酬數據獨立設計成 4 欄並排的詳細分析區塊，提升數據可讀性。
   - **互動式 SMC 圖表標註**：在圖表區塊新增下拉選單，允許使用者勾選特定時區，並將該時區的 RRR 策略（Entry, Stop Loss, Take Profit）以水平虛線與動態標籤直接畫在最新的 K 線圖上，實現所見即所得的視覺化圖表。