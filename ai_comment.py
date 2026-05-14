import requests
import streamlit as st


def _build_prompt(recommendation: dict, metrics: dict) -> str:
    snap = recommendation.get("mtf_snapshot", {})
    rr = recommendation.get("risk_reward_plan", {})
    rr_details = snap.get("rr_details", {})

    def fmt(val, decimals=2):
        if val is None:
            return "N/A"
        try:
            return f"{val:,.{decimals}f}"
        except Exception:
            return str(val)

    def rr_block(tf: str) -> str:
        d = rr_details.get(tf, {})
        entry = d.get("entry")
        if entry is None or (hasattr(entry, "__float__") and entry != entry):
            return "（無有效設置）"
        return (
            f"進場：{fmt(entry)} / 停損：{fmt(d.get('stop_loss'))} / "
            f"止盈：{fmt(d.get('take_profit'))} / RR：{fmt(d.get('rr_ratio'))}"
        )

    conflict_str = "是" if snap.get("mtf_conflict") else "否"
    rr_valid = rr.get("risk_reward_valid", False)
    current_rrr_block = (
        f"- 進場價：{fmt(rr.get('entry_price'))}\n"
        f"- 停損價：{fmt(rr.get('stop_loss_price'))}\n"
        f"- 止盈價：{fmt(rr.get('take_profit_price'))}\n"
        f"- 風報比：{fmt(rr.get('risk_reward_ratio'))}\n"
        f"- 計算依據：{rr.get('take_profit_basis', 'N/A')}"
        if rr_valid else "（無有效風報比）"
    )

    return f"""你是一位專業的量化交易分析師，擅長 Smart Money Concepts (SMC) 與強化學習回測策略分析。
請根據以下指標，給出專業的交易評語，並建議下一步行動。

## 當前交易訊號

- 最新收盤價：{fmt(recommendation.get('latest_close'))}
- 建議動作：{recommendation.get('best_action_name', 'N/A')}
- 交易方向：{recommendation.get('trade_direction', 'N/A')}
- 建議倉位：{recommendation.get('target_position_ratio', 0):.0%}

## MTF SMC 多時框偏向分析

- W1（週線）偏向：{snap.get('w1_smc_bias', 0):.0f}（+1 看多 / -1 看空 / 0 中性）
- D1（日線）偏向：{snap.get('d1_smc_bias', 0):.0f}
- H4（4小時）偏向：{snap.get('h4_smc_bias', 0):.0f}
- H1（1小時）偏向：{snap.get('h1_smc_bias', 0):.0f}
- 多時框共識分數：{snap.get('mtf_confluence_score', 0):.2f}（範圍 -1 ~ +1）
- 方向衝突：{conflict_str}

## 回測績效指標

- 總報酬：{metrics.get('total_return', 0) * 100:.1f}%
- 最大回撤：{metrics.get('max_drawdown', 0) * 100:.1f}%
- 夏普比率：{metrics.get('sharpe_ratio', 0):.2f}
- 獲利因子：{metrics.get('profit_factor', 0):.2f}

## 當前目標風報比（RRR）

{current_rrr_block}

## 多時框風報比分析

### W1 週線級別
- {rr_block('w1')}

### D1 日線級別
- {rr_block('d1')}

### H4 四小時級別
- {rr_block('h4')}

### H1 一小時級別
- {rr_block('h1')}

---

請依照以下格式以**繁體中文**回覆：

### 📊 指標綜合評語
（對以上所有指標的整體解讀，包含多時框偏向是否一致、回測績效是否可信、風報比是否合理）

### ⚠️ 風險提示
（指出當前最主要的風險因素，例如：時框衝突、回撤偏高、共識分數偏低等）

### 🎯 建議下一步行動
（具體建議：進場 / 觀望 / 減倉等，並說明理由）

### 💡 理由說明
（結合以上指標，解釋為何給出此建議）
"""


def generate_ai_comment(recommendation: dict, metrics: dict) -> str:
    api_key = st.secrets.get("OLLAMA_API_KEY", "")
    if not api_key:
        return "❌ 未設定 OLLAMA_API_KEY，請在 Streamlit Cloud Secrets 中加入 `OLLAMA_API_KEY = \"your-key\"`。"

    prompt = _build_prompt(recommendation, metrics)
    resp = requests.post(
        "https://ollama.com/api/chat",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": "gemma4:31b-cloud",
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]
