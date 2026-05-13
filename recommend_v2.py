"""Strategy recommendation with Risk Reward Ratio plan (V2)."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from config import ACTION_POSITION_RATIOS, ACTION_NAMES
from utils.data_utils_v2 import (
    FEATURE_COLUMNS, apply_standardizer,
    download_and_build_mtf, build_mtf_dataset,
)


def build_latest_state(
    latest_mtf_raw: pd.DataFrame,
    feature_cols: List[str],
    feature_mean: pd.Series,
    feature_std: pd.Series,
    current_cash: float,
    current_shares: float,
    initial_cash: float,
):
    latest_scaled = apply_standardizer(latest_mtf_raw, feature_cols, feature_mean, feature_std)
    latest_row_scaled = latest_scaled.iloc[-1]
    latest_row_raw = latest_mtf_raw.iloc[-1]
    features = latest_row_scaled[feature_cols].values.astype(np.float32)
    price = float(latest_row_raw["close"])
    portfolio_value = current_cash + current_shares * price
    cash_ratio = current_cash / max(portfolio_value, 1e-8)
    position_ratio = (current_shares * price) / max(portfolio_value, 1e-8)
    unrealized_pnl_ratio = portfolio_value / max(initial_cash, 1e-8) - 1.0
    portfolio_features = np.array([cash_ratio, position_ratio, unrealized_pnl_ratio], dtype=np.float32)
    state = np.concatenate([features, portfolio_features])
    return state, latest_row_raw, price, portfolio_value


def calculate_risk_reward_plan(
    latest_row: pd.Series,
    current_price: float,
    target_position_ratio: float,
    current_position_ratio: float,
    atr_multiplier: float = 1.5,
    target_rr: float = 2.0,
    swing_buffer_pct: float = 0.002,
) -> Dict:
    entry_price = float(current_price)
    is_long_entry = target_position_ratio > current_position_ratio
    if not is_long_entry:
        return {"risk_reward_valid": False, "risk_reward_note": "No new long entry suggested."}

    h1_atr = latest_row.get("h1_atr", np.nan)
    h4_atr = latest_row.get("h4_atr", np.nan)
    atr_value = h1_atr if not (pd.isna(h1_atr) or h1_atr <= 0) else h4_atr
    if pd.isna(atr_value) or atr_value <= 0:
        return {"risk_reward_valid": False, "entry_price": entry_price, "risk_reward_note": "ATR unavailable."}

    atr_stop = entry_price - atr_multiplier * float(atr_value)
    stop_candidates = []
    if atr_stop < entry_price:
        stop_candidates.append(atr_stop)
    for key in ["h1_last_swing_low", "h4_last_swing_low"]:
        val = latest_row.get(key, np.nan)
        if not pd.isna(val) and float(val) < entry_price:
            stop_candidates.append(float(val) * (1 - swing_buffer_pct))

    if not stop_candidates:
        return {"risk_reward_valid": False, "entry_price": entry_price, "risk_reward_note": "No valid stop-loss."}

    stop_loss_price = max(stop_candidates)
    risk_per_share = entry_price - stop_loss_price
    if risk_per_share <= 0:
        return {"risk_reward_valid": False, "entry_price": entry_price, "stop_loss_price": stop_loss_price, "risk_reward_note": "Invalid stop-loss."}

    rr_take_profit = entry_price + risk_per_share * target_rr
    take_profit_price = rr_take_profit
    take_profit_basis = f"Target RR {target_rr:.1f}x"

    resistance_candidates = []
    for key in ["h1_last_swing_high", "h4_last_swing_high", "d1_last_swing_high"]:
        val = latest_row.get(key, np.nan)
        if not pd.isna(val) and float(val) > entry_price:
            resistance_candidates.append(float(val))

    if resistance_candidates:
        nearest = min(resistance_candidates)
        structure_rr = (nearest - entry_price) / risk_per_share
        if structure_rr >= 1.2:
            take_profit_price = nearest
            take_profit_basis = "Nearest SMC swing high / resistance"

    reward_per_share = take_profit_price - entry_price
    return {
        "risk_reward_valid": True,
        "entry_price": entry_price,
        "stop_loss_price": stop_loss_price,
        "take_profit_price": take_profit_price,
        "risk_per_share": risk_per_share,
        "reward_per_share": reward_per_share,
        "risk_reward_ratio": reward_per_share / risk_per_share,
        "take_profit_basis": take_profit_basis,
        "atr_used": float(atr_value),
        "risk_reward_note": "Long entry risk plan generated.",
    }


def recommend_strategy_v2(
    agent,
    latest_mtf_raw: pd.DataFrame,
    cfg,
    feature_cols: List[str],
    feature_mean: pd.Series,
    feature_std: pd.Series,
    current_cash: float = 100000.0,
    current_shares: float = 0.0,
) -> Dict:
    state, latest_row, price, portfolio_value = build_latest_state(
        latest_mtf_raw, feature_cols, feature_mean, feature_std,
        current_cash, current_shares, cfg.initial_cash,
    )
    q_values = agent.get_q_values(state)
    best_action = int(np.argmax(q_values))
    target_ratio = ACTION_POSITION_RATIOS[best_action]
    target_position_value = portfolio_value * target_ratio
    current_position_value = current_shares * price
    trade_value = target_position_value - current_position_value
    current_position_ratio = current_position_value / max(portfolio_value, 1e-8)

    rr_plan = calculate_risk_reward_plan(
        latest_row, price, target_ratio, current_position_ratio,
    )

    trade_direction = "BUY" if trade_value > 0 else ("SELL" if trade_value < 0 else "HOLD")

    return {
        "ticker": cfg.ticker,
        "latest_close": price,
        "best_action_name": ACTION_NAMES[best_action],
        "target_position_ratio": target_ratio,
        "trade_direction": trade_direction,
        "suggested_trade_value": abs(trade_value),
        "q_values": {ACTION_NAMES[i]: float(q_values[i]) for i in range(len(q_values))},
        "risk_reward_plan": rr_plan,
        "mtf_snapshot": {
            "w1_smc_bias": float(latest_row.get("w1_smc_bias", 0)),
            "d1_smc_bias": float(latest_row.get("d1_smc_bias", 0)),
            "h4_smc_bias": float(latest_row.get("h4_smc_bias", 0)),
            "h1_smc_bias": float(latest_row.get("h1_smc_bias", 0)),
            "mtf_confluence_score": float(latest_row.get("mtf_confluence_score", 0)),
            "mtf_conflict": int(latest_row.get("mtf_conflict", 0)),
            "higher_tf_bullish": int(latest_row.get("higher_tf_bullish", 0)),
            "higher_tf_bearish": int(latest_row.get("higher_tf_bearish", 0)),
            "rr_details": {
                tf: {
                    "entry": float(latest_row.get(f"{tf}_rr_entry_price", np.nan)),
                    "stop_loss": float(latest_row.get(f"{tf}_rr_stop_loss_price", np.nan)),
                    "take_profit": float(latest_row.get(f"{tf}_rr_take_profit_price", np.nan)),
                    "rr_ratio": float(latest_row.get(f"{tf}_rr_ratio", 0)),
                    "basis": str(latest_row.get(f"{tf}_rr_take_profit_basis", "")),
                }
                for tf in ["w1", "d1", "h4", "h1"]
            }
        },
    }
