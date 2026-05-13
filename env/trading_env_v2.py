"""Multi-timeframe trading environment with target position rebalancing (V2)."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from config import ACTION_POSITION_RATIOS, ACTION_NAMES
from utils.data_utils_v2 import ensure_datetime_index, reset_datetime_index


class MTFTradingEnvV2:
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        initial_cash: float = 100000.0,
        transaction_cost_rate: float = 0.001425,
        tax_rate: float = 0.003,
        reward_scale: float = 100.0,
        drawdown_penalty: float = 0.10,
        trade_penalty: float = 0.001,
        mtf_bonus_weight: float = 0.003,
        higher_tf_conflict_penalty: float = 0.002,
    ):
        df = ensure_datetime_index(df)
        self.df = reset_datetime_index(df)
        self.feature_cols = feature_cols
        self.initial_cash = initial_cash
        self.transaction_cost_rate = transaction_cost_rate
        self.tax_rate = tax_rate
        self.reward_scale = reward_scale
        self.drawdown_penalty = drawdown_penalty
        self.trade_penalty = trade_penalty
        self.mtf_bonus_weight = mtf_bonus_weight
        self.higher_tf_conflict_penalty = higher_tf_conflict_penalty
        self.action_size = len(ACTION_POSITION_RATIOS)
        self.reset()

    def reset(self):
        self.step_idx = 0
        self.cash = self.initial_cash
        self.shares = 0.0
        self.portfolio_value = self.initial_cash
        self.peak_value = self.initial_cash
        self.done = False
        self.trades = []
        self.equity_curve = []
        return self._get_state()

    def _current_price(self) -> float:
        return float(self.df.loc[self.step_idx, "close"])

    def _get_state(self) -> np.ndarray:
        row = self.df.loc[self.step_idx]
        features = row[self.feature_cols].values.astype(np.float32)
        price = self._current_price()
        current_value = self.cash + self.shares * price
        cash_ratio = self.cash / max(current_value, 1e-8)
        position_ratio = (self.shares * price) / max(current_value, 1e-8)
        unrealized_pnl_ratio = current_value / max(self.initial_cash, 1e-8) - 1.0
        portfolio_features = np.array([cash_ratio, position_ratio, unrealized_pnl_ratio], dtype=np.float32)
        return np.concatenate([features, portfolio_features])

    def _rebalance_to_ratio(self, target_ratio: float) -> float:
        price = self._current_price()
        portfolio_value_before = self.cash + self.shares * price
        target_position_value = portfolio_value_before * target_ratio
        current_position_value = self.shares * price
        trade_value = target_position_value - current_position_value
        cost = 0.0
        if abs(trade_value) < 1e-8:
            return 0.0

        if trade_value > 0:
            buy_value = min(trade_value, self.cash)
            fee = buy_value * self.transaction_cost_rate
            actual = max(buy_value - fee, 0)
            shares_bought = actual / price
            self.shares += shares_bought
            self.cash -= buy_value
            cost += fee
            self.trades.append({
                "step": self.step_idx,
                "datetime": self.df.loc[self.step_idx, "datetime"],
                "type": "BUY", "price": price, "value": buy_value, "cost": fee,
            })
        else:
            sell_value = min(abs(trade_value), current_position_value)
            shares_sold = sell_value / price
            fee = sell_value * self.transaction_cost_rate
            tax = sell_value * self.tax_rate
            self.shares -= shares_sold
            self.cash += sell_value - fee - tax
            cost += fee + tax
            self.trades.append({
                "step": self.step_idx,
                "datetime": self.df.loc[self.step_idx, "datetime"],
                "type": "SELL", "price": price, "value": sell_value, "cost": fee + tax,
            })
        return cost

    def step(self, action: int):
        if self.done:
            raise ValueError("Episode is done. Please call reset().")

        price_before = self._current_price()
        value_before = self.cash + self.shares * price_before
        target_ratio = ACTION_POSITION_RATIOS[action]
        cost = self._rebalance_to_ratio(target_ratio)

        self.step_idx += 1
        if self.step_idx >= len(self.df) - 1:
            self.done = True

        price_after = self._current_price()
        value_after = self.cash + self.shares * price_after
        self.portfolio_value = value_after
        self.peak_value = max(self.peak_value, value_after)

        period_return = value_after / max(value_before, 1e-8) - 1.0
        drawdown = (self.peak_value - value_after) / max(self.peak_value, 1e-8)

        row = self.df.loc[self.step_idx]
        mtf_confluence = float(row["mtf_confluence_score"])
        w1_bias = float(row["w1_smc_bias"])
        d1_bias = float(row["d1_smc_bias"])
        mtf_conflict = int(row["mtf_conflict"])
        higher_tf_bearish = int(row["higher_tf_bearish"])
        higher_tf_bullish = int(row["higher_tf_bullish"])
        position_after = (self.shares * price_after) / max(value_after, 1e-8)

        mtf_alignment = np.tanh(mtf_confluence) * (position_after - 0.5)

        conflict_penalty = 0.0
        if higher_tf_bearish == 1 and position_after > 0.5:
            conflict_penalty += self.higher_tf_conflict_penalty * 1.5
        if w1_bias < 0 and position_after > 0.75:
            conflict_penalty += self.higher_tf_conflict_penalty
        if d1_bias < 0 and position_after > 0.75:
            conflict_penalty += self.higher_tf_conflict_penalty
        if higher_tf_bullish == 1 and position_after < 0.25:
            conflict_penalty += self.higher_tf_conflict_penalty * 0.5
        if mtf_conflict == 1 and position_after > 0.75:
            conflict_penalty += self.higher_tf_conflict_penalty

        reward = (
            period_return * self.reward_scale
            - self.drawdown_penalty * drawdown
            - self.trade_penalty * (1 if cost > 0 else 0)
            + self.mtf_bonus_weight * mtf_alignment
            - conflict_penalty
        )

        self.equity_curve.append({
            "step": self.step_idx,
            "datetime": row["datetime"],
            "portfolio_value": value_after,
            "cash": self.cash,
            "shares": self.shares,
            "position_ratio": position_after,
            "drawdown": drawdown,
            "mtf_confluence_score": mtf_confluence,
            "w1_smc_bias": w1_bias,
            "d1_smc_bias": d1_bias,
            "h4_smc_bias": float(row["h4_smc_bias"]),
            "h1_smc_bias": float(row["h1_smc_bias"]),
            "higher_tf_bullish": higher_tf_bullish,
            "higher_tf_bearish": higher_tf_bearish,
        })

        next_state = self._get_state()
        info = {
            "portfolio_value": value_after,
            "period_return": period_return,
            "drawdown": drawdown,
            "cost": cost,
            "target_ratio": target_ratio,
            "position_ratio": position_after,
            "mtf_confluence_score": mtf_confluence,
        }
        return next_state, reward, self.done, info


def make_env_v2(df: pd.DataFrame, cfg) -> MTFTradingEnvV2:
    from utils.data_utils_v2 import FEATURE_COLUMNS
    return MTFTradingEnvV2(
        df=df,
        feature_cols=FEATURE_COLUMNS,
        initial_cash=cfg.initial_cash,
        transaction_cost_rate=cfg.transaction_cost_rate,
        tax_rate=cfg.tax_rate,
        reward_scale=cfg.reward_scale,
        drawdown_penalty=cfg.drawdown_penalty,
        trade_penalty=cfg.trade_penalty,
        mtf_bonus_weight=cfg.mtf_bonus_weight,
        higher_tf_conflict_penalty=cfg.higher_tf_conflict_penalty,
    )
