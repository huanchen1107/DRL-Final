from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple


@dataclass
class Config:
    # ── Paths ──
    project_dir: Path = Path(__file__).resolve().parent
    outputs_dir: Path = project_dir / "outputs"

    # ── Ticker & Date ──
    ticker: str = "2330.TW"
    start_date: str = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    end_date: str = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

    # 主交易時區
    base_interval: str = "1h"

    # yfinance intraday fallback periods，從長到短嘗試
    intraday_fallback_periods: Tuple[str, ...] = ("730d", "365d", "180d", "90d", "60d")

    # 為了讓 W1 / D1 指標有足夠 rolling window，Daily 往前多抓的天數
    daily_extra_days: int = 800

    # ── Data split ──
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # ── Trading environment ──
    initial_cash: float = 100000.0
    transaction_cost_rate: float = 0.001425
    tax_rate: float = 0.003

    # ── DQN training ──
    episodes: int = 25
    batch_size: int = 64
    gamma: float = 0.95
    lr: float = 1e-4
    replay_size: int = 100000
    min_replay_size: int = 1200
    target_update_freq: int = 300

    # ── Epsilon-greedy ──
    epsilon_start: float = 1.0
    epsilon_min: float = 0.03
    epsilon_decay: float = 0.965

    # ── Reward weights ──
    reward_scale: float = 100.0
    drawdown_penalty: float = 0.10
    trade_penalty: float = 0.003
    mtf_bonus_weight: float = 0.003
    higher_tf_conflict_penalty: float = 0.002

    # ── SMC feature params ──
    swing_window: int = 5
    lookback_range: int = 60

    # ── Risk Reward params ──
    rr_atr_multiplier: float = 1.5
    rr_target: float = 2.0
    rr_swing_buffer_pct: float = 0.002
    min_rr_threshold: float = 1.5

    # ── Reproducibility ──
    seed: int = 42

    # ── Legacy compat (used by app.py for smartmoneyconcepts chart) ──
    rolling_window: int = 100


# 現貨多方模型：目標部位比例
ACTION_POSITION_RATIOS = [0.0, 0.25, 0.50, 1.0]
ACTION_NAMES = {
    0: "Stay in Cash / 0% Position",
    1: "Adjust to 25% Position",
    2: "Adjust to 50% Position",
    3: "Adjust to 100% Position",
}
