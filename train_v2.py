"""Multi-timeframe DQN+SMC+RRR training pipeline (V2)."""
from __future__ import annotations

from typing import Dict, Optional, Callable

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agent.dqn_agent import DQNAgent
from backtest import backtest
from config import Config, ACTION_POSITION_RATIOS
from env.trading_env_v2 import MTFTradingEnvV2, make_env_v2 as make_env
from utils.data_utils_v2 import (
    FEATURE_COLUMNS, set_seed,
    download_and_build_mtf, build_mtf_dataset,
    split_data_time_order, fit_standardizer, apply_standardizer,
)


def run_episode(env: MTFTradingEnvV2, agent: DQNAgent, training: bool = True) -> Dict:
    state = env.reset()
    total_reward = 0.0
    losses = []

    while True:
        action = agent.select_action(state, training=training)
        next_state, reward, done, info = env.step(action)
        total_reward += reward

        if training:
            agent.replay_buffer.push(state, action, reward, next_state, done)
            loss = agent.update()
            if loss is not None:
                losses.append(loss)

        state = next_state
        if done:
            break

    return {
        "total_reward": total_reward,
        "final_value": env.portfolio_value,
        "total_return": env.portfolio_value / env.initial_cash - 1.0,
        "avg_loss": np.mean(losses) if losses else np.nan,
        "trades": len(env.trades),
    }


def train_agent(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: Config,
    progress_callback: Optional[Callable] = None,
):
    train_env = make_env(train_df, cfg)
    val_env = make_env(val_df, cfg)

    state_dim = train_env.reset().shape[0]
    action_dim = train_env.action_size
    agent = DQNAgent(state_dim, action_dim, cfg)

    logs = []
    best_val_return = -np.inf
    best_state_dict = None

    for ep in range(1, cfg.episodes + 1):
        train_result = run_episode(train_env, agent, training=True)
        val_result = run_episode(val_env, agent, training=False)
        agent.decay_epsilon()

        if val_result["total_return"] > best_val_return:
            best_val_return = val_result["total_return"]
            best_state_dict = {k: v.detach().cpu().clone() for k, v in agent.policy_net.state_dict().items()}

        log = {
            "episode": ep,
            "epsilon": agent.epsilon,
            "train_return": train_result["total_return"],
            "val_return": val_result["total_return"],
            "train_reward": train_result["total_reward"],
            "val_reward": val_result["total_reward"],
            "avg_loss": train_result["avg_loss"],
            "train_trades": train_result["trades"],
            "val_trades": val_result["trades"],
        }
        logs.append(log)

        log_line = (
            f"EP {ep:03d}/{cfg.episodes} | "
            f"eps={agent.epsilon:.4f} | "
            f"train_ret={train_result['total_return']:.2%} | "
            f"val_ret={val_result['total_return']:.2%} | "
            f"loss={train_result['avg_loss']:.5f} | "
            f"trades={train_result['trades']}"
        )
        print(log_line)
        if progress_callback:
            progress_callback(log_line)

    if best_state_dict is not None:
        agent.policy_net.load_state_dict(best_state_dict)
        agent.target_net.load_state_dict(best_state_dict)

    return agent, pd.DataFrame(logs)


def run_training_pipeline_v2(
    cfg: Config,
    progress_callback: Optional[Callable] = None,
) -> Dict:
    """Full pipeline: download → build MTF → train → backtest → save (V2)."""
    set_seed(cfg.seed)
    cfg.outputs_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download & build MTF dataset
    mtf_df, df_h1_raw, df_d1_raw = download_and_build_mtf(cfg, progress_callback)

    # 2. Split & standardize
    train_df, val_df, test_df = split_data_time_order(mtf_df, cfg)
    feature_mean, feature_std = fit_standardizer(train_df, FEATURE_COLUMNS)
    train_df = apply_standardizer(train_df, FEATURE_COLUMNS, feature_mean, feature_std)
    val_df = apply_standardizer(val_df, FEATURE_COLUMNS, feature_mean, feature_std)
    test_df = apply_standardizer(test_df, FEATURE_COLUMNS, feature_mean, feature_std)

    if progress_callback:
        progress_callback(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # 3. Train
    agent, logs_df = train_agent(train_df, val_df, cfg, progress_callback)

    # 4. Backtest on test set
    test_env = make_env(test_df, cfg)
    bt = backtest(test_env, agent)
    metrics = bt["metrics"]

    # 5. Save model (V2)
    model_path = cfg.outputs_dir / "mtf_dqn_model_v2.pth"
    agent.save(
        str(model_path),
        feature_columns=FEATURE_COLUMNS,
        feature_mean=feature_mean.to_dict(),
        feature_std=feature_std.to_dict(),
        config={k: v for k, v in cfg.__dict__.items() if not isinstance(v, type(cfg.project_dir))},
        action_position_ratios=ACTION_POSITION_RATIOS,
    )

    # 6. Save plots (V2)
    plt.figure(figsize=(10, 5))
    plt.plot(logs_df["episode"], logs_df["train_return"], label="Train Return")
    plt.plot(logs_df["episode"], logs_df["val_return"], label="Val Return")
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.title("MTF DQN Training / Validation Return (V2)")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(cfg.outputs_dir / "training_returns_v2.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(logs_df["episode"], logs_df["avg_loss"], label="Avg Loss")
    plt.title("DQN Training Loss (V2)")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(cfg.outputs_dir / "training_losses_v2.png", dpi=150)
    plt.close()

    print(f"Model saved to: {model_path}")
    print(f"Artifacts saved in: {cfg.outputs_dir}")

    return {
        "status": "success",
        "logs_df": logs_df,
        "model_path": str(model_path),
        "metrics": metrics,
        "agent": agent,
        "mtf_df": mtf_df,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "test_backtest": bt,
        "is_v2": True,
    }


def main() -> None:
    cfg = Config()
    result = run_training_pipeline_v2(cfg)
    metrics = result["metrics"]
    print("\n========== Test Backtest Metrics ==========")
    for k, v in metrics.items():
        if isinstance(v, float):
            if "rate" in k or "return" in k or "drawdown" in k:
                print(f"{k}: {v:.2%}")
            else:
                print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
