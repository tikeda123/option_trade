#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from enum import Enum

# sb3_contrib.MaskablePPO
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.wrappers import ActionMasker

# 追加で必要なインポート
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback


class Actions(Enum):
    HOLD = 0
    BUY1 = 1
    BUY5 = 2
    SELL1 = 3
    SELL5 = 4


class EnhancedTradingEnv(gym.Env):
    def __init__(self,
                 df: pd.DataFrame,
                 window_size: int = 300,
                 max_position_size: int = 10,
                 n_future_steps: int = 3):
        super().__init__()
        self.df = df.copy().reset_index(drop=True)
        # 移動平均例
        self.df["ma10"] = self.df["close"].rolling(10).mean().fillna(method="bfill")

        self.window_size = window_size
        self.data_size = len(self.df)
        self.max_position_size = max_position_size
        self.n_future_steps = n_future_steps

        # 観測空間: [現在価格, MA, 未来平均価格, ポジションサイズ]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )
        # アクション空間: HOLD, BUY1, BUY5, SELL1, SELL5
        self.action_space = spaces.Discrete(len(Actions))

        # 内部状態
        self.current_step = 0
        self.position_size = 0
        self.avg_entry_price = 0.0
        self.cumulative_profit = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.position_size = 0
        self.avg_entry_price = 0.0
        self.cumulative_profit = 0.0
        return self._get_observation(), {}

    def step(self, action: int):
        next_idx = min(self.current_step + 1, self.data_size - 1)
        next_price = float(self.df.loc[next_idx, "close"])
        realized_pnl = 0.0

        # アクションによるポジション操作
        if action == Actions.HOLD.value:
            pass
        elif action == Actions.BUY1.value:
            realized_pnl += self._execute_buy(1, next_price)
        elif action == Actions.BUY5.value:
            realized_pnl += self._execute_buy(5, next_price)
        elif action == Actions.SELL1.value:
            realized_pnl += self._execute_sell(1, next_price)
        elif action == Actions.SELL5.value:
            realized_pnl += self._execute_sell(5, next_price)

        self.cumulative_profit += realized_pnl

        # 報酬計算
        current_price = float(self.df.loc[next_idx, "close"])
        unrealized_pnl = (current_price - self.avg_entry_price) * self.position_size
        reward = realized_pnl + unrealized_pnl

        # ポジション保持ペナルティ (例)
        hold_penalty = 0.0
        if self.position_size > 0:
            hold_penalty = 0.01 * self.position_size
            reward -= hold_penalty

        self.current_step += 1
        terminated = (self.current_step >= self.window_size) or (next_idx >= self.data_size - 1)
        truncated = False

        info = {
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "hold_penalty": hold_penalty,
            "position_size": self.position_size,
            "avg_entry_price": self.avg_entry_price,
            "cumulative_profit": self.cumulative_profit,
        }

        return self._get_observation(), reward, terminated, truncated, info

    def _execute_buy(self, amount, price) -> float:
        if amount <= 0:
            return 0.0
        new_size = self.position_size + amount
        if new_size > self.max_position_size:
            # 超過分は買わずに、最大まで
            amount = self.max_position_size - self.position_size
            new_size = self.max_position_size
        if amount <= 0:
            return 0.0

        total_cost = self.avg_entry_price * self.position_size + price * amount
        self.position_size = new_size
        self.avg_entry_price = total_cost / self.position_size
        return 0.0

    def _execute_sell(self, amount, price) -> float:
        if self.position_size <= 0:
            return 0.0
        sell_amount = min(self.position_size, amount)
        realized_pnl = sell_amount * (price - self.avg_entry_price)
        self.position_size -= sell_amount
        if self.position_size == 0:
            self.avg_entry_price = 0.0
        return realized_pnl

    def _get_observation(self):
        idx = min(self.current_step, self.data_size - 1)
        price_now = float(self.df.loc[idx, "close"])
        ma_now = float(self.df.loc[idx, "ma10"])

        start_future = idx + 1
        end_future = min(idx + self.n_future_steps, self.data_size - 1)
        future_prices = self.df.loc[start_future:end_future, "close"].values
        if len(future_prices) == 0:
            future_mean_price = price_now
        else:
            future_mean_price = float(np.mean(future_prices))

        return np.array([
            price_now,
            ma_now,
            future_mean_price,
            float(self.position_size)
        ], dtype=np.float32)

    def get_action_mask(self) -> np.ndarray:
        """
        ポジションが最大サイズなら BUY 系を禁止
        ポジションが0なら SELL 系を禁止
        それ以外はすべて可能
        """
        mask = np.ones(self.action_space.n, dtype=np.int8)
        if self.position_size >= self.max_position_size:
            mask[Actions.BUY1.value] = 0
            mask[Actions.BUY5.value] = 0
        if self.position_size == 0:
            mask[Actions.SELL1.value] = 0
            mask[Actions.SELL5.value] = 0
        return mask


# 学習中にステップをプリントする簡易コールバック
class PrintCallback(BaseCallback):
    def __init__(self, verbose=1, print_freq=1000):
        super().__init__(verbose=verbose)
        self.print_freq = print_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.print_freq == 0:
            print(f"[Callback] Step={self.n_calls}")
        return True


def make_synthetic_data(num_steps=2000, seed=42):
    np.random.seed(seed)
    prices = [10000]
    for _ in range(num_steps - 1):
        change_pct = np.random.normal(loc=0.0, scale=0.01)
        prices.append(prices[-1] * (1 + change_pct))
    df = pd.DataFrame({"close": prices})
    return df


def main():
    # データ作成
    df = make_synthetic_data(num_steps=3000)

    # 生の環境を作成
    base_env = EnhancedTradingEnv(df, window_size=500, max_position_size=10, n_future_steps=3)

    # 1. Monitor でラップ (エピソードごとのリワードを monitor.csv に保存するなど)
    env = Monitor(base_env)

    # 2. ActionMasker でラップし、mask_fnでは必ず unwrapped にアクセス
    def mask_fn(env):
        return env.unwrapped.get_action_mask()

    env = ActionMasker(env, mask_fn)

    # ここでMaskablePPOを作成
    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=1,
        ent_coef=0.02,
        n_steps=2048,
        batch_size=64,
        seed=123
    )

    print("=== Start Training ===")
    model.learn(total_timesteps=30000, callback=PrintCallback())
    print("=== Training Finished ===")

    # 評価 (複数エピソード)
    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=5,
        deterministic=True,
        use_masking=True
    )
    print(f"[Evaluation] mean_reward={mean_reward:.4f}, std={std_reward:.4f}")

    # テスト：1エピソードを手動で実行
    print("\n=== Start Test Episode ===")
    obs, info = env.reset()
    done = False
    total_reward = 0.0
    step_count = 0
    action_counts = {act.name: 0 for act in Actions}

    while not done:
        # 明示的にマスクを取得（env.unwrappedで最内層へアクセス）
        current_mask = env.unwrapped.get_action_mask()

        action, _ = model.predict(
            obs,
            deterministic=True,
            action_masks=current_mask
        )

        obs, reward, terminated, truncated, info = env.step(action)
        action_counts[Actions(action).name] += 1
        total_reward += reward
        step_count += 1
        done = terminated or truncated

    print(f"[Test Episode] Steps={step_count}, total_reward={total_reward:.4f}")
    print(f"Final Cumulative Profit={env.unwrapped.cumulative_profit:.4f}")
    print(f"Action counts: {action_counts}")


if __name__ == "__main__":
    main()
