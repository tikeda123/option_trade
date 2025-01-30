#!/usr/bin/env python

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from enum import Enum

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

import matplotlib.pyplot as plt

# ========================
# Actions & Simple Env
# (旧API: 4要素で step)
# ========================
class Actions(Enum):
    HOLD = 0
    BUY1 = 1
    SELL1 = 2

class SimpleTradingEnv(gym.Env):
    def __init__(self, df, window_size=300, max_position_size=10,
                 hold_penalty=0.01, trade_bonus=1.0):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.data_size = len(self.df)
        self.max_position_size = max_position_size

        self.hold_penalty = hold_penalty
        self.trade_bonus = trade_bonus

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(len(Actions))

        self.current_step = 0
        self.position_size = 0
        self.avg_entry_price = 0.0
        self.cumulative_profit = 0.0
        self.consecutive_hold_steps = 0
        self.trade_log = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.position_size = 0
        self.avg_entry_price = 0.0
        self.cumulative_profit = 0.0
        self.consecutive_hold_steps = 0
        self.trade_log = []
        return self._get_observation(), {}

    def step(self, action):
        next_idx = min(self.current_step + 1, self.data_size - 1)
        next_price = float(self.df.loc[next_idx, "close"])

        realized_pnl = 0.0
        bonus = 0.0

        if action == Actions.HOLD.value:
            self.consecutive_hold_steps += 1
        else:
            self.consecutive_hold_steps = 0
            if action == Actions.BUY1.value:
                realized_pnl += self._execute_buy(1, next_price)
                bonus = self.trade_bonus
            elif action == Actions.SELL1.value:
                realized_pnl += self._execute_sell(1, next_price)
                bonus = self.trade_bonus

        self.cumulative_profit += realized_pnl

        current_price = float(self.df.loc[next_idx, "close"])
        unrealized_pnl = (current_price - self.avg_entry_price) * self.position_size

        penalty_for_hold = self.hold_penalty * self.consecutive_hold_steps
        reward = (realized_pnl + unrealized_pnl) - penalty_for_hold + bonus

        self.current_step += 1
        done = (self.current_step >= self.window_size) or (next_idx >= self.data_size - 1)

        info = {
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "position_size": self.position_size,
            "avg_entry_price": self.avg_entry_price,
            "cumulative_profit": self.cumulative_profit,
            "consecutive_hold_steps": self.consecutive_hold_steps,
            "action_bonus": bonus
        }

        self.trade_log.append({
            "step": self.current_step,
            "action": action,
            "price": current_price,
            "position_size": self.position_size,
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "cumulative_profit": self.cumulative_profit,
            "reward": reward,
            "consecutive_hold_steps": self.consecutive_hold_steps,
            "action_bonus": bonus
        })

        return self._get_observation(), reward, done, info

    def get_action_mask(self):
        mask = np.ones(self.action_space.n, dtype=np.int8)
        if self.position_size >= self.max_position_size:
            mask[Actions.BUY1.value] = 0
        if self.position_size == 0:
            mask[Actions.SELL1.value] = 0
        return mask

    def _get_observation(self):
        idx_now = min(self.current_step, self.data_size - 1)
        idx_next = min(self.current_step + 1, self.data_size - 1)
        price_now = float(self.df.loc[idx_now, "close"])
        price_next = float(self.df.loc[idx_next, "close"])
        return np.array([price_now, price_next, float(self.position_size)], dtype=np.float32)

    def _execute_buy(self, amount, price) -> float:
        if amount <= 0:
            return 0.0
        new_size = self.position_size + amount
        if new_size > self.max_position_size:
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

# ========================
# 環境をマスク付きで作成
# ========================
def mask_fn(env):
    return env.unwrapped.get_action_mask()

def make_synthetic_data(num_steps=1000, seed=42):
    np.random.seed(seed)
    prices = []
    for i in range(num_steps):
        price = 1000 + 100 * np.sin(2.0 * np.pi * i / 50)
        prices.append(price)
    return pd.DataFrame({"close": prices})

# -------------------------------------------------------
# コールバック (旧API: 4要素)
# -------------------------------------------------------
class RandomInitCallback(BaseCallback):
    """
    学習開始直後にランダム行動を挟む (旧API対応: step() -> 4要素)
    """
    def __init__(self, random_steps=1000, verbose=0):
        super().__init__(verbose=verbose)
        self.random_steps = random_steps

    def _on_step(self) -> bool:
        if self.model.num_timesteps < self.random_steps:
            # n_envs=1 前提。actionをリスト化
            action_sample = self.training_env.action_space.sample()
            action_array = [action_sample]

            obs, reward, done, info = self.training_env.step(action_array)
            if done[0]:
                self.training_env.reset()
            return False
        return True

class PrintCallback(BaseCallback):
    def __init__(self, verbose=1, print_freq=5000):
        super().__init__(verbose=verbose)
        self.print_freq = print_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.print_freq == 0:
            print(f"[Callback] Step={self.n_calls}, Timesteps={self.model.num_timesteps}")
        return True

def main():
    # ---------------------
    # Step1: シングルEnvにMasker適用 → DummyVecEnv化
    # ---------------------
    df = make_synthetic_data()

    def make_env():
        def _init():
            env = SimpleTradingEnv(df=df, window_size=300)
            env = Monitor(env)
            # ActionMasker は「単一Env」をラップする
            env = ActionMasker(env, mask_fn)
            return env
        return _init

    # n_envs=1
    vec_env = DummyVecEnv([make_env()])

    # ---------------------
    # Step2: 学習
    # ---------------------
    model = MaskablePPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=3e-4,
        ent_coef=0.5,
        n_steps=1024,
        batch_size=64,
        seed=123,
        policy_kwargs=dict(net_arch=[256, 256])
    )

    print("=== Start Training ===")
    model.learn(
        total_timesteps=50000,
        callback=[RandomInitCallback(random_steps=3000), PrintCallback(print_freq=5000)]
    )
    print("=== Training Finished ===")

    # ---------------------
    # Step3: 評価
    # ---------------------
    from sb3_contrib.common.maskable.evaluation import evaluate_policy
    mean_reward, std_reward = evaluate_policy(
        model,
        vec_env,
        n_eval_episodes=5,
        deterministic=True,
        use_masking=True
    )
    print(f"[Evaluation] mean_reward={mean_reward:.4f}, std={std_reward:.4f}")

    # ---------------------
    # Step4: テスト (旧API: 4要素)
    # ---------------------
    print("\n=== Start Test Episode ===")
    obs, info = vec_env.reset()
    done = [False]
    step_count = 0
    total_reward = 0.0

    action_counts = {act.name: 0 for act in Actions}

    while not done[0]:
        # Maskはenvごとに (n_envs=1 → index=0)
        current_mask = vec_env.envs[0].unwrapped.get_action_mask()

        # predict() は (n_envs, ) 形のアクション返却
        # maskは [current_mask] の形で渡す
        action, _ = model.predict(obs, deterministic=True, action_masks=[current_mask])

        obs, reward, done, info = vec_env.step(action)
        action_counts[Actions(action[0]).name] += 1
        total_reward += reward[0]
        step_count += 1

    # 強制クローズ
    unwrapped = vec_env.envs[0].unwrapped
    if unwrapped.position_size != 0:
        final_price = float(unwrapped.df.loc[unwrapped.current_step - 1, "close"])
        forced_pnl = unwrapped._execute_sell(unwrapped.position_size, final_price)
        total_reward += forced_pnl
        unwrapped.cumulative_profit += forced_pnl
        print(f"[Force Close] position_size -> 0, realized_pnl={forced_pnl:.4f}")

    print(f"[Test Episode] Steps={step_count}, total_reward={total_reward:.4f}")
    print(f"Final Cumulative Profit={unwrapped.cumulative_profit:.4f}")
    print(f"Action counts: {action_counts}")

    # ログ可視化
    trade_df = pd.DataFrame(unwrapped.trade_log)
    print("\n--- Trade Log (head) ---")
    print(trade_df.head())

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()
    ax1.set_title("Price and Position Over Time")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Price", color="blue")
    ax2.set_ylabel("Position Size", color="red")

    ax1.plot(trade_df["step"], trade_df["price"], color="blue", label="Price")
    ax2.plot(trade_df["step"], trade_df["position_size"], color="red", label="Position Size")

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()


