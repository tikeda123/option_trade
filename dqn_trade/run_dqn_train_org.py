# dqn_trade/run_dqn_train.py
# dqn_trade/run_dqn_train.py
import numpy as np
import pandas as pd

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

from data_fetch import fetch_data
from simple_btc_env import SimpleBTCEnv  # ← 改修後のクラスをインポート


def fetch_and_split_data(df_ratio: float = 0.8, randomize_price: bool = True):
    """
    改善策 3.5「より多様な相場データで学習する」ための例として、
    取得したデータに対してランダムシフトやスケーリングを挟む処理を追加。
    """
    df = fetch_data()

    # --- ここで、たとえばランダムに数%上下のスケールを加える例 ---
    if randomize_price:
        scale_factor = np.random.uniform(0.8, 1.2)  # 0.8～1.2倍など
        shift_factor = np.random.uniform(-50, 50)   # -50～50ドルの平行移動など
        df["close"] = df["close"] * scale_factor + shift_factor
        # 必要に応じて high, low, open などにも同様に適用

    split_idx = int(len(df) * df_ratio)
    df_train = df.iloc[:split_idx].copy().reset_index(drop=True)
    df_test = df.iloc[split_idx:].copy().reset_index(drop=True)
    return df_train, df_test


def create_btc_env(
    df: pd.DataFrame,
    window_size: int,
    fee_ratio: float = 0.0001,
    random_start: bool = False,
    debug: bool = False,

    # 報酬クリップ関連のパラメータ
    use_reward_clip=False,
    clip_positive=1.0,
    clip_negative=-1.0,
    no_pos_sell_penalty=-1.0,
    no_pos_penalty=-0.1,

    # 追加のパラメータ
    pnl_scale_factor=1.0,
    time_hold_penalty=-0.01,
    final_close_penalty_factor=0.5,

    # ★ 改修で追加
    partial_close_bonus=0.0,
    rebuy_penalty=-0.0,
    disallow_excess_buy=False,  # ← 既に10Lot保有なら BUY10 を無効化するかどうか
):
    env = SimpleBTCEnv(
        df=df,
        window_size=window_size,
        fee_ratio=fee_ratio,
        random_start=random_start,
        debug=debug,
        use_reward_clip=use_reward_clip,
        clip_positive=clip_positive,
        clip_negative=clip_negative,
        no_pos_sell_penalty=no_pos_sell_penalty,
        no_pos_penalty=no_pos_penalty,

        pnl_scale_factor=pnl_scale_factor,
        time_hold_penalty=time_hold_penalty,
        final_close_penalty_factor=final_close_penalty_factor,

        max_hold_count=99999,
        max_position_steps=300,

        partial_close_bonus=partial_close_bonus,
        rebuy_penalty=rebuy_penalty,
        disallow_excess_buy=disallow_excess_buy,
    )
    return Monitor(env)


def make_env(
    df: pd.DataFrame,
    window_size: int,
    fee_ratio: float,
    random_start: bool,
    debug: bool,
    use_reward_clip: bool,
    clip_positive: float,
    clip_negative: float,
    no_pos_sell_penalty: float,
    no_pos_penalty: float,
    pnl_scale_factor: float,
    time_hold_penalty: float,
    final_close_penalty_factor: float,

    # ★ 改修で追加
    partial_close_bonus: float,
    rebuy_penalty: float,
    disallow_excess_buy: bool,
):
    def _init():
        return create_btc_env(
            df=df,
            window_size=window_size,
            fee_ratio=fee_ratio,
            random_start=random_start,
            debug=debug,
            use_reward_clip=use_reward_clip,
            clip_positive=clip_positive,
            clip_negative=clip_negative,
            no_pos_sell_penalty=no_pos_sell_penalty,
            no_pos_penalty=no_pos_penalty,
            pnl_scale_factor=pnl_scale_factor,
            time_hold_penalty=time_hold_penalty,
            final_close_penalty_factor=final_close_penalty_factor,

            partial_close_bonus=partial_close_bonus,
            rebuy_penalty=rebuy_penalty,
            disallow_excess_buy=disallow_excess_buy,
        )
    return _init


def create_vectorized_envs(
    df: pd.DataFrame,
    window_size: int,
    fee_ratio: float = 0.0001,
    random_start: bool = True,
    debug: bool = False,
    use_reward_clip: bool = False,
    clip_positive: float = 1.0,
    clip_negative: float = -1.0,
    no_pos_sell_penalty: float = -1.0,
    no_pos_penalty: float = -0.1,
    pnl_scale_factor: float = 1.0,
    time_hold_penalty: float = -0.01,
    final_close_penalty_factor: float = 0.5,
    num_envs: int = 4,
    use_subproc: bool = False,

    # ★ 改修で追加
    partial_close_bonus: float = 0.0,
    rebuy_penalty: float = 0.0,
    disallow_excess_buy: bool = False,
):
    env_fns = [
        make_env(
            df=df,
            window_size=window_size,
            fee_ratio=fee_ratio,
            random_start=random_start,
            debug=debug,
            use_reward_clip=use_reward_clip,
            clip_positive=clip_positive,
            clip_negative=clip_negative,
            no_pos_sell_penalty=no_pos_sell_penalty,
            no_pos_penalty=no_pos_penalty,
            pnl_scale_factor=pnl_scale_factor,
            time_hold_penalty=time_hold_penalty,
            final_close_penalty_factor=final_close_penalty_factor,

            partial_close_bonus=partial_close_bonus,
            rebuy_penalty=rebuy_penalty,
            disallow_excess_buy=disallow_excess_buy,
        )
        for _ in range(num_envs)
    ]
    if use_subproc:
        return SubprocVecEnv(env_fns)
    else:
        return DummyVecEnv(env_fns)


def train_ppo_model(
    train_env: VecEnv,
    learning_rate: float = 3e-4,
    n_steps: int = 1024,
    batch_size: int = 64,
    n_epochs: int = 10,
    total_timesteps: int = 300_000,
    gae_lambda: float = 0.90,
    clip_range: float = 0.2,
    gamma: float = 0.95,

    # ★ 改修：探索を増やすため ent_coef を追加
    ent_coef: float = 0.0,
):
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        gamma=gamma,
        verbose=1,
        tensorboard_log="./tb_logs/",
        ent_coef=ent_coef,  # ← 追加
    )
    model.learn(total_timesteps=total_timesteps)
    return model


def evaluate_ppo_model(
    model: PPO,
    env: gym.Env,
    n_eval_episodes: int = 3,
    deterministic: bool = True
):
    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=n_eval_episodes,
        deterministic=deterministic
    )
    print(f"[Evaluation] mean_reward={mean_reward:.6f} +/- {std_reward:.6f}")


def run_debug_test(model: PPO, df_test: pd.DataFrame):
    debug_test_env = create_btc_env(
        df=df_test,
        window_size=len(df_test),
        fee_ratio=0.0001,
        random_start=False,
        debug=True,

        use_reward_clip=False,
        clip_positive=1.0,
        clip_negative=-1.0,
        no_pos_sell_penalty=-1.0,
        no_pos_penalty=-0.1,

        # 改修後のパラメータ例
        pnl_scale_factor=1.0,
        time_hold_penalty=-0.2,          # HOLDコストをさらに増やす例 (-0.2)
        final_close_penalty_factor=0.01, # 強制クローズ利益を 1% に圧縮
        partial_close_bonus=0.2,         # 部分決済ボーナス大
        rebuy_penalty=-5.0,             # 大きめ再BUYペナルティ
        disallow_excess_buy=True,       # 10Lot保有中の BUY10 は無効化
    )

    obs, info = debug_test_env.reset()
    terminated, truncated = False, False
    total_reward = 0.0

    print("\n[DEBUG TEST SIMULATION] Start step-by-step logging...\n")
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = debug_test_env.step(action)
        total_reward += reward

    print(f"\n[DEBUG TEST SIMULATION] Final PnL (Cumulative Reward) = {total_reward:.4f}\n")
    action_counts = info.get("action_counts", None)
    if action_counts is not None:
        print("[DEBUG TEST SIMULATION] Action counts:", action_counts)


def main():
    # --- 3.5 相場をランダムにシフト・スケールさせる例として randomize_price=True で呼び出す ---
    df_train, df_test = fetch_and_split_data(df_ratio=0.8, randomize_price=True)
    print(f"Train data size: {len(df_train)}, Test data size: {len(df_test)}")

    # 学習用のベクトル化環境
    train_env = create_vectorized_envs(
        df=df_train,
        window_size=300,
        fee_ratio=0.0001,
        random_start=True,
        debug=False,

        use_reward_clip=False,
        clip_positive=5.0,
        clip_negative=-5.0,
        no_pos_sell_penalty=-1.0,
        no_pos_penalty=-0.1,

        # 改修パラメータ例
        pnl_scale_factor=1.0,

        # 3.4: HOLDペナルティを強化
        time_hold_penalty=-0.2,

        # 3.1: 最終決済ペナルティをさらに大きく (強制クローズ利益を削る)
        final_close_penalty_factor=0.01,

        # 3.3: 部分決済ボーナスを増やす
        partial_close_bonus=0.2,

        # 3.2: 再BUYペナルティを強める
        rebuy_penalty=-5.0,

        # 3.2 (別解): 10Lot保有中の BUY10 を無効化
        disallow_excess_buy=True,

        num_envs=4,
        use_subproc=False
    )

    # テスト環境
    test_env = create_btc_env(
        df=df_test,
        window_size=len(df_test),
        fee_ratio=0.0001,
        random_start=False,
        debug=False,

        use_reward_clip=False,
        clip_positive=5.0,
        clip_negative=-5.0,
        no_pos_sell_penalty=-1.0,
        no_pos_penalty=-0.1,

        pnl_scale_factor=1.0,
        time_hold_penalty=-0.2,
        final_close_penalty_factor=0.01,
        partial_close_bonus=0.2,
        rebuy_penalty=-5.0,
        disallow_excess_buy=True,
    )

    # PPOモデルを学習
    # 3.6: gammaを下げて早期報酬を重視し、ent_coefを上げて探索を促す
    model = train_ppo_model(
        train_env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        total_timesteps=300_000,
        gae_lambda=0.90,
        clip_range=0.2,

        gamma=0.85,     # 早期利確を重視 (さらに下げるなら0.8など)
        ent_coef=0.05,  # 行動の多様性を高める
    )

    # 評価
    evaluate_ppo_model(model, test_env, n_eval_episodes=3, deterministic=True)

    # デバッグ実行
    run_debug_test(model, df_test)


if __name__ == "__main__":
    main()

