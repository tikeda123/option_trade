# dqn_trade/simple_btc_env.py
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from enum import Enum

class Actions(Enum):
    HOLD = 0
    BUY1 = 1
    BUY5 = 2
    BUY10 = 3
    SELL_ALL = 4
    SELL1 = 5
    SELL5 = 6
    SELL10 = 7

class BaseTradingEnv(gym.Env):
    def __init__(
        self,
        df,
        window_size=300,
        fee_ratio=0.0001,
        random_start=True,
        debug=False,

        use_reward_clip=False,
        clip_positive=1.0,
        clip_negative=-1.0,

        no_pos_penalty=-0.001,
        no_pos_sell_penalty=-1.0,

        pnl_scale_factor=1.0,
        time_hold_penalty=-0.01,
        final_close_penalty_factor=0.5,
        max_hold_count=99999,
        max_position_steps=300,

        partial_close_bonus=0.0,
        rebuy_penalty=0.0,

        # ★ 追加
        disallow_excess_buy=False,  # Trueの場合、最大ロット(10)を超えるBUYを無効化
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.df["ma10"] = self.df["close"].rolling(10).mean().fillna(method="bfill")

        self.data_size = len(self.df)
        self.window_size = window_size
        self.fee_ratio = fee_ratio
        self.random_start = random_start
        self.debug = debug

        self.use_reward_clip = use_reward_clip
        self.clip_positive = clip_positive
        self.clip_negative = clip_negative

        self.no_pos_penalty = no_pos_penalty
        self.no_pos_sell_penalty = no_pos_sell_penalty

        self.pnl_scale_factor = pnl_scale_factor
        self.time_hold_penalty = time_hold_penalty
        self.final_close_penalty_factor = final_close_penalty_factor
        self.max_hold_count = max_hold_count
        self.max_position_steps = max_position_steps

        self.partial_close_bonus = partial_close_bonus
        self.rebuy_penalty = rebuy_penalty
        self.disallow_excess_buy = disallow_excess_buy

        if self.data_size < self.window_size:
            raise ValueError(
                f"Not enough data: data_size={self.data_size}, window_size={self.window_size}."
            )

        self._setup_spaces()

    def _setup_spaces(self):
        # 観測空間: [log_price, log_ma10, position_size]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(len(Actions))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.random_start:
            max_start = self.data_size - self.window_size + 1
            self.start_step = self.np_random.integers(low=0, high=max_start)
        else:
            self.start_step = 0

        self._reset_internal_states()
        obs = self._get_observation()
        info = {}
        if self.debug:
            print(f"[RESET] start_step={self.start_step}, current_step={self.current_step}")
        return obs, info

    def _reset_internal_states(self):
        self.current_step = self.start_step
        self.position_size = 0
        self.entry_price = 0.0
        self.prev_price = float(self.df.loc[self.current_step, "close"])
        self.cumulative_reward = 0.0
        self.action_counts = {
            "BUY1": 0, "BUY5": 0, "BUY10": 0,
            "SELL_ALL": 0, "SELL1": 0, "SELL5": 0, "SELL10": 0,
            "HOLD": 0,
        }
        self.position_steps = 0

    def step(self, action):
        """
        修正ポイント:
        ノーポジでSELL系アクションを選んだら強制的にHOLDへ、
        ポジション保有中なのにBUY系アクションを選んだらHOLDへ置き換え。
        """
        # --- 行動制限ロジックここから ---
        if self.position_size == 0:
            # ノーポジ時 → SELL系(SELL_ALL, SELL1, SELL5, SELL10)を選べない
            if action in [Actions.SELL_ALL.value, Actions.SELL1.value,
                          Actions.SELL5.value, Actions.SELL10.value]:
                action = Actions.HOLD.value
        else:
            # ポジション保有時 → BUY系(BUY1, BUY5, BUY10)を選べない
            if action in [Actions.BUY1.value, Actions.BUY5.value, Actions.BUY10.value]:
                action = Actions.HOLD.value
        # --- 行動制限ロジックここまで ---

        if self.current_step >= self.start_step + self.window_size:
            obs = self._get_observation_safe()
            reward = 0.0
            terminated = True
            truncated = False
            info = {
                "info": "Episode finished (out_of_range)",
                "action_counts": self.action_counts
            }
            return obs, reward, terminated, truncated, info

        current_price = float(self.df.loc[self.current_step, "close"])

        # reward_breakdownで報酬の内訳を管理
        reward_breakdown = {
            "action_reward": 0.0,
            "hold_penalty": 0.0,
            "forced_close_penalty": 0.0,
            "partial_close_bonus": 0.0,
        }

        # アクションによる報酬
        action_reward = self._take_action(action, current_price)
        reward_breakdown["action_reward"] = action_reward
        reward = action_reward

        # 保有中のペナルティや強制クローズ
        terminated = False
        if self.position_size > 0:
            # HOLDペナルティ
            hold_pen = self.time_hold_penalty
            reward += hold_pen
            reward_breakdown["hold_penalty"] = hold_pen

            self.position_steps += 1
            if self.position_steps > self.max_position_steps:
                # 最大保有期間超過による強制クローズ
                forced_close_reward = self._finalize_position(current_price, forced=True)
                reward_breakdown["forced_close_penalty"] = forced_close_reward
                reward += forced_close_reward
                terminated = True
        else:
            self.position_steps = 0

        truncated = False
        self.current_step += 1
        self.prev_price = current_price

        # ウィンドウ終端なら終了 & ポジション清算
        if (not terminated) and (self.current_step >= self.start_step + self.window_size):
            terminated = True
            if self.position_size > 0:
                final_price = float(self.df.loc[self.current_step - 1, "close"])
                forced_close_reward = self._finalize_position(final_price, forced=True)
                reward_breakdown["forced_close_penalty"] = forced_close_reward
                reward += forced_close_reward

        self.cumulative_reward += reward

        obs = self._get_observation_safe()

        # infoに細かいデバッグ情報を詰め込む
        info = {
            "current_step": self.current_step,
            "current_price": current_price,
            "entry_price": self.entry_price,
            "position_size": self.position_size,
            "action_name": Actions(action).name,
            "reward_breakdown": reward_breakdown,
            "cumulative_reward": self.cumulative_reward,
            "unrealized_pnl": self._calc_unrealized_pnl(current_price),
        }

        if terminated or truncated:
            info["action_counts"] = self.action_counts

        if self.debug:
            print("[STEP]", {
                "current_step": self.current_step,
                "action": action,
                "action_name": Actions(action).name,
                "current_price": current_price,
                "position_size": self.position_size,
                "step_reward": reward,
                "cumulative_reward": self.cumulative_reward,
                "terminated": terminated,
                "position_steps": self.position_steps,
                "reward_breakdown": reward_breakdown,
            })

        return obs, reward, terminated, truncated, info

    def _take_action(self, action, current_price):
        # アクションカウント計上
        if action == Actions.HOLD.value:
            self.action_counts["HOLD"] += 1
            return self._handle_hold_action()

        elif action == Actions.BUY1.value:
            self.action_counts["BUY1"] += 1
            return self._handle_buy_action(current_price, lot=1)

        elif action == Actions.BUY5.value:
            self.action_counts["BUY5"] += 1
            return self._handle_buy_action(current_price, lot=5)

        elif action == Actions.BUY10.value:
            self.action_counts["BUY10"] += 1
            return self._handle_buy_action(current_price, lot=10)

        elif action == Actions.SELL_ALL.value:
            self.action_counts["SELL_ALL"] += 1
            return self._handle_sell_action(current_price, sell_all=True)

        elif action == Actions.SELL1.value:
            self.action_counts["SELL1"] += 1
            return self._handle_sell_action(current_price, sell_all=False, lot=1)

        elif action == Actions.SELL5.value:
            self.action_counts["SELL5"] += 1
            return self._handle_sell_action(current_price, sell_all=False, lot=5)

        elif action == Actions.SELL10.value:
            self.action_counts["SELL10"] += 1
            return self._handle_sell_action(current_price, sell_all=False, lot=10)

        return 0.0

    def _handle_buy_action(self, current_price, lot=1):
        """
        - 既にポジションがある状態での再BUYをペナルティ化、あるいは
          disallow_excess_buy によってロット上限を超えたBUYを無効化
        """
        if self.position_size > 0:
            return self._reward_clip(self.rebuy_penalty)

        if self.disallow_excess_buy:
            if self.position_size + lot > 10:
                return -5.0  # 大きめ固定ペナルティ(例)

        cost = self._apply_fee(current_price, lot)
        self.position_size = lot
        self.entry_price = current_price
        self.position_steps = 1

        return -cost * 0.001

    def _handle_sell_action(self, current_price, sell_all=True, lot=1):
        if self.position_size == 0:
            # ノーポジでSELL (ただし今回の改修ではstep()段階でほぼHOLDに置換される想定)
            return self._reward_clip(self.no_pos_sell_penalty)

        if sell_all:
            return self._finalize_position(current_price, forced=False)
        else:
            if self.position_size >= lot:
                return self._finalize_position(current_price, forced=False, portion=lot)
            else:
                # lot指定より保有数が少ない場合は残り全てを売却
                return self._finalize_position(current_price, forced=False)

    def _handle_hold_action(self):
        if self.position_size == 0:
            return self._reward_clip(self.no_pos_penalty)
        else:
            return 0.0

    def _finalize_position(self, final_price, forced=False, portion=None):
        if portion is None:
            portion = self.position_size

        old_pos = self.position_size
        lot = portion
        cost = self._apply_fee(final_price, lot)
        raw_pnl = (final_price - self.entry_price) * lot - cost

        # PnLスケーリング
        scaled_pnl = raw_pnl * self.pnl_scale_factor

        # 強制クローズならペナルティを乗算
        if forced:
            scaled_pnl *= self.final_close_penalty_factor

        # クリップ処理
        clipped_reward = self._pnl_to_clip_reward(scaled_pnl)

        # ポジションサイズを減らす (部分決済ならボーナス加算)
        self.position_size -= lot
        if self.position_size == 0:
            self.entry_price = 0.0
            self.position_steps = 0
        else:
            # 部分決済の場合はボーナス加算
            if lot < old_pos:
                clipped_reward += self.partial_close_bonus

        return clipped_reward

    def _pnl_to_clip_reward(self, raw_pnl: float) -> float:
        if not self.use_reward_clip:
            return raw_pnl
        else:
            return np.clip(raw_pnl, self.clip_negative, self.clip_positive)

    def _apply_fee(self, price, lot):
        return price * self.fee_ratio * lot

    def _reward_clip(self, val: float):
        if self.use_reward_clip:
            return np.clip(val, self.clip_negative, self.clip_positive)
        else:
            return val

    def _get_observation(self):
        return self._make_observation(self.current_step)

    def _get_observation_safe(self):
        if 0 <= self.current_step < self.data_size:
            return self._get_observation()
        else:
            temp_step = self.data_size - 1
            return self._make_observation(temp_step)

    def _make_observation(self, step_index):
        price = float(self.df.loc[step_index, "close"])
        ma10  = float(self.df.loc[step_index, "ma10"])
        log_price = np.log(price + 1e-8)
        log_ma10  = np.log(ma10 + 1e-8)
        return np.array([log_price, log_ma10, float(self.position_size)], dtype=np.float32)

    def _calc_unrealized_pnl(self, current_price):
        if self.position_size <= 0:
            return 0.0
        raw_pnl = (current_price - self.entry_price) * self.position_size
        return raw_pnl * self.pnl_scale_factor


class SimpleBTCEnv(BaseTradingEnv):
    pass
