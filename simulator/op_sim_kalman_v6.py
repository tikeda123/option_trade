import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from scipy.linalg import inv

# Matplotlibのフォント設定（必要に応じて）
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


###############################################################################
# 1) ExtendedKalmanFilter: EKF汎用基底クラス（ドメイン非依存）
###############################################################################
class ExtendedKalmanFilter:
    """
    ドメインに依存しない汎用的な拡張カルマンフィルタ基底クラス。
    具体的な状態遷移方程式 f, 観測方程式 h は継承先で実装する。
    """
    def __init__(
        self,
        state_dim,
        observation_dim,
        observation_covariance=1.0,
        process_noise=0.01
    ):
        """
        Parameters:
        -----------
        state_dim : int
            状態ベクトルの次元数
        observation_dim : int
            観測ベクトルの次元数
        observation_covariance : float
            観測ノイズ共分散のスケール（単位行列に乗じる形）
        process_noise : float
            状態遷移ノイズ共分散のスケール（同上）
        """
        self.state_dim = state_dim
        self.observation_dim = observation_dim
        self.observation_covariance = observation_covariance
        self.process_noise = process_noise

        # 初期状態
        self.initial_state_mean = np.zeros(self.state_dim)
        self.initial_state_covariance = np.eye(self.state_dim)

        # プロセスノイズ共分散
        self.process_covariance = np.eye(self.state_dim) * self.process_noise

    def f(self, x, dt=1.0):
        """
        状態遷移関数（非線形）。
        継承先で具体的に実装。
        """
        raise NotImplementedError

    def F_jacobian(self, x, dt=1.0):
        """
        状態遷移関数 f(x) のJacobian。
        継承先で具体的に実装。
        """
        raise NotImplementedError

    def h(self, x):
        """
        観測関数（非線形または線形）。
        継承先で具体的に実装。
        """
        raise NotImplementedError

    def H_jacobian(self, x):
        """
        観測関数 h(x) のJacobian。
        継承先で具体的に実装。
        """
        raise NotImplementedError

    def filter(self, observations, dt=1.0):
        """
        拡張カルマンフィルタを実行する。

        Parameters
        ----------
        observations : np.ndarray
            観測データ。 shape = (N, observation_dim)
        dt : float
            時間ステップ（デフォルトは1）

        Returns
        -------
        filtered_states : np.ndarray
            推定された状態列。 shape = (N, state_dim)
        """
        n = len(observations)
        x = self.initial_state_mean.copy()
        P = self.initial_state_covariance.copy()

        filtered_states = np.zeros((n, self.state_dim))

        for t in range(n):
            # 予測ステップ
            F = self.F_jacobian(x, dt)
            x_pred = self.f(x, dt)
            P_pred = F @ P @ F.T + self.process_covariance

            # 観測更新ステップ
            z = observations[t]
            H = self.H_jacobian(x_pred)
            y = z - self.h(x_pred)
            R = np.eye(self.observation_dim) * self.observation_covariance
            S = H @ P_pred @ H.T + R
            K = P_pred @ H.T @ inv(S)

            x = x_pred + K @ y
            P = (np.eye(self.state_dim) - K @ H) @ P_pred

            filtered_states[t] = x

        return filtered_states


###############################################################################
# 2) BTCTrendEKF: BTCに特化した状態遷移・観測方程式
###############################################################################
class BTCTrendEKF(ExtendedKalmanFilter):
    """
    BTCなどの価格を扱う際の状態ベクトル:
    x[0] = log_price
    x[1] = log_velocity
    x[2] = log_acceleration
    x[3] = composite_indicator (観測用)
    """
    def __init__(
        self,
        observation_covariance=1.0,
        process_noise=0.01,
        damping=0.1
    ):
        super().__init__(
            state_dim=4,
            observation_dim=2,  # [log_price, composite_indicator] を観測
            observation_covariance=observation_covariance,
            process_noise=process_noise
        )
        self.damping = damping

    def f(self, x, dt=1.0):
        """
        非線形状態遷移:
        x[0] = log_price, x[1] = log_velocity, x[2] = log_acceleration, x[3] = indicator
        """
        log_price = x[0]
        log_velocity = x[1]
        log_acceleration = x[2]
        indicator = x[3]

        next_log_price = log_price + log_velocity*dt + 0.5*log_acceleration*(dt**2)
        next_log_velocity = log_velocity + log_acceleration*dt
        # 減衰をかける
        next_log_acceleration = log_acceleration * (1 - self.damping)
        # indicatorは変化しない
        next_indicator = indicator

        return np.array([
            next_log_price,
            next_log_velocity,
            next_log_acceleration,
            next_indicator
        ])

    def F_jacobian(self, x, dt=1.0):
        """
        状態遷移関数 f(x) のJacobian (4x4)
        """
        F = np.array([
            [1,      dt,    0.5*(dt**2),  0],
            [0,      1,     dt,           0],
            [0,      0,     1 - self.damping, 0],
            [0,      0,     0,            1]
        ])
        return F

    def h(self, x):
        """
        観測関数:
        観測は [log_price, composite_indicator]
        """
        log_price = x[0]
        indicator = x[3]
        return np.array([log_price, indicator])

    def H_jacobian(self, x):
        """
        観測関数 h(x) のJacobian (2x4)
        """
        H = np.array([
            [1, 0, 0, 0],  # log_priceのみ
            [0, 0, 0, 1]   # indicatorのみ
        ])
        return H


###############################################################################
# 3) インジケータ作成ユーティリティ
###############################################################################
def prepare_composite_indicator(df):
    """
    DataFrameから複数列を取り出し、標準化して平均した複合インジケータを作成する。
    """
    indicators = pd.DataFrame(index=df.index)
    available_indicators = []

    # 例: close 価格
    if 'close' in df.columns:
        indicators['close'] = df['close']
        available_indicators.append('close')
        # サンプル: 14日間の価格変化率
        indicators['price_momentum'] = df['close'].pct_change(periods=14)
        available_indicators.append('price_momentum')

    # 他の指標も例示
    if 'roc' in df.columns:
        indicators['roc'] = df['roc']
        available_indicators.append('roc')

    if 'rsi' in df.columns:
        indicators['rsi'] = df['rsi']
        available_indicators.append('rsi')
        # rsiの変化率など
        indicators['rsi_slope'] = df['rsi'].diff(3)
        available_indicators.append('rsi_slope')

    if 'macdhist' in df.columns:
        indicators['macdhist'] = df['macdhist']
        available_indicators.append('macdhist')

    if 'mfi' in df.columns:
        indicators['mfi'] = df['mfi']
        available_indicators.append('mfi')

    if 'volatility' in df.columns:
        indicators['volatility'] = df['volatility']
        available_indicators.append('volatility')

    # 欠損値のある行を除去
    indicators.dropna(inplace=True)

    # 各指標を標準化
    normalized = pd.DataFrame(index=indicators.index)
    for col in available_indicators:
        std_val = indicators[col].std()
        if std_val > 1e-10:
            normalized[col] = (indicators[col] - indicators[col].mean()) / std_val
        else:
            normalized[col] = 0.0

    # 平均値を1つの複合インジケータに
    indicators['composite_indicator'] = normalized.mean(axis=1)

    return indicators


###############################################################################
# 4) トレンドを分類するロジック
###############################################################################
def classify_trend(log_velocity, log_acceleration, threshold=0.02):
    """
    log_velocity, log_acceleration からトレンド状態を返す。
    """
    if log_velocity > threshold and log_acceleration >= 0:
        return 'Uptrend'
    elif log_velocity < -threshold and log_acceleration <= 0:
        return 'Downtrend'
    elif abs(log_velocity) < threshold * 0.5:
        return 'Range-bound'
    elif log_velocity > 0:
        # velocity > 0かつ加速度が大きくマイナスなら「上昇失速」
        if log_acceleration < -threshold * 2:
            return 'Slowing Uptrend'
        else:
            return 'Weak Uptrend'
    else:
        # velocity < 0かつ加速度が大きくプラスなら「下降失速」
        if log_acceleration > threshold * 2:
            return 'Slowing Downtrend'
        else:
            return 'Weak Downtrend'

def classify_trends_from_states(filtered_states, threshold=0.02):
    """
    フィルタリング後の状態列（[log_price, log_velocity, log_acceleration, indicator]）から
    トレンド状態を判定してリストを返す。
    """
    log_velocity = filtered_states[:, 1]
    log_acceleration = filtered_states[:, 2]

    trend_states = []
    for vel, acc in zip(log_velocity, log_acceleration):
        trend = classify_trend(vel, acc, threshold)
        trend_states.append(trend)
    return trend_states


###############################################################################
# 5) トレンド検出の関数 (EKF適用 + 分類)
###############################################################################
def detect_trend(df, ekf=None, threshold=0.02):
    """
    DataFrame df に対してEKFを適用し、トレンド分類した結果を返す。

    Parameters
    ----------
    df : pd.DataFrame
        'close' や各種インジケータ列を含むDataFrame（IndexはDatetime）
    ekf : ExtendedKalmanFilter
        既存のEKFインスタンスを渡す場合。指定がなければBTCTrendEKFを生成。
    threshold : float
        log_velocity のしきい値。
    """
    if ekf is None:
        ekf = BTCTrendEKF(
            observation_covariance=1.0,
            process_noise=0.01,
            damping=0.1
        )

    indicators = prepare_composite_indicator(df)
    close = indicators['close']
    composite = indicators['composite_indicator']

    # log変換した価格と複合インジケータを観測データとして使う
    log_close = np.log(close)
    obs_data = np.column_stack([log_close.values, composite.values])

    # EKFの初期状態の設定（必要なら調整）
    ekf.initial_state_mean = np.array([
        log_close.iloc[0],
        0.0,                # 初期のvelocity
        0.0,                # 初期のacceleration
        composite.iloc[0]   # 初期のindicator
    ])
    ekf.initial_state_covariance = np.eye(4)

    # フィルタリング実行
    filtered_states = ekf.filter(obs_data, dt=1.0)

    # 推定結果を分解
    filtered_log_price        = filtered_states[:, 0]
    filtered_log_velocity     = filtered_states[:, 1]
    filtered_log_acceleration = filtered_states[:, 2]
    filtered_indicator        = filtered_states[:, 3]

    # トレンドを分類
    trend_states = classify_trends_from_states(filtered_states, threshold)

    # 結果をDataFrame化
    result_df = pd.DataFrame({
        'price': close.values,
        'composite_indicator': composite.values,
        'filtered_log_price': filtered_log_price,
        'filtered_log_velocity': filtered_log_velocity,
        'filtered_log_acceleration': filtered_log_acceleration,
        'trend_state': trend_states
    }, index=indicators.index)

    return result_df


###############################################################################
# 6) 可視化クラス: TrendVisualizer
###############################################################################
class TrendVisualizer:
    """
    トレンド検出結果のDataFrameを受け取り、可視化用のメソッドを提供する。
    """
    def __init__(self, results_df):
        self.results = results_df.copy()

    def plot_price_with_trend(self, title="Price with Trend Background"):
        """
        価格チャートの背景色でトレンド状態を表示。
        """
        colors = {
            'Uptrend': 'green',
            'Downtrend': 'red',
            'Range-bound': 'gray',
            'Weak Uptrend': 'lightgreen',
            'Weak Downtrend': 'lightcoral',
            'Slowing Uptrend': 'yellowgreen',
            'Slowing Downtrend': 'lightsalmon'
        }
        color_meanings = {
            'Uptrend':          'Uptrend (log velocity > +threshold, accel >= 0)',
            'Downtrend':        'Downtrend (log velocity < -threshold, accel <= 0)',
            'Range-bound':      'Range-bound (|log velocity| < threshold*0.5)',
            'Weak Uptrend':     'Weak Uptrend (velocity > 0, below threshold)',
            'Weak Downtrend':   'Weak Downtrend (velocity < 0, above -threshold)',
            'Slowing Uptrend':  'Slowing Uptrend (velocity > 0 with strong decel)',
            'Slowing Downtrend':'Slowing Downtrend (velocity < 0 with strong accel)'
        }

        fig, ax = plt.subplots(figsize=(12, 6))
        date_format = mdates.DateFormatter('%Y-%m-%d')
        ax.plot(self.results.index, self.results['price'],
                label='Price', color='black', linewidth=1.5)

        prev_state = None
        start_idx = 0
        for i in range(len(self.results)):
            state = self.results['trend_state'].iloc[i]
            # トレンド状態の変化箇所、または最終行でaxvspanを区切る
            if prev_state is not None and (state != prev_state or i == len(self.results) - 1):
                color = colors.get(prev_state, 'gray')
                alpha = 0.3 if prev_state in ['Uptrend', 'Downtrend'] else 0.2
                ax.axvspan(self.results.index[start_idx], self.results.index[i],
                           color=color, alpha=alpha)
                start_idx = i
            if prev_state is None or state != prev_state:
                prev_state = state

        ax.set_title(title)
        ax.xaxis.set_major_formatter(date_format)
        ax.grid(True, alpha=0.3)

        # カラーパッチの凡例
        import matplotlib.patches as mpatches
        patches_list = []
        for state, color in colors.items():
            alpha_val = 0.3 if state in ['Uptrend', 'Downtrend'] else 0.2
            label_text = f"{state}: {color_meanings[state]}"
            patch = mpatches.Patch(color=color, alpha=alpha_val, label=label_text)
            patches_list.append(patch)
        ax.legend(handles=patches_list, loc='upper left')

        fig.autofmt_xdate()
        plt.tight_layout()
        plt.show()

    def plot_indicators_and_velocity(self, threshold=0.02):
        """
        複合インジケータと推定されたlog velocityを2段構成で描画。
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        date_format = mdates.DateFormatter('%Y-%m-%d')

        # Subplot 1: Composite Indicator
        ax1.plot(self.results.index, self.results['composite_indicator'],
                 label='Composite Indicator', color='purple')
        ax1.set_title('Composite Indicator')
        ax1.xaxis.set_major_formatter(date_format)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        # Subplot 2: Estimated Log Velocity
        ax2.plot(self.results.index, self.results['filtered_log_velocity'],
                 label='Estimated Log Velocity', color='blue')
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.axhline(y=threshold, color='green', linestyle='--', alpha=0.5,
                    label=f'Uptrend Threshold: +{threshold}')
        ax2.axhline(y=-threshold, color='red', linestyle='--', alpha=0.5,
                    label=f'Downtrend Threshold: -{threshold}')
        ax2.set_title('Estimated Log Velocity')
        ax2.xaxis.set_major_formatter(date_format)
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)

        fig.autofmt_xdate()
        plt.tight_layout()
        plt.show()


###############################################################################
# 7) ユーティリティ関数: 日付範囲でDataFrameをフィルタリング
###############################################################################
def filter_by_date_range(df, start_date=None, end_date=None):
    """
    日付範囲でDataFrameをフィルタリングする。
    """
    if start_date is not None:
        df = df.loc[df.index >= start_date]
    if end_date is not None:
        df = df.loc[df.index <= end_date]
    return df


###############################################################################
# 8) メイン実行関数: run_btc_analysis
###############################################################################
def run_btc_analysis(
    data_df,
    filter_start=None,
    filter_end=None,
    display_start=None,
    display_end=None,
    threshold=0.01
):
    """
    BTCの価格データに対してEKFを適用し、トレンド分析・可視化を行う。

    Parameters
    ----------
    data_df : pd.DataFrame
        ['start_at', 'close', 'roc', 'rsi', 'macdhist', 'mfi', 'volatility']などを含むDataFrame
    filter_start, filter_end : datetime or None
        フィルタリングする期間（EKFの適用範囲）
    display_start, display_end : datetime or None
        可視化する期間（グラフ描画の範囲）
    threshold : float
        トレンド検出用の閾値（log velocity の絶対値）
    """
    # 'start_at' があればDatetimeIndexに
    if 'start_at' in data_df.columns:
        data_df['start_at'] = pd.to_datetime(data_df['start_at'])
        df = data_df.set_index('start_at').copy()
    else:
        df = data_df.copy()

    # EKF用のフィルタ期間
    df_filtered = filter_by_date_range(df, filter_start, filter_end)

    # トレンド検出
    results = detect_trend(df_filtered, ekf=None, threshold=threshold)

    # 可視化用にさらに絞る
    display_results = filter_by_date_range(results, display_start, display_end)

    print("=== Filtered Results (last 10 rows) ===")
    print(results.tail(10))

    # 可視化
    viz = TrendVisualizer(display_results)
    viz.plot_price_with_trend(title="BTC/USDT Price Chart with Trend Background")
    viz.plot_indicators_and_velocity(threshold=threshold)

    return results


###############################################################################
# 9) 実行ブロック (MongoDB部分を削除せずに残す)
###############################################################################
if __name__ == "__main__":
    import os
    import sys
    from datetime import datetime, timedelta

    # カレントパス調整（元のコードに準拠）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)

    # MongoDB関連の読み込み（元のコード通り）
    from mongodb.data_loader_mongo import MongoDataLoader
    from common.constants import MARKET_DATA_TECH

    # MongoDBからデータを取得
    db = MongoDataLoader()
    raw_df = db.load_data_from_datetime_period(
        datetime(2023, 1, 1),
        datetime(2025, 3, 4),
        coll_type=MARKET_DATA_TECH,
        symbol='BTCUSDT',
        interval=720
    )

    # フィルタリングと表示範囲
    filter_start_date = datetime(2023, 1, 1)
    filter_end_date   = datetime(2025, 3, 4)
    display_start_date = datetime(2025, 1, 1)
    display_end_date   = datetime(2025, 3, 4)

    # 分析実行
    results = run_btc_analysis(
        data_df=raw_df,
        filter_start=filter_start_date,
        filter_end=filter_end_date,
        display_start=display_start_date,
        display_end=display_end_date,
        threshold=0.01
    )




