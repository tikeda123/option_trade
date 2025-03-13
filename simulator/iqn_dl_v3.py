import os
import sys
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras

# ユーザー環境に合わせたパス設定（必要に応じて修正）
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# MongoDBからデータを取得するためのモジュールや定数（環境に合わせて実装済みとする）
from mongodb.data_loader_mongo import MongoDataLoader
from common.constants import MARKET_DATA_TECH


############################
# CRPSを計算する関数を追加
############################
def compute_crps(
    pred_distributions: np.ndarray,
    actual_values: np.ndarray,
    v_min: float,
    v_max: float,
    num_atoms: int
) -> float:
    """
    離散化した分布予測に対し、実際の値との CRPS を計算する。
    """
    atom_range = np.linspace(v_min, v_max, num_atoms)
    bin_width = (v_max - v_min) / (num_atoms - 1)
    N = len(actual_values)
    total_crps = 0.0

    for i in range(N):
        cdf_pred = np.cumsum(pred_distributions[i])
        x = np.clip(actual_values[i], v_min, v_max)
        i_x = np.searchsorted(atom_range, x, side='left')
        indicator = np.zeros(num_atoms)
        indicator[i_x:] = 1.0
        diff = cdf_pred - indicator
        crps_i = np.sum(diff**2) * bin_width
        total_crps += crps_i

    mean_crps = total_crps / N
    return mean_crps


############################
# 1. 分布的RLモデルの構築（C51ベース）
############################
def create_distributional_rl_model(
    state_dim: int,
    num_atoms: int,
    v_min: float,
    v_max: float
) -> keras.Model:
    inputs = keras.Input(shape=(state_dim,))
    x = keras.layers.Dense(64, activation='relu')(inputs)
    x = keras.layers.Dense(64, activation='relu')(x)
    logits = keras.layers.Dense(num_atoms)(x)
    probabilities = keras.layers.Softmax(axis=1)(logits)

    model = keras.Model(inputs=inputs, outputs=probabilities)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy'
    )
    return model


############################
# 2. データ前処理と特徴量エンジニアリング
############################
def preprocess_data(
    df: pd.DataFrame,
    lookback_window: int = 24,
    future_gap: int = 12
) -> Tuple[np.ndarray, np.ndarray, Any]:
    df = df.sort_values('start_at').reset_index(drop=True)

    features = ['close', 'volume', 'macdhist', 'rsi', 'volatility', 'mfi']
    data = df[features].values

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    X = []
    y = []

    for i in range(len(df) - lookback_window - future_gap):
        X.append(data_scaled[i : i + lookback_window].flatten())
        current_close = df.loc[i + lookback_window - 1, 'close']
        future_close = df.loc[i + lookback_window - 1 + future_gap, 'close']
        diff = future_close - current_close
        y.append(diff)

    X = np.array(X)
    y = np.array(y)

    return X, y, scaler


############################
# 3. モデルのトレーニング (CRPS導入)
############################
def train_distributional_rl_model(
    model: keras.Model,
    X: np.ndarray,
    y: np.ndarray,
    num_atoms: int,
    v_min: float,
    v_max: float,
    epochs: int = 10,
    batch_size: int = 32
):
    """
    C51モデルの学習を行い、学習後にCRPSを計算して表示する。
    """
    y_clipped = np.clip(y, v_min, v_max)
    atom_range = np.linspace(v_min, v_max, num_atoms)
    target_distributions = np.zeros((len(y_clipped), num_atoms))

    for i in range(len(y_clipped)):
        idx = np.argmin(np.abs(atom_range - y_clipped[i]))
        target_distributions[i, idx] = 1.0

    history = model.fit(
        X,
        target_distributions,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    # CRPS計算
    pred_distributions = model.predict(X, verbose=0)
    crps_score = compute_crps(
        pred_distributions=pred_distributions,
        actual_values=y,  # 内部でクリップ
        v_min=v_min,
        v_max=v_max,
        num_atoms=num_atoms
    )
    print(f"Training data CRPS: {crps_score:.5f}")

    return history


############################
# 4. 未来の価格分布予測
############################
def predict_price_distribution(
    model: keras.Model,
    current_state: np.ndarray,
    v_min: float,
    v_max: float,
    num_atoms: int
) -> np.ndarray:
    distribution = model.predict(current_state, verbose=0)[0]
    distribution = distribution / np.sum(distribution)
    return distribution


############################
# 5. 予測分布の可視化と結果出力
############################
def visualize_distribution(
    distribution: np.ndarray,
    v_min: float,
    v_max: float,
    num_atoms: int,
    current_price: float,
    future_gap: int
):
    atom_range = np.linspace(v_min, v_max, num_atoms)
    plt.figure(figsize=(8, 4))
    sns.barplot(x=atom_range, y=distribution, color='blue')
    plt.title(f'Predicted Price Difference Distribution ({future_gap} hours later)')
    plt.xlabel('Price Difference from Current')
    plt.ylabel('Probability')
    plt.show()


def print_distribution_summary(
    distribution: np.ndarray,
    v_min: float,
    v_max: float,
    num_atoms: int,
    current_price: float,
    forecast_date: datetime,
    future_gap: int
):
    atom_range = np.linspace(v_min, v_max, num_atoms)

    up_2_prob = 0.0
    up_1_2_prob = 0.0
    around_1_prob = 0.0
    down_1_2_prob = 0.0
    down_2_prob = 0.0

    for i, diff in enumerate(atom_range):
        pct_change = diff / current_price * 100
        prob = distribution[i]

        if pct_change >= 2:
            up_2_prob += prob
        elif 1 < pct_change < 2:
            up_1_2_prob += prob
        elif -1 <= pct_change <= 1:
            around_1_prob += prob
        elif -2 < pct_change < -1:
            down_1_2_prob += prob
        elif pct_change <= -2:
            down_2_prob += prob

    price_up_1_threshold = current_price * 1.01
    price_up_2_threshold = current_price * 1.02
    price_down_1_threshold = current_price * 0.99
    price_down_2_threshold = current_price * 0.98

    if around_1_prob > 0.5:
        confidence = "高"
    elif around_1_prob > 0.2:
        confidence = "中"
    else:
        confidence = "低"

    market_vol_comment = "ボラティリティが高めです。" if around_1_prob < 0.3 else "ボラティリティは比較的落ち着いています。"

    forecast_date_str = forecast_date.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[基準日時: {forecast_date_str}] のBTC価格は {current_price:.2f} と想定します。\n")
    print(f"{future_gap}時間後のビットコイン価格について、以下の確率分布が得られました:\n")
    print(f"- {up_2_prob*100:.1f}% の確率で「現在より2%以上上昇」 (約 {price_up_2_threshold:.2f} 以上)")
    print(f"- {up_1_2_prob*100:.1f}% の確率で「現在より1%〜2%上昇」 (約 {price_up_1_threshold:.2f} 〜 {price_up_2_threshold:.2f})")
    print(f"- {around_1_prob*100:.1f}% の確率で「現在より±1%以内の変動」 (約 {price_down_1_threshold:.2f} 〜 {price_up_1_threshold:.2f})")
    print(f"- {down_1_2_prob*100:.1f}% の確率で「現在より1%〜2%下落」 (約 {price_down_2_threshold:.2f} 〜 {price_down_1_threshold:.2f})")
    print(f"- {down_2_prob*100:.1f}% の確率で「現在より2%以上下落」 (約 {price_down_2_threshold:.2f} 以下)\n")
    print(f"この予測の信頼度は【{confidence}】です。")
    print(f"市場状況：{market_vol_comment}\n")

def compute_v_range(y: np.ndarray, future_gap: int, base_future_gap: int = 12, base_safety_margin: float = 0.1) -> Tuple[float, float]:
    """
    学習データのターゲット値 y（価格差分）の1パーセンタイルと99パーセンタイルから
    v_min, v_max を設定する関数です。

    future_gap の値に応じて安全マージンを調整します。
    例えば、base_future_gap（例：12 時間）に対して future_gap が大きい場合は、
    安全マージンも比例して大きくなります。

    Parameters:
      y: ターゲット値の配列（価格差分）
      future_gap: 現在設定している future_gap の値（時間）
      base_future_gap: 安全マージンの基準となる future_gap（デフォルト12時間）
      base_safety_margin: 基本の安全マージン（デフォルト0.1＝10%）

    Returns:
      v_min, v_max: モデルのターゲットレンジの下限と上限
    """
    q_low = np.percentile(y, 1)
    q_high = np.percentile(y, 99)

    # future_gap に応じた安全マージンの調整（例: future_gapが倍なら安全マージンも倍）
    adjusted_margin = base_safety_margin * (future_gap / base_future_gap)

    # q_low が負の場合、より下方向に広げるため (1 + adjusted_margin) を乗じる（値はより小さくなる）
    if q_low < 0:
        v_min = q_low * (1 + adjusted_margin)
    else:
        v_min = q_low * (1 - adjusted_margin)

    # q_high が正の場合、より上方向に広げるため (1 + adjusted_margin) を乗じる
    if q_high > 0:
        v_max = q_high * (1 + adjusted_margin)
    else:
        v_max = q_high * (1 - adjusted_margin)

    return v_min, v_max


############################
# メイン処理
############################
def main(
    future_gap: int = 12,
    lookback_window: int = 24
):
    df = MongoDataLoader().load_data_from_datetime_period(
        datetime(2023, 1, 1),
        datetime(2025, 3, 1),
        coll_type=MARKET_DATA_TECH,
        symbol='BTCUSDT',
        interval=60
    )

    forecast_date_str = "2025-02-08 00:00:00"
    forecast_date = datetime.strptime(forecast_date_str, "%Y-%m-%d %H:%M:%S")
    training_end_date = forecast_date - timedelta(hours=future_gap)

    df_for_training = df[df['start_at'] <= training_end_date].copy()
    graph_df_for_training = df_for_training[['start_at', 'close', 'volume', 'macdhist', 'rsi', 'volatility', 'mfi']]

    X_train, y_train, scaler = preprocess_data(
        df=graph_df_for_training,
        lookback_window=lookback_window,
        future_gap=future_gap
    )
    state_dim = X_train.shape[1]

    # future_gap を考慮して自動的に v_min, v_max を算出
    v_min, v_max = compute_v_range(y_train, future_gap=future_gap, base_future_gap=12, base_safety_margin=0.1)
    print(f"設定されたレンジ: v_min={v_min:.2f}, v_max={v_max:.2f}")

    num_atoms = 51

    model = create_distributional_rl_model(state_dim, num_atoms, v_min, v_max)

    # 学習時に CRPS を計算
    train_distributional_rl_model(
        model=model,
        X=X_train,
        y=y_train,
        num_atoms=num_atoms,
        v_min=v_min,
        v_max=v_max,
        epochs=50,
        batch_size=32
    )

    # 予測用データの準備
    df_before_forecast = df[df['start_at'] <= forecast_date].copy()
    graph_df_before_forecast = df_before_forecast[['start_at', 'close', 'volume', 'macdhist', 'rsi', 'volatility', 'mfi']]

    if len(graph_df_before_forecast) < lookback_window:
        raise ValueError("予測に必要な十分なデータがありません。")

    latest_data = graph_df_before_forecast.tail(lookback_window)
    current_price = latest_data['close'].iloc[-1]

    latest_features = latest_data[['close', 'volume', 'macdhist', 'rsi', 'volatility', 'mfi']].values
    latest_scaled = scaler.transform(latest_features)
    current_state = latest_scaled.flatten()[np.newaxis, :]

    distribution = predict_price_distribution(model, current_state, v_min, v_max, num_atoms)

    visualize_distribution(
        distribution=distribution,
        v_min=v_min,
        v_max=v_max,
        num_atoms=num_atoms,
        current_price=current_price,
        future_gap=future_gap
    )
    print_distribution_summary(
        distribution=distribution,
        v_min=v_min,
        v_max=v_max,
        num_atoms=num_atoms,
        current_price=current_price,
        forecast_date=forecast_date,
        future_gap=future_gap
    )

if __name__ == "__main__":
    main(future_gap=24, lookback_window=48)
