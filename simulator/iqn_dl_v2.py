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
# 1. 分布的RLモデルの構築（C51ベース）
############################
def create_distributional_rl_model(
    state_dim: int,
    num_atoms: int,
    v_min: float,
    v_max: float
) -> keras.Model:
    """
    分布的強化学習（C51）モデルを構築し、確率分布を出力できるネットワークを返す。

    state_dim: 入力状態の次元数
    num_atoms: アトムの数（分布を構成する離散点数）
    v_min, v_max: 価値分布の最小値と最大値
    """
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
    """
    特徴量エンジニアリングと前処理を行い、学習用データ（X, y）を返す。

    df: 入力のDataFrame。'start_at', 'close', 'volume', 'macdhist', 'rsi', 'volatility', 'mfi'
    lookback_window: 各サンプル作成時に利用する直近の時間足の数
    future_gap: 何時間後の価格差分を予測するか（例: 12時間後）

    戻り値:
      X: (サンプル数, 状態次元)
      y: (サンプル数,) future_gap時間後の価格差分
      scaler: 特徴量のスケーリングに使用したオブジェクト
    """
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
# 3. モデルのトレーニング
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
    C51モデルの学習を行う。

    model: 分布的RLモデル
    X: 学習用入力データ
    y: 教師信号としての (future_gap時間後の) 価格差分
    num_atoms, v_min, v_max: 分布を表現するためのパラメータ
    epochs, batch_size: 学習パラメータ
    """
    y_clipped = np.clip(y, v_min, v_max)
    atom_range = np.linspace(v_min, v_max, num_atoms)
    target_distributions = np.zeros((len(y), num_atoms))
    for i in range(len(y)):
        idx = np.argmin(np.abs(atom_range - y_clipped[i]))
        target_distributions[i, idx] = 1.0
    history = model.fit(
        X,
        target_distributions,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
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
    """
    現在の状態ベクトルから、将来の価格差分の確率分布を予測する。

    model: 学習済みの分布的RLモデル
    current_state: (1, state_dim) の形状を想定
    v_min, v_max, num_atoms: 分布のパラメータ
    """
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
    """
    予測された確率分布をヒストグラムで可視化する。
    """
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
    """
    予測された分布から、指定の価格変動レンジごとの確率を計算・出力します。
    ここでは区分を以下のように設定:
      - 2%以上上昇
      - 1%〜2%上昇
      - ±1%以内の変動
      - 1%〜2%下落
      - 2%以上下落
    """
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
        elif pct_change > 1 and pct_change < 2:
            up_1_2_prob += prob
        elif -1 <= pct_change <= 1:
            around_1_prob += prob
        elif pct_change < -1 and pct_change > -2:
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
    if around_1_prob < 0.3:
        market_vol_comment = "ボラティリティが高めです。"
    else:
        market_vol_comment = "ボラティリティは比較的落ち着いています。"
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


############################
# ★ 分類的評価のための追加関数
############################
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

def create_test_dataset(
    df: pd.DataFrame,
    scaler: Any,
    lookback_window: int,
    future_gap: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    学習と同じ手順でテスト用のX, yを作成するが、
    scalerは学習時にfit済みのものを使ってtransformのみ実施。

    戻り値:
      X_test: (サンプル数, state_dim)
      y_test: (サンプル数,)  future_gap時間後の価格差分
      prices: (サンプル数,)  各サンプルの「現在価格」 (クラス分け時に閾値計算で使用)
    """
    df = df.sort_values('start_at').reset_index(drop=True)
    features = ['close', 'volume', 'macdhist', 'rsi', 'volatility', 'mfi']
    data = df[features].values
    data_scaled = scaler.transform(data)
    X_test = []
    y_test = []
    prices = []
    for i in range(len(df) - lookback_window - future_gap):
        X_test.append(data_scaled[i : i + lookback_window].flatten())
        cur_close = df.loc[i + lookback_window - 1, 'close']
        future_close = df.loc[i + lookback_window - 1 + future_gap, 'close']
        diff = future_close - cur_close
        y_test.append(diff)
        prices.append(cur_close)
    return np.array(X_test), np.array(y_test), np.array(prices)

def evaluate_event_classification(
    model: keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    prices: np.ndarray,
    v_min: float,
    v_max: float,
    num_atoms: int,
    threshold: float = 0.01
):
    """
    ±(threshold*100)% を閾値とした 3クラス分類精度を評価 (Down / Stable / Up)。

    クラス定義:
      - Up(=+1):     diff / current_price >  +threshold
      - Stable(=0):  abs(diff / current_price) <= threshold
      - Down(=-1):   diff / current_price <  -threshold

    予測は、C51で得られる分布をアトムごとに合計し、
    (Up確率 / Stable確率 / Down確率) のうち最も高いものを選択する。
    """
    atom_values = np.linspace(v_min, v_max, num_atoms)
    y_true_classes = []
    y_pred_classes = []
    for diff, cur_price in zip(y_test, prices):
        ratio = diff / cur_price
        if ratio > threshold:
            y_true_classes.append(1)
        elif ratio < -threshold:
            y_true_classes.append(-1)
        else:
            y_true_classes.append(0)
    for i in range(len(X_test)):
        dist = model.predict(X_test[i:i+1], verbose=0)[0]
        dist /= np.sum(dist)
        cur_price = prices[i]
        up_prob = 0.0
        down_prob = 0.0
        stable_prob = 0.0
        for atom_val, p in zip(atom_values, dist):
            ratio = atom_val / cur_price
            if ratio > threshold:
                up_prob += p
            elif ratio < -threshold:
                down_prob += p
            else:
                stable_prob += p
        probs = [down_prob, stable_prob, up_prob]  # 順序: (-1, 0, +1)
        pred_class_idx = np.argmax(probs)
        if pred_class_idx == 0:
            y_pred_classes.append(-1)
        elif pred_class_idx == 1:
            y_pred_classes.append(0)
        else:
            y_pred_classes.append(1)
    y_true_classes = np.array(y_true_classes)
    y_pred_classes = np.array(y_pred_classes)
    acc = accuracy_score(y_true_classes, y_pred_classes)
    precision = precision_score(y_true_classes, y_pred_classes, average='macro', zero_division=0)
    recall = recall_score(y_true_classes, y_pred_classes, average='macro', zero_division=0)
    cm = confusion_matrix(y_true_classes, y_pred_classes, labels=[-1, 0, 1])
    print(f"\n=== 分類的評価 (threshold={threshold*100:.1f}% ) ===")
    print("3クラス (Down / Stable / Up)")
    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {precision:.3f}  (macro平均)")
    print(f"Recall   : {recall:.3f}    (macro平均)")
    print("Confusion Matrix (行=True, 列=Pred) => クラス順序=[Down(-1), Stable(0), Up(1)]")
    print(cm)


############################
# メイン処理
############################
def main(
    future_gap: int = 12,
    lookback_window: int = 24
):
    """
    main 関数で 予測の未来時間 (future_gap) と ルックバックウィンドウ (lookback_window) を
    パラメータとして受け取るサンプル。
    """
    df = MongoDataLoader().load_data_from_datetime_period(
        datetime(2021, 1, 1),
        datetime(2025, 3, 1),
        coll_type=MARKET_DATA_TECH,
        symbol='BTCUSDT',
        interval=60
    )
    forecast_date_str = "2025-01-01 00:00:00"
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
    num_atoms = 51
    v_min = -2000
    v_max = 2000
    model = create_distributional_rl_model(state_dim, num_atoms, v_min, v_max)
    train_distributional_rl_model(
        model=model,
        X=X_train,
        y=y_train,
        num_atoms=num_atoms,
        v_min=v_min,
        v_max=v_max,
        epochs=80,
        batch_size=32
    )
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
    # テスト期間を forecast_date の5日後までに拡大
    test_end_date = forecast_date + timedelta(days=10)
    df_test = df[(df['start_at'] > training_end_date) & (df['start_at'] <= test_end_date)].copy()
    X_test, y_test, prices_test = create_test_dataset(
        df_test,
        scaler,
        lookback_window,
        future_gap
    )
    if len(X_test) == 0:
        print("テストデータが不足しているため、分類的評価をスキップします。")
        return
    evaluate_event_classification(
        model=model,
        X_test=X_test,
        y_test=y_test,
        prices=prices_test,
        v_min=v_min,
        v_max=v_max,
        num_atoms=num_atoms,
        threshold=0.01
    )


if __name__ == "__main__":
    main(future_gap=8, lookback_window=48)


