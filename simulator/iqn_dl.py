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

    # 隠れ層の例
    x = keras.layers.Dense(64, activation='relu')(inputs)
    x = keras.layers.Dense(64, activation='relu')(x)

    # 出力層：num_atomsのロジットを出力し、後でソフトマックスにより確率分布に変換
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
    # 時系列順にソート
    df = df.sort_values('start_at').reset_index(drop=True)

    features = ['close', 'volume', 'macdhist', 'rsi', 'volatility', 'mfi']
    data = df[features].values

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    X = []
    y = []

    # ターゲット: 現在の最後の価格と future_gap 時間後の価格の差分
    for i in range(len(df) - lookback_window - future_gap):
        # 直近 lookback_window 時間分のデータを1サンプルの特徴量として利用
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
    # yを[v_min, v_max]の範囲にクリップ
    y_clipped = np.clip(y, v_min, v_max)

    # 教師ラベルとして各サンプルをnum_atomsのone-hot表現に変換（簡易な実装）
    atom_range = np.linspace(v_min, v_max, num_atoms)
    target_distributions = np.zeros((len(y), num_atoms))

    for i in range(len(y)):
        idx = np.argmin(np.abs(atom_range - y_clipped[i]))
        target_distributions[i, idx] = 1.0

    # 学習（クロスエントロピー損失を使用）
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

    distribution: (num_atoms,) の確率分布
    v_min, v_max, num_atoms: 分布パラメータ
    current_price: 現在のBTC価格（予測対象の基準値として利用）
    future_gap: 予測する時間ギャップ
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
    今回は区分を以下のように変更しています:
      - 2%以上上昇
      - 1%〜2%上昇
      - ±1%以内の変動
      - 1%〜2%下落
      - 2%以上下落
    """

    atom_range = np.linspace(v_min, v_max, num_atoms)

    # 各区分ごとの確率を初期化
    up_2_prob = 0.0         # 2%以上上昇
    up_1_2_prob = 0.0       # 1%〜2%上昇
    around_1_prob = 0.0     # ±1%以内の変動
    down_1_2_prob = 0.0     # 1%〜2%下落
    down_2_prob = 0.0       # 2%以上下落

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

    # 新しい価格閾値の計算
    price_up_1_threshold = current_price * 1.01
    price_up_2_threshold = current_price * 1.02
    price_down_1_threshold = current_price * 0.99
    price_down_2_threshold = current_price * 0.98

    # 簡易的な信頼度評価（例：±1%以内の確率が大きいほど信頼度が高い）
    if around_1_prob > 0.5:
        confidence = "高"
    elif around_1_prob > 0.2:
        confidence = "中"
    else:
        confidence = "低"

    # 市場状況のコメント
    if around_1_prob < 0.3:
        market_vol_comment = "ボラティリティが高めです。"
    else:
        market_vol_comment = "ボラティリティは比較的落ち着いています。"


    forecast_date_str = forecast_date.strftime("%Y-%m-%d %H:%M:%S")

    # 結果表示
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
# メイン処理
############################
def main(
    future_gap: int = 12,
    lookback_window: int = 24
):
    """
    main 関数で 予測の未来時間 (future_gap) と ルックバックウィンドウ (lookback_window) を
    パラメータとして受け取るように変更。
    """
    # -------------------------------
    # 1. データを取得する期間の指定
    # -------------------------------
    df = MongoDataLoader().load_data_from_datetime_period(
        datetime(2023, 1, 1),
        datetime(2025, 3, 1),  # 元の例と同様にこの範囲を取得
        coll_type=MARKET_DATA_TECH,
        symbol='BTCUSDT',
        interval=60
    )

    # -------------------------------
    # 2. 予測基準日時の設定
    # -------------------------------
    forecast_date_str = "2025-02-08 00:00:00"
    forecast_date = datetime.strptime(forecast_date_str, "%Y-%m-%d %H:%M:%S")

    # ※ データリークを防ぐため、学習データの最終時刻を "予測基準日時 - future_gap" に設定する
    training_end_date = forecast_date - timedelta(hours=future_gap)

    # -------------------------------
    # 3. 学習に使う期間を「training_end_date まで」に限定
    # -------------------------------
    df_for_training = df[df['start_at'] <= training_end_date].copy()
    # 利用カラムのみ
    graph_df_for_training = df_for_training[['start_at', 'close', 'volume', 'macdhist', 'rsi', 'volatility', 'mfi']]

    # -------------------------------
    # 4. 前処理・学習データの作成
    # -------------------------------
    X_train, y_train, scaler = preprocess_data(
        df=graph_df_for_training,
        lookback_window=lookback_window,
        future_gap=future_gap
    )
    state_dim = X_train.shape[1]

    # -------------------------------
    # 5. モデルの構築と学習
    # -------------------------------
    num_atoms = 51
    v_min = -2000  # 価格差分の最小値（例）
    v_max = 2000   # 価格差分の最大値（例）

    model = create_distributional_rl_model(state_dim, num_atoms, v_min, v_max)

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

    # -------------------------------
    # 6. 推論用（予測基準日時まで）のデータ準備
    # -------------------------------
    df_before_forecast = df[df['start_at'] <= forecast_date].copy()
    graph_df_before_forecast = df_before_forecast[['start_at', 'close', 'volume', 'macdhist', 'rsi', 'volatility', 'mfi']]

    if len(graph_df_before_forecast) < lookback_window:
        raise ValueError("予測に必要な十分なデータがありません。")

    # 直近 lookback_window 行を推論用特徴量として使用
    latest_data = graph_df_before_forecast.tail(lookback_window)

    # 現在のBTC価格
    current_price = latest_data['close'].iloc[-1]

    # -------------------------------
    # 7. 状態ベクトルの作成
    # -------------------------------
    latest_features = latest_data[['close', 'volume', 'macdhist', 'rsi', 'volatility', 'mfi']].values
    latest_scaled = scaler.transform(latest_features)
    current_state = latest_scaled.flatten()[np.newaxis, :]

    # -------------------------------
    # 8. 価格差分の確率分布を予測
    # -------------------------------
    distribution = predict_price_distribution(model, current_state, v_min, v_max, num_atoms)

    # -------------------------------
    # 9. 可視化と結果の出力
    # -------------------------------
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
    # future_gap と lookback_window を必要に応じて変更可能
    main(future_gap=24, lookback_window=96)


