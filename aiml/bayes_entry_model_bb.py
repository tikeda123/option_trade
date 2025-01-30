import os
import sys
from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import beta

# Get the absolute path of the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the path of the parent directory to sys.path
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# ここから先はお使いの環境に合わせて import を調整
from mongodb.data_loader_mongo import MongoDataLoader
from common.constants import MARKET_DATA_TECH

# ====== 前処理 ====== #
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """前処理（特徴量エンジニアリング）を行う関数"""
    df = df.copy()
    # 欠損値処理
    df.dropna(subset=["open", "high", "low", "close", "volume"], inplace=True)

    # 例: EMA / SMA
    df["ema"] = df["close"].ewm(span=20, adjust=False).mean()
    df["sma"] = df["close"].rolling(window=20).mean()

    # 欠損値が残る行を削除
    df.dropna(inplace=True)
    return df


# ====== ボリンジャーバンド-2σ を下回った際の n 日後の上昇 / 下落を数える ====== #
def count_bollinger_reversal(df: pd.DataFrame, future_days: int = 7) -> Tuple[int, int]:
    """
    ボリンジャーバンド-2σを下回ったタイミングから future_days 日後の上昇(up)/下落(down)回数をカウント。
    """
    n_ups = 0
    n_downs = 0

    # 「close」が「lower2」を下回ったインデックスを抽出
    reversal_indices = df[df["close"] < df["lower2"]].index

    # DatetimeIndex を想定 → future_days 日後は pd.Timedelta(days=future_days) で計算
    for idx in reversal_indices:
        future_idx = idx + pd.Timedelta(days=future_days)
        if future_idx <= df.index.max():
            # future_days後の価格がエントリー時より高いかどうか
            if df["close"].loc[future_idx] > df["close"].loc[idx]:
                n_ups += 1
            else:
                n_downs += 1

    return n_ups, n_downs


# ====== 経験ベイズ用: サブサンプルの (n_ups, n_downs) から alpha, beta を推定（モーメント法） ====== #
def empirical_bayes_alpha_beta(subsample_results: pd.DataFrame) -> Tuple[float, float]:
    """
    サブサンプルごとの (n_ups, n_downs) を集計した DataFrame から、
    Beta-Binomialモデルの alpha, beta を推定する。簡易実装としてモーメント法を用いる。
    """
    subsample_results = subsample_results.copy()
    subsample_results["n_total"] = subsample_results["n_ups"] + subsample_results["n_downs"]
    subsample_results["p_i"] = subsample_results["n_ups"] / subsample_results["n_total"]  # サブサンプル成功率

    # 重み付き平均成功率 p_bar
    total_success = subsample_results["n_ups"].sum()
    total_trials = subsample_results["n_total"].sum()
    p_bar = total_success / total_trials if total_trials > 0 else 0.5

    # 2次モーメント m2 = E[p_i^2]
    m2 = (subsample_results["n_total"] * subsample_results["p_i"]**2).sum() / total_trials if total_trials > 0 else (p_bar**2)
    var = m2 - p_bar**2  # Var[p] = E[p^2] - (E[p])^2

    # モーメント法
    if var <= 0 or var >= p_bar * (1 - p_bar):
        # 分散がゼロまたは異常な場合 → α=β=1.0 (一様分布に近い) にフォールバック
        alpha_hat = 1.0
        beta_hat = 1.0
    else:
        alpha_beta_sum = (p_bar * (1 - p_bar) / var) - 1
        alpha_hat = p_bar * alpha_beta_sum
        beta_hat = (1 - p_bar) * alpha_beta_sum

    return alpha_hat, beta_hat


# ====== 経験ベイズ法：サブサンプル分割 → alpha, beta を推定 → 全体に適用して事後分布 ====== #
def empirical_bayes_bollinger_reversal(
    df: pd.DataFrame,
    future_days: int = 7,
    freq: str = "M"  # 月ごと("M") / 週ごと("W") / 日ごと("D") etc.
) -> Tuple[float, float, float]:
    """
    経験ベイズ法でボリンジャーバンド-2σ反発戦略の有効性を検証する関数。
      1) freq 単位でサブサンプルを切り、(n_ups, n_downs) をカウント
      2) それらを用いて alpha, beta を推定（= 経験ベイズ事前分布）
      3) 全データの (n_ups, n_downs) を加えて最終的な事後分布を得る

    Returns:
        (alpha_posterior, beta_posterior, expected_prob)
         事後分布パラメータと、その期待値
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df.index must be a DatetimeIndex for pd.Grouper(freq=...).")

    # 1. freq ごとにグルーピング → サブサンプルでの n_ups, n_downs を集計
    grouped = df.groupby(pd.Grouper(freq=freq))
    results_list = []

    for group_key, subdf in grouped:
        # データが少なすぎるサブサンプルはスキップ
        if len(subdf) < future_days:
            continue

        n_ups, n_downs = count_bollinger_reversal(subdf, future_days=future_days)
        if (n_ups + n_downs) > 0:
            results_list.append((group_key, n_ups, n_downs))

    # サブサンプルが得られない場合は既定値
    if len(results_list) == 0:
        print("[Warning] サブサンプルなし → 経験ベイズ推定不可。デフォルト値を返します。")
        return (1.0, 1.0, 0.5)

    subsample_df = pd.DataFrame(results_list, columns=["group_key", "n_ups", "n_downs"])

    # 2. サブサンプル情報から (alpha, beta) を推定
    alpha_prior, beta_prior = empirical_bayes_alpha_beta(subsample_df)
    print(f"[Empirical Bayes] 推定された事前分布 (α, β) = ({alpha_prior:.4f}, {beta_prior:.4f})")

    # 3. 全データに対して最終的な事後分布 (alpha, beta) を計算
    n_ups_total, n_downs_total = count_bollinger_reversal(df, future_days=future_days)
    alpha_posterior = alpha_prior + n_ups_total
    beta_posterior = beta_prior + n_downs_total

    expected_prob = alpha_posterior / (alpha_posterior + beta_posterior)
    return alpha_posterior, beta_posterior, expected_prob


# ====== メイン関数 ====== #
def main():
    # === 1. データ読み込み ===#
    data_loader = MongoDataLoader()
    start_date = "2020-01-01"
    end_date   = "2024-04-01"

    # 下記の引数はユーザ環境に合わせて変更してください
    df = data_loader.load_data_from_datetime_period(
        start_date,
        end_date,
        coll_type=MARKET_DATA_TECH,
        symbol="BTCUSDT",
        interval=1440
    )

    # ↑ ここで取得した DataFrame に、以下のようなカラムがあることを想定:
    #   df["date"] ... "2025-01-01 16:00:00" の形式（例）
    #   df["close"], df["lower2"], など。
    #   ボリンジャーバンド計算はサーバ側 or コレクション側で既に実施済み、と想定。

    # == 2. df['date'] を DatetimeIndex として設定 ==
    if "date" not in df.columns:
        raise KeyError("DataFrame に 'date' カラムがありません。MongoDBのデータを確認してください。")

    df["date"] = pd.to_datetime(df["date"])  # "YYYY-mm-dd HH:MM:SS" などを datetime64 に変換
    df.set_index("date", inplace=True)       # インデックスを DatetimeIndex に

    # == 3. 前処理 (例: 欠損値処理, 移動平均など)
    df = preprocess_data(df)

    # == 4. 経験ベイズ法による検証 ==
    future_days = 7  # 何日後の価格を判定するか
    alpha_posterior, beta_posterior, expected_prob = empirical_bayes_bollinger_reversal(
        df,
        future_days=future_days,
        freq="M"  # 月ごとのサブサンプル分割 (任意で "W"/"D"/"Q" などに変更)
    )

    # ログ出力
    print(f"\n検証期間: {start_date} ～ {end_date}")
    print(f"ボリンジャーバンド -2σ 反発後、{future_days}日後の価格上昇確率 (事後分布)")
    print(f" => alpha={alpha_posterior:.4f}, beta={beta_posterior:.4f}")
    print(f" => 期待値(平均)={expected_prob:.4f}")

    # == 5. 事後分布を可視化 ==
    x = np.linspace(0, 1, 200)
    y = beta.pdf(x, alpha_posterior, beta_posterior)
    plt.figure(figsize=(6, 4))
    plt.plot(x, y, label="Posterior PDF")
    plt.title("Empirical Bayes: Posterior Distribution\n(Bollinger -2σ Reversal)")
    plt.xlabel("Probability of Price Increase")
    plt.ylabel("Density")
    plt.axvline(x=expected_prob, color="red", linestyle="--", label=f"Mean={expected_prob:.3f}")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
