# file: data_fetcher.py

import numpy as np
import pandas as pd

def fetch_data(
    num_hours=24*365,           # 何時間分のデータを生成するか（デフォルト: 1年分）
    start_date="2020-01-01",    # 日時の開始点
    start_price=10000.0,        # 基準価格
    amplitude=2000.0,           # 正弦波の振幅 (例: 基準価格の +/-2000 くらい)
    period_hours=24*30,         # 1周期の長さ(単位: 時間) → 例: 30日周期
    seed=None,                  # 乱数シード
):
    """
    正弦波(サイン波)ベースで上下に動く価格データを生成する。
    ランダムではなく、周期をもった値動きを模擬したシンプルな実装例。

    Returns:
        df (pd.DataFrame):
            - datetime
            - open, high, low, close
            - volume
    """

    if seed is not None:
        np.random.seed(seed)

    # 1) 時間軸を作成 (1時間刻み)
    date_range = pd.date_range(start=start_date, periods=num_hours, freq='H')

    # 2) 正弦波の生成
    #    price[t] = start_price + amplitude * sin( 2π * t / period_hours )
    #    → t=0,1,2,... (時間ステップ)
    prices = np.zeros(num_hours, dtype=float)
    for t in range(num_hours):
        sine_value = np.sin(2.0 * np.pi * t / period_hours)
        prices[t] = start_price + amplitude * sine_value

    # 3) OHLC の組み立て
    #    - open: 前の close と同一 (初回は close[0])
    #    - close: prices[t]
    #    - high / low: close を中心にわずかに上下させる (乱数で±1%程度)
    open_prices = np.zeros(num_hours)
    high_prices = np.zeros(num_hours)
    low_prices  = np.zeros(num_hours)
    close_prices = np.zeros(num_hours)

    close_prices[0] = prices[0]
    open_prices[0]  = prices[0]
    high_prices[0]  = prices[0]
    low_prices[0]   = prices[0]

    for t in range(1, num_hours):
        open_prices[t] = close_prices[t - 1]  # 前足のcloseを次のopenに
        close_prices[t] = prices[t]

        # ランダムに ±1% 程度上下にブレさせる
        # (あくまでサンプルのため簡易的に実装)
        rand_factor_high = 1.0 + np.random.uniform(-0.01, 0.02)  # -1%～+2%程度
        rand_factor_low  = 1.0 + np.random.uniform(-0.02, 0.01)  # -2%～+1%程度

        possible_high = close_prices[t] * rand_factor_high
        possible_low  = close_prices[t] * rand_factor_low

        high_prices[t] = max(open_prices[t], close_prices[t], possible_high)
        low_prices[t]  = min(open_prices[t], close_prices[t], possible_low)

    # 4) volume の仮想生成 (ここではランダムに 100～1000)
    volumes = np.random.uniform(low=100, high=1000, size=num_hours)

    # 5) DataFrame 化
    df = pd.DataFrame({
        "datetime": date_range,
        "open": open_prices,
        "high": high_prices,
        "low": low_prices,
        "close": close_prices,
        "volume": volumes
    })
    df.set_index("datetime", inplace=True)

    return df.reset_index()


def main():
    # テスト実行例
    df_synth = fetch_data(
        num_hours=24*30*2,    # 30日周期 × 2周期 = 60日分(1440時間)
        start_date="2020-01-01",
        start_price=10000.0,
        amplitude=2000.0,     # 基準価格 ±2000
        period_hours=24*30,   # 30日で1サイクル
        seed=42
    )

    print(df_synth.head(10))
    print(df_synth.tail(10))
    print("Data size:", len(df_synth))

if __name__ == "__main__":
    main()
