import numpy as np
import matplotlib.pyplot as plt

def simulate_gbm(S0, mu, sigma, T, N, seed=None):
    """
    幾何ブラウン運動(GBM)による株価シミュレーションを実行する関数

    Parameters:
      S0 (float): 初期株価
      mu (float): ドリフト（成長率）
      sigma (float): ボラティリティ
      T (float): シミュレーション期間（年単位）
      N (int): タイムステップ数
      seed (int, optional): 乱数シード（再現性のため）

    Returns:
      t (np.ndarray): 時間軸の配列
      S (np.ndarray): シミュレーションした株価の配列
    """
    if seed is not None:
        np.random.seed(seed)
    dt = T / N
    t = np.linspace(0, T, N + 1)
    # 標準正規乱数からホワイトノイズを生成し、累積和を取る
    W = np.random.standard_normal(size=N)
    W = np.concatenate(([0], np.cumsum(np.sqrt(dt) * W)))
    # 幾何ブラウン運動の式に従って株価を計算
    S = S0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * W)
    return t, S

def plot_simulations(simulations, title="Stock Price Simulation using Geometric Brownian Motion"):
    """
    複数のシミュレーション結果をプロットする関数

    Parameters:
      simulations (list of dict): 各辞書は {'t': 時間軸, 'S': 株価配列, 'label': ラベル} の形式
      title (str): プロットタイトル
    """
    plt.figure(figsize=(10, 6))
    for sim in simulations:
        plt.plot(sim['t'], sim['S'], label=sim['label'])
    plt.xlabel("Time (Years)")
    plt.ylabel("Stock Price")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # シミュレーションパラメータ
    S0 = 100      # 初期株価
    T = 1.0       # シミュレーション期間（年）
    N = 365       # タイムステップ数（例：252営業日）

    # シナリオごとのパラメータ設定
    scenarios = [
        {"mu": 0.5, "sigma": 0.3, "label": "Upward Trend"},
        {"mu": -0.5, "sigma": 0.3, "label": "Downward Trend"},
        {"mu": 0.0, "sigma": 0.05, "label": "Range-bound"}
    ]

    simulations = []
    # 各シナリオのシミュレーションを実行
    for scenario in scenarios:
        t, S = simulate_gbm(S0, scenario["mu"], scenario["sigma"], T, N)
        simulations.append({"t": t, "S": S, "label": scenario["label"]})

    # シミュレーション結果をプロット
    plot_simulations(simulations)

if __name__ == '__main__':
    main()
