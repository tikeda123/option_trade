import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# 日本語フォントの設定
# フォントが利用できない場合はインストールが必要
# Windowsの場合は 'MS Gothic' などを使用できます
# Macの場合は 'Hiragino Sans GB' など
# Linuxの場合は 'IPAGothic' などを使用
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['IPAGothic', 'AppleGothic', 'MS Gothic', 'Hiragino Sans GB', 'SimHei', 'Noto Sans CJK JP']
mpl.rcParams['axes.unicode_minus'] = False  # マイナス記号を正しく表示

# 人工的なデータを生成（ステップ状の変化とノイズを含む）
def generate_data(n_points=200):
    t = np.linspace(0, 10, n_points)

    # 真の信号（ステップ状の変化を含む）
    true_signal = np.zeros(n_points)
    true_signal[50:100] = 1.0    # 最初のステップ変化
    true_signal[100:150] = 0.5   # 2番目のステップ変化
    true_signal[150:] = 2.0      # 3番目のステップ変化

    # ノイズを追加
    np.random.seed(42)  # 再現性のため
    noise = np.random.normal(0, 0.3, n_points)
    observed_data = true_signal + noise

    return t, true_signal, observed_data

# 移動平均フィルター
def moving_average(data, window_size):
    n = len(data)
    result = np.zeros(n)

    for i in range(n):
        if i < window_size:
            result[i] = np.mean(data[:i+1])
        else:
            result[i] = np.mean(data[i-window_size+1:i+1])

    return result

# シンプルなカルマンフィルター
def kalman_filter(data, process_var=0.01, measurement_var=0.1):
    n = len(data)
    filtered = np.zeros(n)
    gains = np.zeros(n)

    # 初期推定値と誤差共分散
    x_hat = data[0]  # 最初の測定値から開始
    p = 1.0          # 初期不確実性

    for i, z in enumerate(data):
        # 予測ステップ - 不確実性が増加
        p = p + process_var

        # 更新ステップ - 新しい測定値を組み込む
        # カルマンゲイン計算: 測定値の信頼度が高いほど大きな値
        k = p / (p + measurement_var)
        gains[i] = k

        # 測定値とゲインに基づいて推定値を更新
        x_hat = x_hat + k * (z - x_hat)

        # 誤差共分散の更新
        p = (1 - k) * p

        # 現在の推定値を保存
        filtered[i] = x_hat

    return filtered, gains

# メイン実験
# データ生成
t, true_signal, observed_data = generate_data()

# 移動平均フィルター適用
window_size = 10
ma_filtered = moving_average(observed_data, window_size)

# カルマンフィルター適用
kalman_filtered, kalman_gains = kalman_filter(observed_data)

# 結果をプロット
plt.figure(figsize=(12, 9))

# オリジナルデータ
plt.subplot(3, 1, 1)
plt.plot(t, observed_data, 'gray', alpha=0.6, label='観測データ（ノイズあり）')
plt.plot(t, true_signal, 'k', linewidth=2, label='真の信号')
plt.title('元のノイズ付きデータ', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

# 移動平均フィルター
plt.subplot(3, 1, 2)
plt.plot(t, observed_data, 'gray', alpha=0.3, label='観測データ')
plt.plot(t, ma_filtered, 'r', linewidth=2, label=f'移動平均 (窓幅={window_size})')
plt.plot(t, true_signal, 'k', linewidth=2, label='真の信号')
plt.title('移動平均フィルター', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

# カルマンフィルター
plt.subplot(3, 1, 3)
plt.plot(t, observed_data, 'gray', alpha=0.3, label='観測データ')
plt.plot(t, kalman_filtered, 'g', linewidth=2, label='カルマンフィルター')
plt.plot(t, true_signal, 'k', linewidth=2, label='真の信号')
plt.title('カルマンフィルター', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('kalman_vs_moving_avg.png', dpi=300)  # 画像として保存する場合
plt.show()

# 誤差指標を計算
ma_mse = np.mean((ma_filtered - true_signal) ** 2)
kalman_mse = np.mean((kalman_filtered - true_signal) ** 2)

print("\n誤差指標（平均二乗誤差）:")
print(f"移動平均: {ma_mse:.4f}")
print(f"カルマンフィルター: {kalman_mse:.4f}")
print(f"改善率: {(ma_mse - kalman_mse) / ma_mse * 100:.2f}%")

# ステップ変化への応答時間を計算
def calculate_response_time(filtered, true, step_idx, threshold=0.9):
    """ステップ変化のthreshold%に到達するまでのステップ数を計算"""
    before_step = true[step_idx-1]
    after_step = true[step_idx]
    step_size = after_step - before_step

    # 到達すべき目標値
    target = before_step + threshold * step_size

    # 目標値に到達する時点を探す
    for i in range(step_idx, len(filtered)):
        if step_size > 0 and filtered[i] >= target:
            return i - step_idx
        elif step_size < 0 and filtered[i] <= target:
            return i - step_idx

    # 到達しない場合
    return len(filtered) - step_idx

# ステップ変化の位置
step_indices = [50, 100, 150]
step_descriptions = ["0→1 (上昇)", "1→0.5 (下降)", "0.5→2 (上昇)"]

print("\n応答時間（ステップ変化の90%に到達するまでのステップ数）:")
for idx, desc in zip(step_indices, step_descriptions):
    ma_time = calculate_response_time(ma_filtered, true_signal, idx)
    kalman_time = calculate_response_time(kalman_filtered, true_signal, idx)

    print(f"\nステップ {desc} (時間 {t[idx]:.1f}):")
    print(f"  移動平均: {ma_time} ステップ")
    print(f"  カルマンフィルター: {kalman_time} ステップ")