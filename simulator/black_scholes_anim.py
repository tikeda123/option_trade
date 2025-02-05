import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import norm

# -------------------
# Black-Scholes関連関数
# -------------------
def d1_d2(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return d1, d2

def call_price(S, K, T, r, sigma):
    d1, d2 = d1_d2(S, K, T, r, sigma)
    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

def call_delta(S, K, T, r, sigma):
    d1, _ = d1_d2(S, K, T, r, sigma)
    return norm.cdf(d1)

def call_gamma(S, K, T, r, sigma):
    d1, _ = d1_d2(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

def call_zomma(S, K, T, r, sigma, dsigma=1e-4):
    """
    Gammaに対してsigmaでの数値微分をとりZommaを近似
    """
    g_plus  = call_gamma(S, K, T, r, sigma + dsigma)
    g_minus = call_gamma(S, K, T, r, sigma - dsigma)
    return (g_plus - g_minus) / (2*dsigma)

# -------------------
# パラメータ設定
# -------------------
K = 30.0
r = 0.0415
days_to_expiry = 21
T = days_to_expiry / 365.0

# プロットするSの範囲
S_min, S_max = 20.0, 40.0
S_range = np.linspace(S_min, S_max, 200)

# ★ここがポイント★
# ボラティリティと「注目するS(焦点S)」をアニメーションさせたい
# たとえば vol=60.7% から 27.7% に下がり、
#           S=28.56 から 29.61 へ上昇していく、としてみる
vol_start, vol_end = 0.607, 0.277    # 60.7% → 27.7%
S_focus_start, S_focus_end = 28.56, 29.61

n_frames = 60

# フレーム毎に Vol, S_focus を線形補間で変化させる
vol_array = np.linspace(vol_start, vol_end, n_frames)
S_focus_array = np.linspace(S_focus_start, S_focus_end, n_frames)

# GammaやZommaは値が小さいので拡大表示
gamma_scale = 10.0
zomma_scale = 100.0

# -------------------
# 図と軸の準備
# -------------------
fig, ax1 = plt.subplots(figsize=(8,6))

# 左軸: オプション価格
ax1.set_xlabel("Underlying Price (S)")
ax1.set_ylabel("Call Option Price", color="orange")
ax1.set_xlim(S_min, S_max)
ax1.tick_params(axis='y', labelcolor="orange")
ax1.grid(True)

# 右軸: Delta, Gamma, Zomma
ax2 = ax1.twinx()
ax2.set_ylabel("Greeks")

# とりあえず初期フレーム(=0)で曲線を計算して軸範囲を決める
prices_init = [call_price(s, K, T, r, vol_start) for s in S_range]
deltas_init = [call_delta(s, K, T, r, vol_start) for s in S_range]
gammas_init = [call_gamma(s, K, T, r, vol_start) for s in S_range]
zommas_init = [call_zomma(s, K, T, r, vol_start) for s in S_range]

p_min = min(0, min(prices_init)*1.1)
p_max = max(prices_init)*1.1
ax1.set_ylim(p_min, p_max)

g_vals = np.array(gammas_init)*gamma_scale
z_vals = np.array(zommas_init)*zomma_scale
d_vals = np.array(deltas_init)
low2 = min(g_vals.min(), z_vals.min(), d_vals.min())
high2 = max(g_vals.max(), z_vals.max(), d_vals.max())
ax2.set_ylim(low2*1.1, high2*1.1)

# -------------------
# 描画するラインオブジェクト
# -------------------
# 左軸(オプション価格)
line_price, = ax1.plot([], [], color="orange", label="Call Price")

# 右軸(Delta, Gamma, Zomma)
line_delta, = ax2.plot([], [], color="red",   label="Delta")
line_gamma, = ax2.plot([], [], color="green", label=f"Gamma x{gamma_scale}")
line_zomma, = ax2.plot([], [], color="blue",  label=f"Zomma x{zomma_scale}")

# 基準となる「注目点」 (黒丸と垂直線)
focus_point, = ax1.plot([], [], 'ko', label="Focus Point")
focus_line,  = ax1.plot([], [], color='gray', linestyle='--')

# 凡例
ax2.legend(
    [line_price, line_delta, line_gamma, line_zomma, focus_point],
    ["Call Price", "Delta", f"Gamma x{gamma_scale}", f"Zomma x{zomma_scale}", "Focus Point"],
    loc="upper left"
)

# タイトルのテキストを更新できるように変数化しておく
title_text = ax1.set_title("")

plt.tight_layout()

# -------------------
# アニメーション用関数
# -------------------
def init():
    """
    初期化関数: まだ何も描かない状態にしておく
    """
    line_price.set_data([], [])
    line_delta.set_data([], [])
    line_gamma.set_data([], [])
    line_zomma.set_data([], [])
    focus_point.set_data([], [])
    focus_line.set_data([], [])
    title_text.set_text("")
    return (line_price, line_delta, line_gamma, line_zomma,
            focus_point, focus_line, title_text)

def update(frame):
    """
    フレーム(frame)ごとに呼ばれる更新関数
    """
    # 現在のVol, S_focusを取得
    sigma = vol_array[frame]
    S_f   = S_focus_array[frame]

    # 全体の曲線を再計算 (S_min～S_max)
    prices = [call_price(s, K, T, r, sigma) for s in S_range]
    deltas = [call_delta(s, K, T, r, sigma) for s in S_range]
    gammas = [call_gamma(s, K, T, r, sigma) for s in S_range]
    zommas = [call_zomma(s, K, T, r, sigma) for s in S_range]

    # ライン更新
    line_price.set_data(S_range, prices)
    line_delta.set_data(S_range, deltas)
    line_gamma.set_data(S_range, np.array(gammas)*gamma_scale)
    line_zomma.set_data(S_range, np.array(zommas)*zomma_scale)

    # 注目点を計算(今のVol, 今のS_f のときのオプション価格など)
    price_f = call_price(S_f, K, T, r, sigma)
    delta_f = call_delta(S_f, K, T, r, sigma)

    # 黒丸(焦点)の位置
    focus_point.set_data([S_f], [price_f])

    # 垂直線は「下からオプション価格まで」でもよいし、
    # あるいはグラフ全体に伸ばしたいなら y軸の下端～上端を使う
    y_bottom, y_top = ax1.get_ylim()
    focus_line.set_data([S_f, S_f], [y_bottom, price_f])

    # 角度AやGamma = tan(A) などを実際に入れたい場合は自由に計算してください
    # 例: ざっくり Delta -> A_deg = Delta * 10度, など
    A_deg = delta_f * 10
    gamma_by_A = np.tan(np.deg2rad(A_deg))  # tg(A)

    # タイトル文字列の更新
    title_text.set_text(
        f"S={S_f:.2f}, Vol={sigma*100:.1f}%, "
        f"Delta={delta_f*100:.1f}%, A={A_deg:.3f}°, Gamma_by_A={gamma_by_A*100:.1f}%"
    )

    return (line_price, line_delta, line_gamma, line_zomma,
            focus_point, focus_line, title_text)

# アニメーション作成
ani = animation.FuncAnimation(
    fig,
    update,
    frames=n_frames,
    init_func=init,
    interval=300,  # 更新間隔(ms)
    blit=True
)

plt.show()

# Jupyter Notebook上などで実行する場合は:
# from IPython.display import HTML
# HTML(ani.to_jshtml())
