import os
import sys
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.distributions as dist
import torch.nn.functional as F

# ユーザー環境に合わせたパス設定
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from mongodb.data_loader_mongo import MongoDataLoader
from common.constants import *

# ========================================
# 1) データセット: 時系列を考慮したシーケンス化
# ========================================
class TimeSeriesDataset(Dataset):
    """
    seq_length個の連続した特徴量(X)を入力とし、
    その直後の1点の y を予測対象とする
    """
    def __init__(self, X, y, seq_length=30):
        """
        X: shape (N, num_features)
        y: shape (N,)
        seq_length: 過去何ステップを1サンプルとするか
        """
        self.X = X
        self.y = y
        self.seq_length = seq_length

        # シーケンス化したデータを格納するリスト
        self.data_X = []
        self.data_y = []

        # seq_lengthステップ分をまとめて1サンプルとし、ラベルはその次の時点のy
        for i in range(len(X) - seq_length):
            x_seq = X[i : i + seq_length]   # 過去 seq_length 分
            y_label = y[i + seq_length]     # 直後(または同時)の1点
            self.data_X.append(x_seq)
            self.data_y.append(y_label)

        self.data_X = np.array(self.data_X, dtype=np.float32)  # (N', seq_length, num_features)
        self.data_y = np.array(self.data_y, dtype=np.float32)  # (N',)

    def __len__(self):
        return len(self.data_X)

    def __getitem__(self, idx):
        return self.data_X[idx], self.data_y[idx]


# ========================================
# 2) ベイジアン線形レイヤー (最終出力用)
# ========================================
class BayesianLinear(nn.Module):
    """
    Mean-Field近似によるベイジアン線形レイヤーの簡易実装
    weight_mu, weight_logvar, bias_mu, bias_logvar を学習
    forward時にガウス分布からサンプリングした重み・バイアスで演算
    """
    def __init__(self, in_features, out_features, prior_sigma=1.0):
        super().__init__()
        # 重みの平均
        self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features))
        # 重みの対数分散
        self.weight_logvar = nn.Parameter(torch.full((out_features, in_features), -5.0))

        # バイアスの平均
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        # バイアスの対数分散
        self.bias_logvar = nn.Parameter(torch.full((out_features,), -5.0))

        # 事前分布（平均0, 分散=prior_sigma^2の正規分布）
        self.prior = dist.Normal(
            torch.zeros_like(self.weight_mu),
            prior_sigma * torch.ones_like(self.weight_logvar)
        )

    def forward(self, x):
        """
        通常の確率的 forward: weight, bias をランダムサンプリング
        """
        weight_sigma = torch.exp(0.5 * self.weight_logvar)  # 分散 -> 標準偏差
        bias_sigma   = torch.exp(0.5 * self.bias_logvar)

        eps_w = torch.randn_like(weight_sigma)
        eps_b = torch.randn_like(bias_sigma)

        weight = self.weight_mu + weight_sigma * eps_w
        bias   = self.bias_mu   + bias_sigma   * eps_b

        return torch.addmm(bias, x, weight.t())

    def forward_with_mean(self, x):
        """
        (2) 平均パラメータのみ使用する forward:
        weight_mu, bias_mu をそのまま使い、サンプリングしない
        """
        weight = self.weight_mu
        bias   = self.bias_mu
        return torch.addmm(bias, x, weight.t())

    def kl_loss(self):
        """
        事後分布 q(w|θ) と事前分布 p(w) のKLダイバージェンス
        """
        post_weight = dist.Normal(self.weight_mu, torch.exp(0.5 * self.weight_logvar))
        post_bias   = dist.Normal(self.bias_mu,   torch.exp(0.5 * self.bias_logvar))

        kl_weight = dist.kl_divergence(post_weight, self.prior).sum()
        kl_bias   = dist.kl_divergence(post_bias, dist.Normal(0, 1)).sum()

        return kl_weight + kl_bias


# ========================================
# 3) LSTM + BayesianLinearモデル
# ========================================
class LSTMWithBayesianOutput(nn.Module):
    """
    LSTMで時系列を処理し、最後にBayesianLinearで [mu, log_var] を出力
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=1):
        super().__init__()
        # LSTM部分（通常の確率的ではないLSTM）
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        # 出力は [mu, log_var] = 2次元
        self.blinear = BayesianLinear(hidden_dim, 2)

    def forward(self, x):
        """
        (1) 通常の確率的 forward: BayesianLinear.forward() でサンプリング
        """
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_out = lstm_out[:, -1, :]  # (batch_size, hidden_dim)
        out = self.blinear(last_out)   # (batch_size, 2)
        return out

    def forward_with_mean(self, x):
        """
        (2) 平均パラメータのみ使用する forward
        """
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        out = self.blinear.forward_with_mean(last_out)
        return out

    def kl_loss(self):
        """
        最終出力層(BayesianLinear)の KL損失を返す
        """
        return self.blinear.kl_loss()


# -----------------------------
# (1) 複数サンプリングして平均を取る関数
# -----------------------------
def multiple_sampling_predict(model, x, n_samples=10):
    """
    同じ入力 x に対して n_samples 回サンプリングし、
    mu と var を平均化して返す
    """
    mu_accum = 0.0
    var_accum = 0.0

    for _ in range(n_samples):
        out = model(x)  # (batch_size, 2)
        mu = out[:, 0]
        log_var = out[:, 1]
        log_var = torch.clamp(log_var, min=-10, max=10)
        var = F.softplus(log_var)  # log_var -> var (softplus)

        mu_accum += mu
        var_accum += var

    mu_avg = mu_accum / n_samples
    var_avg = var_accum / n_samples
    return mu_avg, var_avg


# ========================================
# 4) メイン実行
# ========================================
def main():
    # ------- データ読み込み & 前処理 ---------
    db = MongoDataLoader()
    df = db.load_data_from_datetime_period(
        datetime(2024, 1, 1),
        datetime(2025, 3, 1),
        coll_type=MARKET_DATA_TECH,
        symbol='BTCUSDT',
        interval=60
    )

    # 分析に利用するカラムのみ抽出
    graph_df = df[['start_at', 'close', 'volume', 'macdhist', 'rsi', 'volatility', 'mfi']].dropna()
    graph_df.sort_values('start_at', inplace=True)
    graph_df.reset_index(drop=True, inplace=True)

    # -- 特徴量とターゲット --
    lag_feature_cols = [ 'close', 'volume', 'macdhist', 'rsi', 'volatility', 'mfi']
    X_raw = graph_df[lag_feature_cols].values  # 説明変数
    y_raw = graph_df['close'].values             # 予測対象（ここではcloseを予測例に）

    # 8:2 に時系列順で分割
    split_idx = int(len(X_raw) * 0.8)
    X_train_raw, X_test_raw = X_raw[:split_idx], X_raw[split_idx:]
    y_train_raw, y_test_raw = y_raw[:split_idx], y_raw[split_idx:]

    # --- スケーリング ---
    from sklearn.preprocessing import StandardScaler
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train_raw)
    X_test_scaled  = scaler_X.transform(X_test_raw)

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train_raw.reshape(-1, 1)).flatten()
    y_test_scaled  = scaler_y.transform(y_test_raw.reshape(-1, 1)).flatten()

    # -- 時系列シーケンス化 --
    seq_length = 127  # 過去127ステップを入力
    train_ds = TimeSeriesDataset(X_train_scaled, y_train_scaled, seq_length=seq_length)
    test_ds  = TimeSeriesDataset(X_test_scaled,  y_test_scaled,  seq_length=seq_length)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=64, shuffle=False)

    # ------- モデル構築 ----------
    input_dim = X_train_scaled.shape[1]
    hidden_dim = 64
    model = LSTMWithBayesianOutput(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=1)

    # ------- 損失関数 / オプティマイザ ----------
    def negative_log_likelihood_and_kl(output, target, kl_weight=0.01):
        """
        output: shape (batch_size, 2) → [mu, log_var]
        target: shape (batch_size, )
        kl_weightを少し小さめ(0.1など)にしてサンプリングの変動を抑制
        """
        mu      = output[:, 0]
        log_var = output[:, 1]

        log_var = torch.clamp(log_var, min=-10, max=10)
        var = F.softplus(log_var)
        sigma = torch.sqrt(var + 1e-8)

        # 予測分布: Normal(mu, sigma)
        pred_dist = dist.Normal(mu, sigma)
        nll = -pred_dist.log_prob(target).mean()

        kl = model.kl_loss() * kl_weight
        return nll + kl

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # ------- 学習ループ ----------
    n_epochs = 30
    model.train()
    for epoch in range(n_epochs):
        total_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            out = model(batch_X)   # (batch_size, 2)
            loss = negative_log_likelihood_and_kl(out, batch_y)
            loss.backward()

            # 勾配クリッピング
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{n_epochs}] - Loss: {avg_loss:.4f}")

    # ------- テストデータで予測 ----------
    model.eval()

    pred_means_multi = []
    pred_stds_multi  = []
    pred_means_mean  = []
    pred_stds_mean   = []
    all_targets      = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            # ---------------------------
            # (1) 複数サンプリングで平均
            # ---------------------------
            mu_avg, var_avg = multiple_sampling_predict(model, batch_X, n_samples=10)
            # shape: (batch_size, ), (batch_size, )

            # ---------------------------
            # (2) 平均パラメータのみ使用
            # ---------------------------
            out_mean = model.forward_with_mean(batch_X)  # (batch_size, 2)
            mu_mean = out_mean[:, 0]
            log_var_mean = out_mean[:, 1]
            log_var_mean = torch.clamp(log_var_mean, -10, 10)
            var_mean = F.softplus(log_var_mean)

            # ここで numpy に変換
            mu_avg_np = mu_avg.cpu().numpy()
            var_avg_np = var_avg.cpu().numpy()
            mu_mean_np = mu_mean.cpu().numpy()
            var_mean_np = var_mean.cpu().numpy()

            # 標準化を元に戻す
            y_scale = scaler_y.scale_[0]
            y_mean  = scaler_y.mean_[0]

            # (1) multiple-sampling の予測
            pred_mean_multi_org = mu_avg_np * y_scale + y_mean
            pred_std_multi_org  = np.sqrt(var_avg_np + 1e-8) * y_scale

            # (2) mean-only の予測
            pred_mean_mean_org = mu_mean_np * y_scale + y_mean
            pred_std_mean_org  = np.sqrt(var_mean_np + 1e-8) * y_scale

            # リストに格納
            pred_means_multi.append(pred_mean_multi_org)
            pred_stds_multi.append(pred_std_multi_org)

            pred_means_mean.append(pred_mean_mean_org)
            pred_stds_mean.append(pred_std_mean_org)

            all_targets.append(batch_y.cpu().numpy())

    # 連結して1次元に
    pred_means_multi = np.concatenate(pred_means_multi, axis=0)
    pred_stds_multi  = np.concatenate(pred_stds_multi, axis=0)
    pred_means_mean  = np.concatenate(pred_means_mean, axis=0)
    pred_stds_mean   = np.concatenate(pred_stds_mean, axis=0)
    all_targets      = np.concatenate(all_targets, axis=0)
    targets_org      = all_targets * y_scale + y_mean

    # ------- 可視化 ----------
    # テストデータの DataFrame を再構築（split_idx と同じ区間）
    test_df = graph_df.iloc[split_idx:].copy().reset_index(drop=True)
    # TimeSeriesDataset でのシーケンス化により、最初の seq_length 分は予測対象ではない
    timestamps = test_df['start_at'].iloc[seq_length:].values

    plt.figure(figsize=(12, 6))

    # テストデータの実際の終値
    plt.plot(
        test_df['start_at'],
        test_df['close'],
        label='True Price (Test Data)',
        color='blue'
    )

    # (1) 複数サンプリング予測
    plt.plot(
        timestamps,
        pred_means_multi,
        label='Predicted Mean (Multi-sample)',
        color='green'
    )
    plt.fill_between(
        timestamps,
        pred_means_multi - 2*pred_stds_multi,
        pred_means_multi + 2*pred_stds_multi,
        color='green',
        alpha=0.2,
        label='Uncertainty ±2σ (Multi-sample)'
    )

    # (2) 平均パラメータのみ使用
    plt.plot(
        timestamps,
        pred_means_mean,
        label='Predicted Mean (Mean-only)',
        color='red'
    )
    plt.fill_between(
        timestamps,
        pred_means_mean - 2*pred_stds_mean,
        pred_means_mean + 2*pred_stds_mean,
        color='red',
        alpha=0.2,
        label='Uncertainty ±2σ (Mean-only)'
    )

    plt.title('BTC Price Prediction with Bayesian LSTM Output')
    plt.xlabel('Time')
    plt.ylabel('BTC Price (close)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
