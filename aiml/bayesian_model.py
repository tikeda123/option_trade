# bayesian_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

class BayesianLinear(nn.Module):
    """
    Mean-Field近似によるベイジアン線形レイヤーの簡易実装
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
        サンプリングした重み・バイアスで演算
        """
        weight_sigma = torch.exp(0.5 * self.weight_logvar)
        bias_sigma   = torch.exp(0.5 * self.bias_logvar)

        eps_w = torch.randn_like(weight_sigma)
        eps_b = torch.randn_like(bias_sigma)

        weight = self.weight_mu + weight_sigma * eps_w
        bias   = self.bias_mu   + bias_sigma   * eps_b

        return torch.addmm(bias, x, weight.t())

    def forward_with_mean(self, x):
        """
        平均パラメータのみ使用
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

        # 重みのKL
        kl_weight = dist.kl_divergence(post_weight, self.prior).sum()
        # バイアスのKL (事前分布は標準正規)
        kl_bias   = dist.kl_divergence(post_bias, dist.Normal(0, 1)).sum()

        return kl_weight + kl_bias

class LSTMWithBayesianOutput(nn.Module):
    """
    LSTMで時系列を処理し、最後にBayesianLinearで [mu, log_var] を出力
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=1):
        super().__init__()
        # LSTM部分（標準のLSTM）
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
        確率的 forward（BayesianLinearでサンプリング）
        """
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]  # (batch_size, hidden_dim)
        out = self.blinear(last_out)   # (batch_size, 2)
        return out

    def forward_with_mean(self, x):
        """
        平均パラメータのみ使用する forward
        """
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        out = self.blinear.forward_with_mean(last_out)
        return out

    def kl_loss(self):
        """
        最終出力層(BayesianLinear)の KL損失
        """
        return self.blinear.kl_loss()

def multiple_sampling_predict(model, x, n_samples=10):
    """
    同じ入力 x に対して n_samples 回サンプリングし、
    mu と var を平均化して返す
    """
    import torch.nn.functional as F

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

# train_utils.py

def negative_log_likelihood_and_kl(output, target, model, kl_weight=0.01):
    """
    output: shape (batch_size, 2) → [mu, log_var]
    target: shape (batch_size, )
    model:  LSTMWithBayesianOutputなど
    kl_weight: KL項の重みづけ
    """
    mu      = output[:, 0]
    log_var = output[:, 1]

    # log_varが極端に大きく/小さくならないようクリップ
    log_var = torch.clamp(log_var, min=-10, max=10)
    var = F.softplus(log_var)
    sigma = torch.sqrt(var + 1e-8)

    # 予測分布: Normal(mu, sigma)
    pred_dist = dist.Normal(mu, sigma)
    nll = -pred_dist.log_prob(target).mean()

    # ベイジアン線形層のKL損失
    kl = model.kl_loss() * kl_weight
    return nll + kl
