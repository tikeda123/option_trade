# dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    """
    seq_length個の連続した特徴量(X)を入力とし、
    その直後の1点の y を予測対象とする時系列用Dataset。
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
        data_X = []
        data_y = []

        for i in range(len(X) - seq_length):
            x_seq = X[i : i + seq_length]   # 過去 seq_length 分
            y_label = y[i + seq_length]     # 直後(または同時)の1点
            data_X.append(x_seq)
            data_y.append(y_label)

        self.data_X = np.array(data_X, dtype=np.float32)  # (N', seq_length, num_features)
        self.data_y = np.array(data_y, dtype=np.float32)  # (N',)

    def __len__(self):
        return len(self.data_X)

    def __getitem__(self, idx):
        return self.data_X[idx], self.data_y[idx]
