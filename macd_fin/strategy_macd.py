# strategy_macd.py
import pandas as pd

def generate_macd_signals(
    df: pd.DataFrame,
    macdhist_col: str = "kalman_macdhist"
) -> pd.DataFrame:
    """
    MACDヒストグラムの値をもとに以下のポジションを返す。
     - +1: ロング
     - -1: ショート
     -  0: ノーポジ

    シグナル判定は「MACDヒストグラムが正ならロング、負ならショート、ゼロ近辺は様子見」という単純ロジック。
    ただし、実際の約定はルックアヘッドバイアスを避けるため、
    前日の MACDヒスト値を使って「翌日」エントリーというフローに最適化されるのが一般的。
    """

    # 前提: DFは時系列順にソートされているとする
    df = df.copy()
    df['signal'] = 0  # デフォルトはノーポジ
    df['signal'] = df[macdhist_col].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

    return df
