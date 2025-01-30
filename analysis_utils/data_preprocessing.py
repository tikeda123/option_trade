import pandas as pd
import numpy as np

def remove_outliers_iqr(df: pd.DataFrame, col: str, factor: float = 1.5) -> pd.DataFrame:
    """
    特定カラムの外れ値を IQR に基づいて検出し、nan に置き換える。

    Parameters
    ----------
    df : pd.DataFrame
        データフレーム
    col : str
        外れ値判定をするカラム名
    factor : float, optional
        IQRに乗じる係数, by default 1.5

    Returns
    -------
    pd.DataFrame
        外れ値を nan に置き換えた DataFrame
    """
    df = df.copy()
    df[col] = pd.to_numeric(df[col], errors='coerce')
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    # 上限を設定したい場合は上界を求めて df[col] > upper_bound も nan 置換するとよい
    df.loc[df[col] < lower_bound, col] = np.nan
    return df

def fill_zeros_and_interpolate(group: pd.DataFrame,
                               columns_to_clean: list,
                               outlier_factor: float = 1.5) -> pd.DataFrame:
    """
    指定したカラムについて 0 を nan に変換し、IQRによる外れ値処理 → 補間を行う。
    groupby.apply() でグループごとに適用する想定。

    Parameters
    ----------
    group : pd.DataFrame
        グループ化後の DataFrame
    columns_to_clean : list
        クリーニング対象のカラム名リスト
    outlier_factor : float
        remove_outliers_iqr の factor 値

    Returns
    -------
    pd.DataFrame
        前処理済みの DataFrame
    """
    # 時系列順に整列
    group = group.sort_values('date').copy()

    # 0 の値を nan にする & 外れ値処理
    for col in columns_to_clean:
        group.loc[group[col] == 0, col] = np.nan
        group = remove_outliers_iqr(group, col=col, factor=outlier_factor)

    # 時系列インデックスにして補間
    group.set_index('date', inplace=True)
    for col in columns_to_clean:
        # 時間ベースで補間
        group[col] = group[col].interpolate(method='time')
        group[col].fillna(method='ffill', inplace=True)
        group[col].fillna(method='bfill', inplace=True)

    group.reset_index(inplace=True)
    return group

def clean_option_data(df: pd.DataFrame,
                      group_col: str = 'symbol',
                      columns_to_clean: list = None,
                      outlier_factor: float = 1.5,
                      dropna_after: bool = True) -> pd.DataFrame:
    """
    オプションデータ全体に対して、0→nan 変換 & 外れ値処理 & 補間 を行い、
    必要に応じて dropna まで行うラッパ関数。

    Parameters
    ----------
    df : pd.DataFrame
        生データフレーム
    group_col : str, optional
        groupby をする対象カラム (symbol など), by default 'symbol'
    columns_to_clean : list, optional
        クリーニング対象のカラムリスト, by default None
    outlier_factor : float, optional
        remove_outliers_iqr の factor 値, by default 1.5
    dropna_after : bool, optional
        処理後に dropna(subset=columns_to_clean) するかどうか, by default True

    Returns
    -------
    pd.DataFrame
        前処理済みデータフレーム
    """
    if columns_to_clean is None:
        columns_to_clean = ['ask1Price', 'ask1Iv', 'bid1Price', 'bid1Iv']

    # 日付を datetime に変換（再利用想定のためここで実施）
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # グループ単位で 0→nan, 外れ値処理, 補間
    df = df.groupby(group_col, group_keys=False).apply(
        fill_zeros_and_interpolate,
        columns_to_clean=columns_to_clean,
        outlier_factor=outlier_factor
    )

    # 処理後に NaN が残った行を削除
    if dropna_after:
        before_len = len(df)
        df.dropna(subset=columns_to_clean, inplace=True)
        after_len = len(df)
        print(f"[clean_option_data] dropna: {before_len - after_len} rows removed")

    return df
