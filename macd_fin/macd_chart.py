import os
import sys
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# ユーザー環境に合わせたパス設定
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from mongodb.data_loader_mongo import MongoDataLoader
from common.constants import *


def validate_macd_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, bool, dict]:
    """
    MACDデータの整合性を検証し、必要に応じて修正する関数

    Parameters:
    df (pd.DataFrame): 検証するデータフレーム

    Returns:
    Tuple[pd.DataFrame, bool, dict]:
        - 検証・修正後のデータフレーム
        - 修正が行われたかどうかのフラグ
        - 検証結果の詳細を含む辞書
    """
    validation_results = {
        "missing_data": False,
        "infinite_values": False,
        "macdhist_discrepancy": False,
        "discrepancy_stats": {},
        "action_taken": "none"
    }

    modified = False
    df_validated = df.copy()

    # 1. 欠損値のチェック
    missing_count = df.isna().sum()
    if missing_count.sum() > 0:
        validation_results["missing_data"] = True
        validation_results["missing_count"] = missing_count.to_dict()
        print("警告: データに欠損値があります")
        print(missing_count)

        # 欠損値の処理（前方向補間で埋める）
        df_validated = df_validated.fillna(method='ffill')
        if df_validated.isna().sum().sum() > 0:
            # 前方向で埋められなかった場合は後方向補間も試みる
            df_validated = df_validated.fillna(method='bfill')

        modified = True
        validation_results["action_taken"] = "欠損値を補間しました"

    # 2. 無限値のチェック
    inf_mask = df_validated.isin([np.inf, -np.inf])
    inf_count = inf_mask.sum()
    if inf_count.sum() > 0:
        validation_results["infinite_values"] = True
        validation_results["infinite_count"] = inf_count.to_dict()
        print("警告: データに無限値があります")
        print(inf_count)

        # 無限値をNaNに変換してから前方向補間
        df_validated = df_validated.replace([np.inf, -np.inf], np.nan)
        df_validated = df_validated.fillna(method='ffill').fillna(method='bfill')

        modified = True
        validation_results["action_taken"] = "無限値を処理しました"

    # 3. MACDヒストグラムの検証
    # 理論上のMACDヒストグラム値を計算
    df_validated['calculated_macdhist'] = df_validated['kalman_macd'] - df_validated['kalman_macdsignal']

    # 既存のヒストグラム値と計算値の差を計算
    df_validated['hist_diff'] = (df_validated['kalman_macdhist'] - df_validated['calculated_macdhist']).abs()

    # 差の統計を計算
    diff_mean = df_validated['hist_diff'].mean()
    diff_max = df_validated['hist_diff'].max()
    significant_diff_count = (df_validated['hist_diff'] > 0.1).sum()  # 0.1以上の差がある行数

    validation_results["discrepancy_stats"] = {
        "mean_difference": diff_mean,
        "max_difference": diff_max,
        "significant_diff_count": significant_diff_count,
        "total_rows": len(df_validated)
    }

    # ヒストグラム値に問題があるかどうかを判定
    has_discrepancy = diff_mean > 0.01 or significant_diff_count > len(df_validated) * 0.01

    if has_discrepancy:
        validation_results["macdhist_discrepancy"] = True
        print(f"警告: MACDヒストグラムの値に不一致があります")
        print(f"  平均差: {diff_mean:.4f}")
        print(f"  最大差: {diff_max:.4f}")
        print(f"  不一致のレコード数: {significant_diff_count} / {len(df_validated)} ({significant_diff_count/len(df_validated)*100:.2f}%)")

        # 問題がある場合は再計算値で置き換え
        df_validated['kalman_macdhist'] = df_validated['calculated_macdhist']
        modified = True
        validation_results["action_taken"] = "MACDヒストグラム値を修正しました"

    # 検証用の列を削除
    df_validated = df_validated.drop(['calculated_macdhist', 'hist_diff'], axis=1, errors='ignore')

    # 最終チェック - データの範囲が適切かどうか
    macd_range = df_validated['kalman_macd'].max() - df_validated['kalman_macd'].min()
    hist_range = df_validated['kalman_macdhist'].max() - df_validated['kalman_macdhist'].min()

    if hist_range < macd_range * 0.1:
        print("警告: MACDヒストグラムの範囲がMACDラインに比べて極端に小さいです")
        validation_results["range_issue"] = True

    # 検証結果のサマリーを出力
    print("\n** データ検証の結果 **")
    if modified:
        print("データに問題があり、修正が行われました")
    else:
        print("データは正常です")

    # 各データ列の統計情報
    print("\n基本統計量:")
    stats = df_validated[['kalman_macd', 'kalman_macdsignal', 'kalman_macdhist']].describe()
    print(stats)

    return df_validated, modified, validation_results


def plot_macd_chart(df):
    """
    BTCの価格とMACDインジケーターを可視化する関数

    Parameters:
    df (pd.DataFrame): 'start_at', 'close', 'kalman_macdhist', 'kalman_macd', 'kalman_macdsignal'カラムを持つデータフレーム
    """
    # 日付型に変換
    df['start_at'] = pd.to_datetime(df['start_at'])

    # フィギュアサイズの設定
    plt.figure(figsize=(16, 10))

    # スタイル設定
    sns.set_style('darkgrid')

    # サブプロット1: BTC価格チャート
    ax1 = plt.subplot2grid((8, 1), (0, 0), rowspan=5, colspan=1)
    ax1.plot(df['start_at'], df['close'], color='#1f77b4', linewidth=2, label='BTC/USDT')
    ax1.set_title('BTC/USDT価格チャート (2023-2025)', fontsize=15, fontweight='bold')
    ax1.set_ylabel('価格 (USDT)', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # X軸のフォーマット設定
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)

    # サブプロット2: MACDチャート
    ax2 = plt.subplot2grid((8, 1), (5, 0), rowspan=3, colspan=1, sharex=ax1)

    # MACDライン
    ax2.plot(df['start_at'], df['kalman_macd'], color='#ff7f0e', linewidth=1.5, label='MACD')

    # シグナルライン
    ax2.plot(df['start_at'], df['kalman_macdsignal'], color='#2ca02c', linewidth=1.5, label='シグナル')

    # ヒストグラム（バーチャート）
    hist_colors = ['#d62728' if x < 0 else '#1f77b4' for x in df['kalman_macdhist']]

    # 日付をバーのx軸位置に変換
    x = mdates.date2num(df['start_at'])

    # 各バーの幅を計算
    if len(x) > 1:
        width = 0.8 * (x[1] - x[0])
    else:
        width = 1.0

    # ヒストグラムをバーチャートとして描画
    ax2.bar(df['start_at'], df['kalman_macdhist'], width=width, color=hist_colors, alpha=0.7, label='ヒストグラム')

    # ゼロライン
    ax2.axhline(0, color='black', linestyle='--', alpha=0.3)

    # MACDチャートのラベルと凡例
    ax2.set_title('MACDインジケーター', fontsize=13)
    ax2.set_ylabel('MACD値', fontsize=12)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    # Y軸の範囲を適切に設定
    macd_max = max(df['kalman_macd'].max(), df['kalman_macdsignal'].max(), df['kalman_macdhist'].max())
    macd_min = min(df['kalman_macd'].min(), df['kalman_macdsignal'].min(), df['kalman_macdhist'].min())
    y_margin = (macd_max - macd_min) * 0.1
    ax2.set_ylim(macd_min - y_margin, macd_max + y_margin)

    # X軸のラベル
    ax2.set_xlabel('日付', fontsize=12)

    # レイアウトの調整
    plt.tight_layout()

    # 保存
    plt.savefig('btc_macd_analysis.png', dpi=300, bbox_inches='tight')

    # 表示
    plt.show()


def main():
    # MongoDBからBTCの時系列データを取得
    db = MongoDataLoader()
    print("MongoDBからデータを取得中...")
    df = db.load_data_from_datetime_period(datetime(2025, 2, 20),
                                          datetime(2025, 3, 1),
                                          coll_type=MARKET_DATA_TECH,
                                          symbol='BTCUSDT',
                                          interval=60)

    # 利用するカラムのみ抽出
    graph_df = df[['start_at', 'close', 'kalman_macdhist', 'kalman_macd', 'kalman_macdsignal']]

    print(f"取得したデータ: {len(graph_df)}行 x {len(graph_df.columns)}列")

    # データの冒頭を確認
    print("\nデータサンプル:")
    print(graph_df.head())

    # チャート描画前にデータを検証
    print("\n=== MACDデータの検証を実行中 ===")
    validated_df, was_modified, validation_results = validate_macd_data(graph_df)

    if was_modified:
        print("\n修正後のデータサンプル:")
        print(validated_df.head())

        # 修正前と修正後の比較
        if validation_results.get("macdhist_discrepancy", False):
            print("\n修正前後のMACDヒストグラム値の比較 (最初の5件):")
            compare_df = pd.DataFrame({
                'start_at': graph_df['start_at'].head(),
                'original_macdhist': graph_df['kalman_macdhist'].head(),
                'corrected_macdhist': validated_df['kalman_macdhist'].head(),
                'difference': (graph_df['kalman_macdhist'] - validated_df['kalman_macdhist']).head()
            })
            print(compare_df)

    # 検証データを使用してMACDチャートをプロット
    print("\n=== MACDチャートを作成中 ===")
    plot_macd_chart(validated_df)

    print("\nチャートが生成され、'btc_macd_analysis.png'として保存されました。")


if __name__ == "__main__":
    main()