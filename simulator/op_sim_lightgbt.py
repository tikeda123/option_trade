import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# ユーザー環境に合わせたパス設定
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from mongodb.data_loader_mongo import MongoDataLoader
from common.constants import *

def load_and_preprocess_data():
    # MongoDBからBTCの時系列データを取得
    db = MongoDataLoader()
    df = db.load_data_from_datetime_period(
        datetime(2023, 1, 1),
        datetime(2025, 1, 1),
        coll_type=MARKET_DATA_TECH,
        symbol='BTCUSDT',
        interval=60
    )

    # 利用するカラムのみ抽出
    graph_df = df[['start_at', 'close', 'ema', 'macdhist', 'roc', 'mfi', 'aroon', 'volatility']]

    # --- NStep前のデータ抽出と比較データの作成 ---
    NSTEP = 3  # NStepの間隔（必要に応じて変更可能）

    # 時系列順にソート
    graph_df.sort_values('start_at', inplace=True)
    graph_df.reset_index(drop=True, inplace=True)

    # 利用するカラムリスト（start_at以外）
    #cols = ['close', 'ema', 'macdhist', 'roc', 'mfi', 'aroon', 'volatility']
    cols = ['close', 'volatility']


    # (A) 各カラムのNStep前の値を抽出（新規列として追加）
    for col in cols:
        graph_df[f'{col}_NSTEP'] = graph_df[col].shift(NSTEP)

    # (B) 各カラムの現在の値とNStep前の値を比較
    #     【目的変数】 closeの比較結果：値が上昇→1、下降→0
    graph_df['close_binary'] = (graph_df['close'] > graph_df['close'].shift(NSTEP)).astype(int)

    # 【説明変数】 ① NStep前の生データ（全カラム）＋② close以外の比較結果（二値）
    feature_cols = []
    # ① NStep前の生データ
    for col in cols:
        feature_cols.append(f'{col}_NSTEP')

    # ② close以外の比較（二値）データの作成
    for col in cols:
        if col == 'close':
            continue
        graph_df[f'{col}_binary'] = (graph_df[col] > graph_df[col].shift(NSTEP)).astype(int)
        feature_cols.append(f'{col}_binary')

    # シフトによるNaN行の削除
    graph_df.dropna(inplace=True)

    X = graph_df[feature_cols]
    y = graph_df['close_binary']

    return X, y

def train_model(X_train, y_train):
    from lightgbm import LGBMClassifier
    model = LGBMClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    y_pred = model.predict(X_test)

    # 指定された評価指標の算出
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    return accuracy, report, conf_matrix

def plot_feature_importance(model, feature_names):
    # LightGBMの特徴量重要度を取得し、棒グラフで表示
    importance = model.feature_importances_
    # 重要度の低い順にソート
    sorted_idx = np.argsort(importance)

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(importance)), importance[sorted_idx], align='center')
    plt.yticks(range(len(importance)), [feature_names[i] for i in sorted_idx])
    plt.xlabel('Feature Importance')
    plt.title('LightGBM Feature Importance')
    plt.tight_layout()
    plt.show()

def main():
    X, y = load_and_preprocess_data()

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = train_model(X_train, y_train)

    # 評価結果の表示
    accuracy, report, conf_matrix = evaluate_model(model, X_test, y_test)
    print("Accuracy:", accuracy)
    print("Classification Report:")
    print(report)
    print("Confusion Matrix:")
    print(conf_matrix)

    # 特徴量重要度のグラフ表示
    plot_feature_importance(model, X.columns.tolist())

if __name__ == "__main__":
    main()


