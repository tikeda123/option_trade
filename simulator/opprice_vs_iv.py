import pandas as pd
import matplotlib.pyplot as plt


def main():
    """
    オプションデータの可視化を行う関数
    """
# CSVデータの読み込み
    df = pd.read_csv("cleaned_option_data.csv")

# 対象シンボルでフィルタリング
    df = df[df['symbol'] == 'BTC-28MAR25-95000-P']

    # 必要なカラムの抽出と時刻順にソート
    df = df[['date', 'ask1Price', 'bid1Price', 'ask1Iv', 'bid1Iv']]
    df = df.sort_values(by='date')

    # 'date'列をdatetime型に変換
    df['date'] = pd.to_datetime(df['date'])

    # プロットの作成
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 左側のy軸: Price系の値をプロット
    ax1.set_xlabel('時刻')
    ax1.set_ylabel('Price')
    line1, = ax1.plot(df['date'], df['ask1Price'], label='ask1Price', color='red')
    line2, = ax1.plot(df['date'], df['bid1Price'], label='bid1Price', color='blue')
    ax1.tick_params(axis='y')

    # 右側のy軸: IV系の値をプロット
    ax2 = ax1.twinx()
    ax2.set_ylabel('IV')
    line3, = ax2.plot(df['date'], df['ask1Iv'], label='ask1Iv', color='orange')
    line4, = ax2.plot(df['date'], df['bid1Iv'], label='bid1Iv', color='green')
    ax2.tick_params(axis='y')

    # 両軸のレジェンドを統合
    lines = [line1, line2, line3, line4]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper left')

    plt.title("BTC-28MAR25-95000-P オプションデータ")
    plt.show()

if __name__ == "__main__":
    main()
