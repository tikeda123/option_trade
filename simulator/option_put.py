import matplotlib.pyplot as plt

# パラメータ設定
initial_price = 95000   # 現物の購入価格
strike = 90000          # プットの行使価格
premium = 2000          # プットのプレミアムコスト
quantity = 1            # 保有量（BTC数）

# シミュレーションする将来価格帯（70,000～120,000ドルまで1,000ドル刻み）
future_prices = range(70000, 120001, 1000)

profits = []

for S in future_prices:
    # ロング現物損益
    long_pnl = (S - initial_price) * quantity

    # プットオプション損益
    put_exercise_value = max(0, strike - S) * quantity
    put_pnl = put_exercise_value - premium * quantity

    # 総合損益
    total_pnl = long_pnl + put_pnl
    profits.append(total_pnl)

# 結果表示
for S, pnl in zip(future_prices, profits):
    print(f"Future Price: {S:6d} USD | Total P&L: {pnl:8.2f} USD")

# グラフ描画
plt.figure(figsize=(10,5))
plt.plot(future_prices, profits, label='Protective Put P&L')
plt.axhline(0, color='black', linewidth=1)
plt.title('Protective Put Strategy P&L')
plt.xlabel('Future BTC Price (USD)')
plt.ylabel('Profit/Loss (USD)')
plt.grid(True)
plt.legend()
plt.show()
