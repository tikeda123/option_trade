interface OptionParams {
  currentPrice: number
  strikePrice: number
  daysToExpiry: number
  volatility: number
  optionType: 'call' | 'put'
}

// 標準正規分布の累積分布関数
function normalCDF(x: number): number {
  const a1 = 0.254829592
  const a2 = -0.284496736
  const a3 = 1.421413741
  const a4 = -1.453152027
  const a5 = 1.061405429
  const p = 0.3275911

  const sign = x < 0 ? -1 : 1
  x = Math.abs(x) / Math.sqrt(2.0)

  const t = 1.0 / (1.0 + p * x)
  const y =
    ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * Math.exp(-x * x)

  return 0.5 * (1.0 + sign * (1.0 - y))
}

// 標準正規分布の確率密度関数
function normalPDF(x: number): number {
  return (1 / Math.sqrt(2 * Math.PI)) * Math.exp(-0.5 * x * x)
}

export function calculateOptionMetrics(params: OptionParams) {
  const { currentPrice, strikePrice, daysToExpiry, volatility, optionType } = params
  const r = 0.02 // リスクフリーレート
  const t = daysToExpiry / 365 // 年換算
  const sigma = volatility

  const d1 =
    (Math.log(currentPrice / strikePrice) +
      (r + (sigma * sigma) / 2) * t) /
    (sigma * Math.sqrt(t))
  const d2 = d1 - sigma * Math.sqrt(t)

  let delta, gamma, theta, vega, price, pnl

  if (optionType === 'call') {
    delta = normalCDF(d1)
    price =
      currentPrice * normalCDF(d1) -
      strikePrice * Math.exp(-r * t) * normalCDF(d2)
  } else {
    delta = normalCDF(d1) - 1
    price =
      strikePrice * Math.exp(-r * t) * normalCDF(-d2) -
      currentPrice * normalCDF(-d1)
  }

  gamma =
    normalPDF(d1) / (currentPrice * sigma * Math.sqrt(t))

  vega = currentPrice * Math.sqrt(t) * normalPDF(d1) / 100

  const theta1 =
    -(currentPrice * sigma * normalPDF(d1)) /
    (2 * Math.sqrt(t))
  const theta2 = r * strikePrice * Math.exp(-r * t)

  if (optionType === 'call') {
    theta = (-theta1 - theta2 * normalCDF(d2)) / 365
  } else {
    theta = (-theta1 + theta2 * normalCDF(-d2)) / 365
  }

  // 損益計算（プレミアムを100とする）
  const premium = 100
  if (optionType === 'call') {
    pnl = Math.max(currentPrice - strikePrice, 0) - premium
  } else {
    pnl = Math.max(strikePrice - currentPrice, 0) - premium
  }

  return {
    price,
    pnl,
    delta,
    gamma,
    theta,
    vega,
  }
}