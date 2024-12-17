'use client'

import { useState, useEffect, useMemo } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { calculateOptionMetrics } from '@/lib/option-calculations'

export function OptionSimulator() {
  const [mounted, setMounted] = useState(false)
  const [currentPrice, setCurrentPrice] = useState(103000)
  const [strikePrice, setStrikePrice] = useState(103000)
  const [daysToExpiry, setDaysToExpiry] = useState(30)
  const [volatility, setVolatility] = useState(0.5)
  const [optionType, setOptionType] = useState<'call' | 'put'>('call')
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    try {
      setMounted(true)
    } catch (err) {
      console.error('Initialization error:', err)
    }
  }, [])

  const { data, metrics } = useMemo(() => {
    if (!mounted) return { data: [], metrics: null }

    try {
      const range = []
      const minPrice = currentPrice * 0.8
      const maxPrice = currentPrice * 1.2
      const step = (maxPrice - minPrice) / 50

      for (let price = minPrice; price <= maxPrice; price += step) {
        const metrics = calculateOptionMetrics({
          currentPrice: price,
          strikePrice,
          daysToExpiry,
          volatility,
          optionType,
        })
        range.push({
          price,
          pnl: metrics.pnl,
        })
      }

      const currentMetrics = calculateOptionMetrics({
        currentPrice,
        strikePrice,
        daysToExpiry,
        volatility,
        optionType,
      })

      setError(null)
      return { data: range, metrics: currentMetrics }
    } catch (err) {
      console.error('Calculation error:', err)
      setError('計算中にエラーが発生しました')
      return { data: [], metrics: null }
    }
  }, [mounted, currentPrice, strikePrice, daysToExpiry, volatility, optionType])

  if (!mounted) {
    return (
      <div className="animate-pulse bg-gray-800 rounded-lg p-4">
        <div className="h-8 bg-gray-700 rounded w-1/3 mb-4"></div>
        <div className="space-y-3">
          <div className="h-4 bg-gray-700 rounded"></div>
          <div className="h-4 bg-gray-700 rounded w-5/6"></div>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {error && (
        <div className="bg-red-500/10 border border-red-500 text-red-500 p-4 rounded-lg">
          {error}
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-gray-800 p-4 rounded-lg">
          <h2 className="text-xl font-semibold mb-4">パラメータ設定</h2>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-1">現在価格: ${currentPrice.toLocaleString()}</label>
              <input
                type="range"
                min={50000}
                max={150000}
                value={currentPrice}
                onChange={(e) => setCurrentPrice(Number(e.target.value))}
                className="w-full"
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-1">権利行使価格: ${strikePrice.toLocaleString()}</label>
              <input
                type="range"
                min={50000}
                max={150000}
                value={strikePrice}
                onChange={(e) => setStrikePrice(Number(e.target.value))}
                className="w-full"
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-1">残存日数: {daysToExpiry}日</label>
              <input
                type="range"
                min={1}
                max={365}
                value={daysToExpiry}
                onChange={(e) => setDaysToExpiry(Number(e.target.value))}
                className="w-full"
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-1">ボラティリティ: {(volatility * 100).toFixed(1)}%</label>
              <input
                type="range"
                min={0.1}
                max={2}
                step={0.1}
                value={volatility}
                onChange={(e) => setVolatility(Number(e.target.value))}
                className="w-full"
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-2">オプションタイプ</label>
              <div className="flex gap-4">
                <button
                  className={`px-4 py-2 rounded ${
                    optionType === 'call' ? 'bg-blue-600' : 'bg-gray-700'
                  }`}
                  onClick={() => setOptionType('call')}
                >
                  コール
                </button>
                <button
                  className={`px-4 py-2 rounded ${
                    optionType === 'put' ? 'bg-blue-600' : 'bg-gray-700'
                  }`}
                  onClick={() => setOptionType('put')}
                >
                  プット
                </button>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-gray-800 p-4 rounded-lg">
          <h2 className="text-xl font-semibold mb-4">ギリシャ指標</h2>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <p className="text-sm text-gray-400">Delta</p>
              <p className="text-lg font-semibold">{metrics?.delta.toFixed(4) ?? '-'}</p>
            </div>
            <div>
              <p className="text-sm text-gray-400">Gamma</p>
              <p className="text-lg font-semibold">{metrics?.gamma.toFixed(4) ?? '-'}</p>
            </div>
            <div>
              <p className="text-sm text-gray-400">Theta</p>
              <p className="text-lg font-semibold">{metrics?.theta.toFixed(4) ?? '-'}</p>
            </div>
            <div>
              <p className="text-sm text-gray-400">Vega</p>
              <p className="text-lg font-semibold">{metrics?.vega.toFixed(4) ?? '-'}</p>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-gray-800 p-4 rounded-lg">
        <h2 className="text-xl font-semibold mb-4">損益シミュレーション</h2>
        <div className="w-full h-[400px]">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="price"
                tickFormatter={(value) => `$${value.toLocaleString()}`}
              />
              <YAxis
                tickFormatter={(value) => `$${value.toLocaleString()}`}
              />
              <Tooltip
                formatter={(value: number) => [`$${value.toLocaleString()}`, 'PnL']}
                labelFormatter={(label: number) => `価格: $${label.toLocaleString()}`}
              />
              <Legend />
              <Line
                type="monotone"
                dataKey="pnl"
                stroke="#8884d8"
                name="損益"
                dot={false}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  )
}