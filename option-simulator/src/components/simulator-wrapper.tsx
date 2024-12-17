'use client'

import dynamic from 'next/dynamic'
import { Suspense } from 'react'

const DynamicOptionSimulator = dynamic(
  () => import('./option-simulator').then((mod) => mod.OptionSimulator),
  {
    ssr: false,
    loading: () => (
      <div className="animate-pulse bg-gray-800 rounded-lg p-4">
        <div className="h-8 bg-gray-700 rounded w-1/3 mb-4"></div>
        <div className="space-y-3">
          <div className="h-4 bg-gray-700 rounded"></div>
          <div className="h-4 bg-gray-700 rounded w-5/6"></div>
        </div>
      </div>
    ),
  }
)

export function SimulatorWrapper() {
  return (
    <Suspense fallback={null}>
      <DynamicOptionSimulator />
    </Suspense>
  )
}