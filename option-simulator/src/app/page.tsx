import { SimulatorWrapper } from '@/components/simulator-wrapper'

export default function Home() {
  return (
    <main className="min-h-screen bg-gray-900 text-white p-4">
      <h1 className="text-2xl font-bold mb-4">BTC オプション取引シミュレーター</h1>
      <SimulatorWrapper />
    </main>
  )
}
