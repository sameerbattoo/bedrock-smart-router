import { useState, useEffect } from 'react'
import { USE_CASES } from './components/shared'
import ComparePage from './components/ComparePage'
import HistoryPage from './components/HistoryPage'
import ThrottlePage from './components/ThrottlePage'
import StrandsPage from './components/StrandsPage'

export default function App() {
  const [activePage, setActivePage] = useState('compare')
  const [navOpen, setNavOpen] = useState(true)
  const [history, setHistory] = useState(() => {
    try { return JSON.parse(localStorage.getItem('bsr_demo_history') || '[]') } catch { return [] }
  })
  const [restoreState, setRestoreState] = useState(null)
  const [resetKey, setResetKey] = useState(0)

  useEffect(() => {
    try { localStorage.setItem('bsr_demo_history', JSON.stringify(history)) } catch {}
  }, [history])

  const totalRuns = history.length

  function handleHistoryNavigate(h) {
    setActivePage(h.use_case || 'compare')
    setRestoreState({ ...h, _ts: Date.now() }) // _ts forces useEffect re-trigger
  }

  return (
    <div className="h-screen bg-[#0a0e17] text-gray-100 flex flex-col overflow-hidden">
      {/* Header */}
      <header className="border-b border-gray-800/50 px-4 py-3 flex items-center justify-between bg-[#0d1220]/80 backdrop-blur-md sticky top-0 z-50">
        <div className="flex items-center gap-2.5">
          <button onClick={() => setNavOpen(!navOpen)} className="text-gray-400 hover:text-white p-1 rounded hover:bg-gray-800 transition-all mr-1">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16"/></svg>
          </button>
          <div className="w-7 h-7 rounded-md bg-gradient-to-br from-orange-500 to-amber-600 flex items-center justify-center text-sm cursor-pointer" onClick={() => { setActivePage('compare'); setRestoreState(null); setResetKey(k => k + 1) }}>⚡</div>
          <span className="font-bold text-base cursor-pointer hover:text-orange-300 transition-colors" onClick={() => { setActivePage('compare'); setRestoreState(null); setResetKey(k => k + 1) }}>Bedrock Smart Router</span>
          <span className="text-[10px] text-gray-500 bg-gray-800 px-2 py-0.5 rounded-full">Demo</span>
        </div>
        <div className="flex items-center gap-4 text-xs text-gray-500">
          <span className="text-xs text-gray-400">Judge: Opus 4.7</span>
          <span>{totalRuns} runs</span>
          <span className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" />
        </div>
      </header>

      <div className="flex-1 flex overflow-hidden">
        {/* Left Navigation Panel */}
        <div className={`${navOpen ? 'w-56' : 'w-0'} transition-all duration-200 ease-in-out overflow-hidden border-r border-gray-800/50 bg-[#0b1018] flex-shrink-0`}>
          <div className="w-56 flex flex-col h-full">
            <div className="px-3 pt-4 pb-2">
              <div className="text-[10px] text-gray-500 uppercase tracking-wider font-bold px-2 mb-2">Use Cases</div>
            </div>
            <div className="flex-1 px-2 space-y-0.5">
              {USE_CASES.map(uc => (
                <button key={uc.id} onClick={() => !uc.coming && setActivePage(uc.id)}
                  disabled={uc.coming}
                  className={`w-full text-left px-3 py-2.5 rounded-lg text-[12px] transition-all flex items-center gap-2.5 ${
                    activePage === uc.id
                      ? 'bg-orange-600/15 text-orange-300 border border-orange-500/30'
                      : uc.coming
                        ? 'text-gray-600 cursor-not-allowed'
                        : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800/50'
                  }`}>
                  <span className="text-base">{uc.icon}</span>
                  <div className="flex-1 min-w-0">
                    <div className="truncate font-medium">{uc.label}</div>
                    {uc.coming && <span className="text-[9px] text-gray-600">Coming soon</span>}
                  </div>
                </button>
              ))}
            </div>
            <div className="px-2 pb-4 pt-2 border-t border-gray-800/50 mt-2">
              <button onClick={() => setActivePage('history')}
                className={`w-full text-left px-3 py-2.5 rounded-lg text-[12px] transition-all flex items-center gap-2.5 ${
                  activePage === 'history'
                    ? 'bg-orange-600/15 text-orange-300 border border-orange-500/30'
                    : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800/50'
                }`}>
                <span className="text-base">📋</span>
                <div className="flex-1 min-w-0">
                  <div className="truncate font-medium">History</div>
                  <span className="text-[9px] text-gray-500">{history.length} entries</span>
                </div>
              </button>
            </div>
          </div>
        </div>

        {/* Page Content */}
        {activePage === 'history' ? (
          <HistoryPage history={history} setHistory={setHistory} onNavigate={handleHistoryNavigate} />
        ) : activePage === 'compare' ? (
          <ComparePage key={resetKey} history={history} setHistory={setHistory} restoreState={restoreState} onRun={() => setNavOpen(false)} />
        ) : activePage === 'throttling' ? (
          <ThrottlePage history={history} setHistory={setHistory} onRun={() => setNavOpen(false)} restoreState={restoreState} />
        ) : activePage === 'strands' ? (
          <StrandsPage history={history} setHistory={setHistory} onRun={() => setNavOpen(false)} restoreState={restoreState} />
        ) : (
          <div className="flex-1 flex items-center justify-center">
            <div className="text-center text-gray-600">
              <div className="text-4xl mb-3">{USE_CASES.find(u => u.id === activePage)?.icon || '🚧'}</div>
              <div className="text-lg font-medium text-gray-400 mb-1">{USE_CASES.find(u => u.id === activePage)?.label}</div>
              <div className="text-sm text-gray-600">{USE_CASES.find(u => u.id === activePage)?.description}</div>
              <div className="text-xs text-gray-700 mt-3">Coming soon</div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
