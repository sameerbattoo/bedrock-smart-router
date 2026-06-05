import { useState } from 'react'
import { USE_CASES, TIME_FILTERS } from './shared'

export default function HistoryPage({ history, setHistory, onNavigate }) {
  const [useCaseFilter, setUseCaseFilter] = useState('all')
  const [timeFilter, setTimeFilter] = useState('all')
  const [sortBy, setSortBy] = useState('newest')

  const filtered = history.filter(h => {
    if (useCaseFilter !== 'all' && h.use_case !== useCaseFilter) return false
    if (timeFilter !== 'all') {
      const tf = TIME_FILTERS.find(t => t.id === timeFilter)
      if (tf && tf.ms && h.timestamp) {
        if (Date.now() - h.timestamp > tf.ms) return false
      }
    }
    return true
  }).sort((a, b) => {
    if (sortBy === 'newest') return (b.timestamp || 0) - (a.timestamp || 0)
    if (sortBy === 'oldest') return (a.timestamp || 0) - (b.timestamp || 0)
    if (sortBy === 'savings') return (b.savings_pct || 0) - (a.savings_pct || 0)
    if (sortBy === 'cost') return (a.router_cost || 0) - (b.router_cost || 0)
    return 0
  })

  function formatTime(ts) {
    if (!ts) return '—'
    const diff = Date.now() - ts
    if (diff < 60000) return 'just now'
    if (diff < 3600000) return `${Math.floor(diff/60000)}m ago`
    if (diff < 86400000) return `${Math.floor(diff/3600000)}h ago`
    return new Date(ts).toLocaleDateString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
  }

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      {/* Filters bar */}
      <div className="px-5 py-3 border-b border-gray-800/50 bg-gray-900/40">
        <div className="flex items-center gap-4 flex-wrap">
          {/* Use-case filter */}
          <div className="flex items-center gap-2">
            <span className="text-[10px] text-gray-500 uppercase font-bold">Use Case</span>
            <select value={useCaseFilter} onChange={e => setUseCaseFilter(e.target.value)}
              className="bg-gray-800 border border-gray-700 rounded-md px-2 py-1 text-[11px] text-gray-300 focus:outline-none focus:border-orange-600">
              <option value="all">All</option>
              {USE_CASES.map(uc => <option key={uc.id} value={uc.id}>{uc.label}</option>)}
            </select>
          </div>
          {/* Time filter */}
          <div className="flex items-center gap-2">
            <span className="text-[10px] text-gray-500 uppercase font-bold">Time</span>
            <div className="flex bg-gray-800 rounded-lg p-0.5 border border-gray-700">
              {TIME_FILTERS.map(tf => (
                <button key={tf.id} onClick={() => setTimeFilter(tf.id)}
                  className={`text-[10px] px-2 py-1 rounded-md font-medium transition-all ${timeFilter === tf.id ? 'bg-orange-600 text-white' : 'text-gray-500 hover:text-gray-300'}`}>
                  {tf.label}
                </button>
              ))}
            </div>
          </div>
          {/* Sort */}
          <div className="flex items-center gap-2">
            <span className="text-[10px] text-gray-500 uppercase font-bold">Sort</span>
            <select value={sortBy} onChange={e => setSortBy(e.target.value)}
              className="bg-gray-800 border border-gray-700 rounded-md px-2 py-1 text-[11px] text-gray-300 focus:outline-none focus:border-orange-600">
              <option value="newest">Newest First</option>
              <option value="oldest">Oldest First</option>
              <option value="savings">Highest Savings</option>
              <option value="cost">Lowest Cost</option>
            </select>
          </div>
          {/* Clear */}
          <div className="ml-auto flex items-center gap-2">
            <span className="text-[10px] text-gray-500">{filtered.length} of {history.length} entries</span>
            {history.length > 0 && <button onClick={() => { setHistory([]); localStorage.removeItem('bsr_demo_history') }}
              className="text-[10px] text-red-400 hover:text-red-300 px-2 py-1 rounded hover:bg-red-900/20 border border-red-900/30 transition-all">Clear All</button>}
          </div>
        </div>
      </div>

      {/* History list */}
      <div className="flex-1 overflow-y-auto p-5">
        {filtered.length === 0 ? (
          <div className="text-center text-gray-600 mt-20">
            <div className="text-3xl mb-3">📋</div>
            <div className="text-sm">No comparison entries{useCaseFilter !== 'all' || timeFilter !== 'all' ? ' matching filters' : ''}</div>
          </div>
        ) : (
          <div className="space-y-2">
            {filtered.map(h => {
              const uc = USE_CASES.find(u => u.id === h.use_case)
              return (
                <div key={h.id} onClick={() => h.use_case !== 'strands' && onNavigate(h)}
                  className={`bg-gray-900/50 border border-gray-800/50 rounded-lg px-4 py-3 transition-all group ${h.use_case !== 'strands' ? 'cursor-pointer hover:border-orange-500/40 hover:bg-gray-900/70' : ''}`}>
                  <div className="flex items-center gap-3">
                    <span className="text-sm">{h.has_error ? '❌' : '✅'}</span>
                    <span className="text-sm">{uc?.icon || '⚡'}</span>
                    <div className="flex-1 min-w-0">
                      <div className="text-xs text-gray-300 truncate">{h.prompt}</div>
                      <div className="flex items-center gap-3 mt-1 text-[10px] text-gray-500">
                        <span>{uc?.label || h.use_case}</span>
                        <span>•</span>
                        <span>{h.router_model}</span>
                        <span>•</span>
                        <span>{h.complexity}</span>
                        <span>•</span>
                        <span>{formatTime(h.timestamp)}</span>
                      </div>
                    </div>
                    <div className="flex items-center gap-4 text-[10px]">
                      <div className="text-center"><div className="text-gray-600">Latency</div><div className="text-gray-300 font-mono">{h.router_latency}ms</div></div>
                      <div className="text-center"><div className="text-gray-600">Cost</div><div className="text-gray-300 font-mono">${h.router_cost?.toFixed(5)}</div></div>
                      <div className="text-center"><div className="text-gray-600">Savings</div><div className={`font-mono font-medium ${h.savings_pct > 0 ? 'text-green-400' : 'text-red-400'}`}>{h.savings_pct > 0 ? '↓' : '↑'}{Math.abs(h.savings_pct)}%</div></div>
                      {h.router_score && <div className="text-center"><div className="text-gray-600">Score</div><div className="text-orange-300 font-mono">{h.router_score}/10</div></div>}
                    </div>
                    <button onClick={e => { e.stopPropagation(); setHistory(prev => prev.filter(x => x.id !== h.id)) }}
                      className="opacity-0 group-hover:opacity-100 text-gray-600 hover:text-red-400 p-1 rounded hover:bg-red-900/20 transition-all"
                      title="Delete">
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"/></svg>
                    </button>
                    <svg className={`w-4 h-4 text-gray-700 group-hover:text-orange-400 transition-colors ${h.use_case === 'strands' ? 'invisible' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7"/></svg>
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </div>
    </div>
  )
}
