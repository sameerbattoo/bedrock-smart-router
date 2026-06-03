import { useState, useRef, useEffect } from 'react'
import { STREAM_API } from './shared'

// ── User Card ──────────────────────────────────────────────────────
const TIER_STYLES = {
  free: 'bg-gray-700/40 text-gray-300 border-gray-600/40',
  pro: 'bg-purple-900/40 text-purple-300 border-purple-600/40',
  enterprise: 'bg-green-900/40 text-green-300 border-green-600/40',
}

const COLOR_MAP = {
  purple: { border: 'border-purple-500/50', bg: 'bg-purple-900/10', text: 'text-purple-400', dot: 'bg-purple-400' },
  blue: { border: 'border-blue-500/50', bg: 'bg-blue-900/10', text: 'text-blue-400', dot: 'bg-blue-400' },
  green: { border: 'border-green-500/50', bg: 'bg-green-900/10', text: 'text-green-400', dot: 'bg-green-400' },
  orange: { border: 'border-orange-500/50', bg: 'bg-orange-900/10', text: 'text-orange-400', dot: 'bg-orange-400' },
}

function UserCard({ user, selected, onToggle }) {
  const colors = COLOR_MAP[user.color] || COLOR_MAP.purple
  return (
    <button onClick={onToggle}
      className={`p-3 rounded-lg border text-left transition-all ${
        selected
          ? `${colors.border} ${colors.bg} ring-1 ring-${user.color}-500/30`
          : 'border-gray-800/50 bg-gray-900/30 hover:border-gray-700 hover:bg-gray-900/50'
      }`}>
      <div className="flex items-center gap-2 mb-1">
        <span className={`w-2 h-2 rounded-full ${colors.dot}`} />
        <span className={`text-xs font-medium ${selected ? colors.text : 'text-gray-300'}`}>{user.name}</span>
      </div>
      <div className="flex items-center gap-2 text-[10px]">
        <span className={`px-1.5 py-0.5 rounded border ${TIER_STYLES[user.tier]}`}>{user.tier}</span>
        <span className="text-gray-500">{user.team}</span>
        <span className={`ml-auto ${user.on_exceeded === 'reject' ? 'text-red-400' : 'text-yellow-400'}`}>
          ${user.budget < 0.01 ? user.budget.toFixed(4) : user.budget.toFixed(2)}/hr
          {user.on_exceeded === 'reject' ? ' · reject' : ' · downgrade'}
        </span>
      </div>
    </button>
  )
}

// ── Status Badge ───────────────────────────────────────────────────
function StatusBadge({ status }) {
  if (status === 'over') return <span className="text-[9px] px-1.5 py-0.5 rounded bg-red-900/40 text-red-300 border border-red-700/40">Over Budget</span>
  if (status === 'warning') return <span className="text-[9px] px-1.5 py-0.5 rounded bg-yellow-900/40 text-yellow-300 border border-yellow-700/40">&gt;80%</span>
  return <span className="text-[9px] px-1.5 py-0.5 rounded bg-green-900/40 text-green-300 border border-green-700/40">OK</span>
}

// ── Budget Bar ─────────────────────────────────────────────────────
function BudgetBar({ pct }) {
  const color = pct >= 100 ? 'bg-red-500' : pct >= 80 ? 'bg-yellow-500' : 'bg-green-500'
  return (
    <div className="w-full h-1.5 bg-gray-800 rounded-full overflow-hidden">
      <div className={`h-full ${color} transition-all duration-300`} style={{ width: `${Math.min(pct, 100)}%` }} />
    </div>
  )
}

// ── Main Component ─────────────────────────────────────────────────
export default function UsagePage({ onRun }) {
  const USERS = [
    { id: 'alice', name: 'Alice Chen', team: 'Engineering', tier: 'pro', budget: 0.015, on_exceeded: 'downgrade', color: 'purple' },
    { id: 'bob', name: 'Bob Martinez', team: 'Marketing', tier: 'free', budget: 0.005, on_exceeded: 'reject', color: 'blue' },
    { id: 'charlie', name: 'Charlie Kim', team: 'Data Science', tier: 'enterprise', budget: 0.05, on_exceeded: 'downgrade', color: 'green' },
    { id: 'diana', name: 'Diana Patel', team: 'Executive', tier: 'enterprise', budget: 0.05, on_exceeded: 'downgrade', color: 'orange' },
  ]

  const TENANTS = [
    { id: 'acme-corp', name: 'Acme Corp', team: '12 members', tier: 'acme-corp', budget: 0.03, on_exceeded: 'downgrade', color: 'purple' },
    { id: 'globex-inc', name: 'Globex Inc', team: '8 members', tier: 'globex-inc', budget: 0.008, on_exceeded: 'reject', color: 'blue' },
    { id: 'initech', name: 'Initech', team: '6 members', tier: 'initech', budget: 0.05, on_exceeded: 'downgrade', color: 'green' },
    { id: 'umbrella-co', name: 'Umbrella Co', team: '4 members', tier: 'umbrella-co', budget: 0.06, on_exceeded: 'downgrade', color: 'orange' },
  ]

  const [scope, setScope] = useState('user')
  const [selectedUsers, setSelectedUsers] = useState(['alice', 'bob', 'charlie'])
  const [strategy, setStrategy] = useState('quality-optimized')
  const [classifier, setClassifier] = useState('heuristic')
  const [complexity, setComplexity] = useState('mixed')
  const [requestsPerUser, setRequestsPerUser] = useState(20)
  const [speed, setSpeed] = useState('normal')
  const [running, setRunning] = useState(false)
  const [progress, setProgress] = useState({})
  const [dashboard, setDashboard] = useState({})
  const [events, setEvents] = useState([])
  const [totals, setTotals] = useState({ requests: 0, cost: 0, overBudget: 0 })
  const [requestLog, setRequestLog] = useState([])
  const logRef = useRef(null)
  const abortRef = useRef(null)

  // Load existing data on mount
  useEffect(() => {
    fetch(`${STREAM_API}/usage-dashboard?scope=${scope}`)
      .then(r => r.json())
      .then(data => {
        const entities = data.entities || []
        if (entities.length > 0) {
          const dashMap = {}
          let totalCost = 0
          for (const u of entities) {
            dashMap[u.user_id] = {
              requests: 0,
              cumulative_cost: u.total_cost || 0,
              remaining: u.remaining || 0,
              pct_used: u.pct_used || 0,
              model: '—',
              status: u.status || 'ok',
              downgraded: false,
            }
            totalCost += (u.total_cost || 0)
          }
          setDashboard(dashMap)
          setTotals({ requests: 0, cost: totalCost, overBudget: 0 })
        }
      })
      .catch(() => {})
  }, [scope])

  // Auto-scroll log to bottom
  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight
    }
  }, [requestLog])

  function toggleUser(uid) {
    setSelectedUsers(prev =>
      prev.includes(uid) ? prev.filter(u => u !== uid) : [...prev, uid]
    )
  }

  async function startSimulation() {
    if (running) return
    setRunning(true)
    setEvents([])
    setDashboard({})
    setProgress({})
    setTotals({ requests: 0, cost: 0, overBudget: 0 })
    setRequestLog([])
    if (onRun) onRun()

    const form = new FormData()
    form.append('selected_users', selectedUsers.join(','))
    form.append('scope', scope)
    form.append('strategy', strategy)
    form.append('classifier', classifier)
    form.append('complexity', complexity)
    form.append('requests_per_user', requestsPerUser)
    form.append('speed', speed)

    try {
      const controller = new AbortController()
      abortRef.current = controller
      const res = await fetch(`${STREAM_API}/usage-simulate`, { method: 'POST', body: form, signal: controller.signal })
      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let eventType = null

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (line.startsWith('event: ')) {
            eventType = line.slice(7).trim()
          } else if (line.startsWith('data: ') && eventType) {
            const data = JSON.parse(line.slice(6))
            switch (eventType) {
              case 'request_complete':
                // Update dashboard
                setDashboard(prev => ({
                  ...prev,
                  [data.user_id]: {
                    ...data,
                    requests: (prev[data.user_id]?.requests || 0) + 1,
                  }
                }))
                setProgress(prev => ({
                  ...prev,
                  [data.user_id]: { current: data.request_num, total: data.total_requests }
                }))
                setTotals(prev => ({
                  requests: prev.requests + 1,
                  cost: prev.cost + data.cost,
                  overBudget: prev.overBudget,
                }))
                // Add to running log
                setRequestLog(prev => [...prev, {
                  user: data.name,
                  user_id: data.user_id,
                  prompt: data.prompt,
                  response: data.response_text,
                  model: data.model,
                  cost: data.cost,
                  complexity: data.complexity,
                  latency: data.latency_ms,
                  downgraded: data.downgraded,
                  rejected: data.rejected || false,
                  status: data.status,
                }])
                break
              case 'budget_exceeded':
                setEvents(prev => [{
                  time: new Date().toLocaleTimeString(),
                  type: 'budget',
                  user: data.name,
                  message: `Budget exceeded → ${data.action === 'reject' ? 'requests rejected' : 'switched to cost-optimized'}`,
                }, ...prev].slice(0, 20))
                break
              case 'simulation_complete':
                setTotals(prev => ({ ...prev, ...data }))
                break
              case 'request_error':
                setEvents(prev => [{
                  time: new Date().toLocaleTimeString(),
                  type: 'error',
                  user: data.user_id,
                  message: data.error,
                }, ...prev].slice(0, 20))
                break
            }
            eventType = null
          }
        }
      }
    } catch (e) {
      if (e.name !== 'AbortError') {
        setEvents(prev => [{ time: new Date().toLocaleTimeString(), type: 'error', user: 'system', message: e.message }, ...prev])
      }
    } finally {
      setRunning(false)
    }
  }

  function stopSimulation() {
    if (abortRef.current) {
      abortRef.current.abort()
      abortRef.current = null
    }
    setRunning(false)
  }

  async function resetData() {
    await fetch(`${STREAM_API}/usage-reset`, { method: 'POST' })
    setDashboard({})
    setEvents([])
    setTotals({ requests: 0, cost: 0, overBudget: 0 })
    setProgress({})
    setRequestLog([])
  }

  const entities = scope === 'tenant' ? TENANTS : USERS
  const dashboardUsers = selectedUsers.map(uid => {
    const user = entities.find(u => u.id === uid)
    if (!user) return null
    const data = dashboard[uid]
    return {
      ...user,
      requests: data?.requests || 0,
      cost: data?.cumulative_cost || 0,
      remaining: data?.remaining ?? user.budget,
      pct: data?.pct_used || 0,
      model: data?.model || '—',
      status: data?.status || 'ok',
      downgraded: data?.downgraded || false,
      on_exceeded: user.on_exceeded,
    }
  }).filter(Boolean)

  const overBudgetCount = dashboardUsers.filter(u => u.status === 'over').length

  return (
    <div className="flex-1 flex overflow-hidden">
      {/* ═══ Left Panel: Configuration & Simulation ═══ */}
      <div className="flex-1 overflow-y-auto p-4 space-y-3">

        {/* Step 1: Configure Simulation */}
        <div className="border border-gray-800/60 rounded-xl overflow-hidden bg-gray-900/20">
          <div className="px-4 py-2.5 bg-gray-900/60 border-b border-gray-800/40">
            <div className="flex items-center gap-3">
              <span className="w-6 h-6 rounded-full bg-orange-600/20 border border-orange-500/40 flex items-center justify-center text-[11px] font-bold text-orange-400">1</span>
              <span className="text-xs font-medium text-gray-300">Select Users & Configure</span>
            </div>
          </div>
          <div className="p-4 space-y-4">
            {/* Scope toggle */}
            <div>
              <div className="text-[10px] text-gray-500 uppercase tracking-wider font-bold mb-2">Tracking Scope</div>
              <div className="flex gap-1">
                <button onClick={() => { setScope('user'); setSelectedUsers(['alice', 'bob', 'charlie']); setDashboard({}); setRequestLog([]); setEvents([]) }}
                  className={`text-[10px] px-3 py-1.5 rounded-md font-medium transition-all ${scope === 'user' ? 'bg-orange-600 text-white' : 'text-gray-500 hover:text-gray-300 border border-gray-800/50'}`}>
                  👤 Per User
                </button>
                <button onClick={() => { setScope('tenant'); setSelectedUsers(['acme-corp', 'globex-inc', 'initech']); setDashboard({}); setRequestLog([]); setEvents([]) }}
                  className={`text-[10px] px-3 py-1.5 rounded-md font-medium transition-all ${scope === 'tenant' ? 'bg-orange-600 text-white' : 'text-gray-500 hover:text-gray-300 border border-gray-800/50'}`}>
                  🏢 Per Tenant (Department)
                </button>
              </div>
            </div>

            {/* Entity selection */}
            <div>
              <div className="text-[10px] text-gray-500 uppercase tracking-wider font-bold mb-2">Select {scope === 'tenant' ? 'Departments' : 'Users'} (multi-select)</div>
              <div className="grid grid-cols-2 gap-2">
                {entities.map(u => (
                  <UserCard key={u.id} user={u} selected={selectedUsers.includes(u.id)} onToggle={() => toggleUser(u.id)} />
                ))}
              </div>
            </div>

            {/* Controls row */}
            <div className="flex items-center gap-3 flex-wrap">
              <div>
                <div className="text-[10px] text-gray-500 uppercase tracking-wider font-bold mb-1">Strategy</div>
                <div className="flex gap-1">
                  {['balanced', 'cost', 'latency', 'quality'].map(s => (
                    <button key={s} onClick={() => setStrategy(s === 'cost' ? 'cost-optimized' : s === 'quality' ? 'quality-optimized' : s === 'latency' ? 'latency-optimized' : 'balanced')}
                      className={`text-[10px] px-2 py-1 rounded-md font-medium transition-all capitalize ${(strategy === s || strategy === s + '-optimized') ? 'bg-orange-600 text-white' : 'text-gray-500 hover:text-gray-300 border border-gray-800/50'}`}>
                      {s}
                    </button>
                  ))}
                </div>
              </div>
              <div>
                <div className="text-[10px] text-gray-500 uppercase tracking-wider font-bold mb-1">Classifier</div>
                <div className="flex gap-1">
                  {['heuristic', 'ml'].map(c => (
                    <button key={c} onClick={() => setClassifier(c)}
                      className={`text-[10px] px-2 py-1 rounded-md font-medium transition-all capitalize ${classifier === c ? 'bg-purple-600 text-white' : 'text-gray-500 hover:text-gray-300 border border-gray-800/50'}`}>
                      {c}
                    </button>
                  ))}
                </div>
              </div>
              <div>
                <div className="text-[10px] text-gray-500 uppercase tracking-wider font-bold mb-1">Prompt Complexity</div>
                <div className="flex gap-1">
                  {['simple', 'moderate', 'complex', 'mixed'].map(c => (
                    <button key={c} onClick={() => setComplexity(c)}
                      className={`text-[10px] px-2 py-1 rounded-md font-medium transition-all capitalize ${complexity === c ? 'bg-blue-600 text-white' : 'text-gray-500 hover:text-gray-300 border border-gray-800/50'}`}>
                      {c}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* Speed & count & action buttons */}
            <div className="flex items-end gap-4">
              <div>
                <div className="text-[10px] text-gray-500 uppercase tracking-wider font-bold mb-1">Speed</div>
                <div className="flex gap-1">
                  {['slow', 'normal', 'fast'].map(s => (
                    <button key={s} onClick={() => setSpeed(s)}
                      className={`text-[10px] px-2 py-1 rounded-md font-medium transition-all capitalize ${speed === s ? 'bg-gray-600 text-white' : 'text-gray-500 hover:text-gray-300 border border-gray-800/50'}`}>
                      {s}
                    </button>
                  ))}
                </div>
              </div>
              <div>
                <div className="text-[10px] text-gray-500 uppercase tracking-wider font-bold mb-1">Requests/User</div>
                <div className="flex gap-1">
                  {[20, 30, 50, 100].map(n => (
                    <button key={n} onClick={() => setRequestsPerUser(n)}
                      className={`text-[10px] px-2 py-1 rounded-md font-medium transition-all ${requestsPerUser === n ? 'bg-gray-600 text-white' : 'text-gray-500 hover:text-gray-300 border border-gray-800/50'}`}>
                      {n}
                    </button>
                  ))}
                </div>
              </div>
              <div className="flex items-center gap-2 ml-auto">
                <button onClick={startSimulation} disabled={running || selectedUsers.length === 0}
                  className="px-5 py-2 bg-orange-600 hover:bg-orange-500 disabled:bg-gray-700 disabled:text-gray-500 text-white text-xs font-medium rounded-lg transition-all flex items-center gap-2">
                  {running ? <><svg className="w-3.5 h-3.5 animate-spin" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/></svg>Running...</> : '▶ Start Simulation'}
                </button>
                {running && (
                  <button onClick={stopSimulation} className="px-3 py-2 bg-red-600 hover:bg-red-500 text-white text-xs font-medium rounded-lg transition-all">
                    ⏹ Stop
                  </button>
                )}
                <button onClick={resetData} disabled={running}
                  className="px-3 py-2 text-gray-500 hover:text-gray-300 text-xs border border-gray-800/50 rounded-lg transition-all disabled:opacity-50">
                  Reset
                </button>
              </div>
            </div>

          </div>
        </div>

        {/* Running Log — Section 2: prompt & response per request */}
        {requestLog.length > 0 && (
          <div className="border border-gray-800/60 rounded-xl overflow-hidden bg-gray-900/20">
            <div className="px-4 py-2.5 bg-gray-900/60 border-b border-gray-800/40">
              <div className="flex items-center gap-3">
                <span className="w-6 h-6 rounded-full bg-orange-600/20 border border-orange-500/40 flex items-center justify-center text-[11px] font-bold text-orange-400">2</span>
                <span className="text-xs font-medium text-gray-300">Request Log</span>
                <span className="text-[9px] text-gray-500 ml-auto">{requestLog.length} requests</span>
              </div>
            </div>
            <div ref={logRef} className={`${events.length > 0 ? 'max-h-[200px]' : 'max-h-[400px]'} overflow-y-auto divide-y divide-gray-800/20`}>
              {requestLog.map((entry, i) => {
                const colors = COLOR_MAP[entities.find(u => u.id === entry.user_id)?.color] || COLOR_MAP.purple
                const isOver = entry.downgraded || entry.status === 'over'
                return (
                  <div key={i} className={`px-3 py-2 text-[10px] ${isOver ? 'bg-red-900/10 border-l-2 border-red-500' : 'hover:bg-gray-800/20'}`}>
                    <div className="flex items-center gap-2 mb-0.5">
                      <span className={`w-1.5 h-1.5 rounded-full ${colors.dot}`} />
                      <span className={`font-medium ${isOver ? 'text-red-300' : colors.text}`}>{entry.user}</span>
                      <span className="text-gray-600">→</span>
                      <span className={`${isOver ? 'text-red-400' : 'text-gray-400'}`}>{entry.model}</span>
                      <span className={`px-1 py-0.5 rounded text-[8px] ${entry.complexity === 'simple' ? 'bg-green-900/30 text-green-400' : entry.complexity === 'moderate' ? 'bg-yellow-900/30 text-yellow-400' : entry.complexity === 'complex' ? 'bg-orange-900/30 text-orange-400' : 'bg-red-900/30 text-red-400'}`}>{entry.complexity}</span>
                      <span className="text-gray-600 text-[9px]">{entry.latency}ms</span>
                      <span className="text-gray-600 ml-auto">${entry.cost.toFixed(6)}</span>
                      {isOver && <span className="text-red-400 font-bold text-[9px]">OVER BUDGET</span>}
                    </div>
                    <div className="text-gray-500 truncate pl-3">
                      <span className="text-gray-600">Q:</span> {entry.prompt}
                    </div>
                    <details className="pl-3">
                      <summary className={`truncate cursor-pointer hover:text-gray-300 ${isOver ? 'text-red-300/70' : 'text-gray-400'}`}>
                        <span className="text-gray-600">A:</span> {(entry.response || '...').slice(0, 80)}{(entry.response || '').length > 80 ? '...' : ''}
                      </summary>
                      <div className={`text-[10px] mt-1 whitespace-pre-wrap pl-3 ${isOver ? 'text-red-300/70' : 'text-gray-400'}`}>{entry.response || '...'}</div>
                    </details>
                  </div>
                )
              })}
            </div>
          </div>
        )}

        {/* Budget Enforcement Events — Section 3 */}
        {events.length > 0 && (
          <div className="border border-gray-800/60 rounded-xl overflow-hidden bg-gray-900/20">
            <div className="px-4 py-2.5 bg-gray-900/60 border-b border-gray-800/40">
              <div className="flex items-center gap-3">
                <span className="w-6 h-6 rounded-full bg-red-600/20 border border-red-500/40 flex items-center justify-center text-[11px] font-bold text-red-400">3</span>
                <span className="text-xs font-medium text-gray-300">Budget Enforcement Events</span>
                <span className="text-[9px] text-gray-500 ml-auto">{events.length} events</span>
              </div>
            </div>
            <div className="p-3 max-h-40 overflow-y-auto space-y-1">
              {events.map((e, i) => (
                <div key={i} className={`text-[10px] flex items-center gap-2 px-2 py-1 rounded ${e.type === 'budget' ? 'bg-red-900/10 text-red-300' : 'bg-gray-800/30 text-gray-400'}`}>
                  <span className="text-gray-600 font-mono">{e.time}</span>
                  <span className="font-medium">{e.user}:</span>
                  <span>{e.message}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* ═══ Right Panel: Live Dashboard (always visible) ═══ */}
      <div className="w-[420px] border-l border-gray-800/50 bg-[#0b1018] overflow-y-auto flex-shrink-0">
        <div className="p-4 space-y-4">
          {/* Header */}
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium text-gray-300">📊 Live Dashboard</span>
            {running && <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />}
          </div>

          {/* Top metrics */}
          <div className="grid grid-cols-3 gap-2">
            <div className="bg-gray-900/40 border border-gray-800/40 rounded-lg p-2 text-center">
              <div className="text-[9px] text-gray-500 uppercase">Requests</div>
              <div className="text-lg font-bold text-gray-200">{totals.requests}</div>
            </div>
            <div className="bg-gray-900/40 border border-gray-800/40 rounded-lg p-2 text-center">
              <div className="text-[9px] text-gray-500 uppercase">Total Cost</div>
              <div className="text-lg font-bold text-gray-200">${totals.cost.toFixed(6)}</div>
            </div>
            <div className="bg-gray-900/40 border border-gray-800/40 rounded-lg p-2 text-center">
              <div className="text-[9px] text-gray-500 uppercase">Over Budget</div>
              <div className={`text-lg font-bold ${overBudgetCount > 0 ? 'text-red-400' : 'text-green-400'}`}>{overBudgetCount}</div>
            </div>
          </div>

          {/* Per-user table */}
          <div className="border border-gray-800/40 rounded-lg overflow-hidden">
            <div className="px-3 py-2 bg-gray-900/60 border-b border-gray-800/40">
              <span className="text-[10px] text-gray-400 font-bold uppercase">Per-{scope === 'tenant' ? 'Tenant' : 'User'} Tracking (rolling 1-hour window)</span>
            </div>
            <div className="divide-y divide-gray-800/30">
              {dashboardUsers.map(u => {
                const colors = COLOR_MAP[u.color] || COLOR_MAP.purple
                return (
                  <div key={u.id} className="px-3 py-2.5 hover:bg-gray-800/20 transition-colors">
                    <div className="flex items-center gap-2 mb-1.5">
                      <span className={`w-2 h-2 rounded-full ${colors.dot}`} />
                      <span className={`text-[11px] font-medium ${colors.text}`}>{u.name}</span>
                      <span className="text-[9px] text-gray-600">{u.team}</span>
                      <StatusBadge status={u.status} />
                    </div>
                    <div className="grid grid-cols-4 gap-2 text-[10px] mb-1.5">
                      <div><span className="text-gray-600">Reqs:</span> <span className="text-gray-300">{u.requests}</span></div>
                      <div><span className="text-gray-600">Cost:</span> <span className="text-gray-300">${u.cost.toFixed(6)}</span></div>
                      <div><span className="text-gray-600">Left:</span> <span className={u.remaining <= 0 ? 'text-red-400' : 'text-gray-300'}>${u.remaining.toFixed(6)}</span></div>
                      <div><span className="text-gray-600">Model:</span> <span className={`${u.downgraded ? 'text-red-400 font-bold' : u.status === 'over' && u.on_exceeded === 'reject' ? 'text-red-400 font-bold' : 'text-gray-300'}`}>{u.status === 'over' && u.on_exceeded === 'reject' ? '🚫 Blocked' : u.downgraded ? '⬇ ' + u.model.split(' ')[0] : u.model.split(' ')[0]}</span></div>
                    </div>
                    <BudgetBar pct={u.pct} />
                    {(u.downgraded || (u.status === 'over' && u.on_exceeded === 'reject')) && (
                      <div className={`text-[9px] mt-1 flex items-center gap-1 px-2 py-1 rounded border ${
                        u.on_exceeded === 'reject'
                          ? 'text-red-400 bg-red-900/10 border-red-800/30'
                          : 'text-yellow-400 bg-yellow-900/10 border-yellow-800/30'
                      }`}>
                        {u.on_exceeded === 'reject'
                          ? '🚫 Budget exceeded → requests rejected (no cost incurred)'
                          : '⚠️ Budget exceeded → switched to cost-optimized strategy'}
                      </div>
                    )}
                  </div>
                )
              })}
              {dashboardUsers.length === 0 && (
                <div className="px-3 py-6 text-center text-[11px] text-gray-600">
                  Select users and start simulation to see tracking data
                </div>
              )}
            </div>
          </div>

          {/* Model distribution */}
          {totals.requests > 0 && (
            <div className="border border-gray-800/40 rounded-lg overflow-hidden">
              <div className="px-3 py-2 bg-gray-900/60 border-b border-gray-800/40">
                <span className="text-[10px] text-gray-400 font-bold uppercase">How It Works</span>
              </div>
              <div className="p-3 text-[10px] text-gray-400 space-y-2">
                <div className="flex items-start gap-2">
                  <span className="text-green-400">1.</span>
                  <span>Each request is tagged with <code className="text-orange-300">{scope === 'tenant' ? 'team' : 'user_id'}</code> and <code className="text-orange-300">tier</code> via routing metadata</span>
                </div>
                <div className="flex items-start gap-2">
                  <span className="text-green-400">2.</span>
                  <span>Cost is tracked per-{scope} via <code className="text-orange-300">BudgetTracker</code> + <code className="text-orange-300">SQLiteBudgetStore</code> (in-memory hot path, async persistence)</span>
                </div>
                <div className="flex items-start gap-2">
                  <span className="text-green-400">3.</span>
                  <span><code className="text-orange-300">BudgetRule</code> per tier defines <code className="text-orange-300">max_hourly_spend</code> (rolling 1-hour window) and <code className="text-orange-300">on_exceeded</code> action</span>
                </div>
                <div className="flex items-start gap-2">
                  <span className="text-yellow-400">4.</span>
                  <span><strong className="text-yellow-300">on_exceeded: "downgrade"</strong> — switches to <code className="text-orange-300">cost-optimized</code> strategy. Classifier still picks the right tier, just the cheapest model in it</span>
                </div>
                <div className="flex items-start gap-2">
                  <span className="text-red-400">5.</span>
                  <span><strong className="text-red-300">on_exceeded: "reject"</strong> — request is blocked entirely. No model called, $0 cost. Raises <code className="text-orange-300">BudgetExceededError</code></span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
