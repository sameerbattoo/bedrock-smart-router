import { useState, useRef, useEffect } from 'react'
import { STREAM_API } from './shared'

const COLOR_MAP = {
  control: { text: 'text-blue-400', bg: 'bg-blue-900/20', border: 'border-blue-500/40', dot: 'bg-blue-400' },
  treatment: { text: 'text-orange-400', bg: 'bg-orange-900/20', border: 'border-orange-500/40', dot: 'bg-orange-400' },
  baseline: { text: 'text-blue-400', bg: 'bg-blue-900/20', border: 'border-blue-500/40', dot: 'bg-blue-400' },
  canary: { text: 'text-green-400', bg: 'bg-green-900/20', border: 'border-green-500/40', dot: 'bg-green-400' },
  primary: { text: 'text-purple-400', bg: 'bg-purple-900/20', border: 'border-purple-500/40', dot: 'bg-purple-400' },
  unknown: { text: 'text-gray-400', bg: 'bg-gray-900/20', border: 'border-gray-500/40', dot: 'bg-gray-400' },
}

export default function RolloutPage({ onRun }) {
  const [mode, setMode] = useState('ab_test')
  const [configs, setConfigs] = useState({})
  const [numUsers, setNumUsers] = useState(8)
  const [requestsPerUser, setRequestsPerUser] = useState(20)
  const [complexity, setComplexity] = useState('mixed')
  const [speed, setSpeed] = useState('fast')
  const [running, setRunning] = useState(false)
  const [requestLog, setRequestLog] = useState([])
  const [stats, setStats] = useState({})
  const [events, setEvents] = useState([])
  const logRef = useRef(null)
  const abortRef = useRef(null)

  useEffect(() => {
    fetch(`${STREAM_API}/rollout-configs`).then(r => r.json()).then(setConfigs).catch(() => {})
  }, [])

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight
  }, [requestLog])

  async function startSimulation() {
    if (running) return
    setRunning(true)
    setRequestLog([])
    setStats({})
    setEvents([])
    if (onRun) onRun()

    const form = new FormData()
    form.append('mode', mode)
    form.append('num_users', numUsers)
    form.append('requests_per_user', requestsPerUser)
    form.append('complexity', complexity)
    form.append('speed', speed)

    // Reset the router stats before starting
    const resetForm = new FormData()
    resetForm.append('mode', mode)
    await fetch(`${STREAM_API}/rollout-reset`, { method: 'POST', body: resetForm })

    try {
      const controller = new AbortController()
      abortRef.current = controller
      const res = await fetch(`${STREAM_API}/rollout-simulate`, { method: 'POST', body: form, signal: controller.signal })
      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = '', eventType = null

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''
        for (const line of lines) {
          if (line.startsWith('event: ')) eventType = line.slice(7).trim()
          else if (line.startsWith('data: ') && eventType) {
            const data = JSON.parse(line.slice(6))
            if (eventType === 'request_complete') {
              setRequestLog(prev => [...prev, data])
            } else if (eventType === 'stats_update') {
              setStats(data)
            } else if (eventType === 'request_error') {
              setEvents(prev => [{ time: new Date().toLocaleTimeString(), type: 'error', message: `${data.user_id}: ${data.error}` }, ...prev].slice(0, 20))
            }
            eventType = null
          }
        }
      }
    } catch (e) {
      if (e.name !== 'AbortError') setEvents(prev => [{ time: new Date().toLocaleTimeString(), type: 'error', message: e.message }, ...prev])
    } finally { setRunning(false) }
  }

  function stopSimulation() {
    if (abortRef.current) { abortRef.current.abort(); abortRef.current = null }
    setRunning(false)
  }

  async function resetData() {
    const form = new FormData(); form.append('mode', mode)
    await fetch(`${STREAM_API}/rollout-reset`, { method: 'POST', body: form })
    setRequestLog([]); setStats({}); setEvents([])
  }

  const currentConfig = configs[mode] || {}

  function syntaxHighlightJson(text) {
    if (!text) return ''
    // Process line by line, character-aware tokenization
    return text.split('\n').map(line => {
      let result = ''
      let i = 0
      while (i < line.length) {
        // Comment
        if (line[i] === '/' && line[i+1] === '/') {
          const comment = line.slice(i).replace(/</g, '&lt;').replace(/>/g, '&gt;')
          result += '<span style="color:#6b7280;font-style:italic">' + comment + '</span>'
          break
        }
        // String
        if (line[i] === '"') {
          let j = i + 1
          while (j < line.length && line[j] !== '"') { if (line[j] === '\\') j++; j++ }
          const str = line.slice(i, j + 1).replace(/</g, '&lt;').replace(/>/g, '&gt;')
          // Check if it's a key (followed by :)
          let k = j + 1
          while (k < line.length && line[k] === ' ') k++
          if (line[k] === ':') {
            result += '<span style="color:#93c5fd">' + str + '</span>'
          } else {
            result += '<span style="color:#86efac">' + str + '</span>'
          }
          i = j + 1
          continue
        }
        // Boolean/null
        if (line.slice(i).match(/^(true|false|null)/)) {
          const m = line.slice(i).match(/^(true|false|null)/)[0]
          result += '<span style="color:#c084fc">' + m + '</span>'
          i += m.length
          continue
        }
        // Number (after : or at start of value)
        if (/\d/.test(line[i]) && (i === 0 || /[\s:,[]/.test(line[i-1]))) {
          let j = i
          while (j < line.length && /[\d.]/.test(line[j])) j++
          result += '<span style="color:#fdba74">' + line.slice(i, j) + '</span>'
          i = j
          continue
        }
        // Regular character
        const ch = line[i] === '<' ? '&lt;' : line[i] === '>' ? '&gt;' : line[i] === '&' ? '&amp;' : line[i]
        result += ch
        i++
      }
      return result
    }).join('\n')
  }

  return (
    <div className="flex-1 flex overflow-hidden">
      {/* ═══ Left Panel ═══ */}
      <div className="flex-1 overflow-y-auto p-4 space-y-3">

        {/* Step 1: Select Mode & Configure */}
        <div className="border border-gray-800/60 rounded-xl overflow-hidden bg-gray-900/20">
          <div className="px-4 py-2.5 bg-gray-900/60 border-b border-gray-800/40">
            <div className="flex items-center gap-3">
              <span className="w-6 h-6 rounded-full bg-orange-600/20 border border-orange-500/40 flex items-center justify-center text-[11px] font-bold text-orange-400">1</span>
              <span className="text-xs font-medium text-gray-300">Select Rollout Mode & Configure</span>
            </div>
          </div>
          <div className="p-4 space-y-4">
            {/* Mode toggle */}
            <div>
              <div className="text-[10px] text-gray-500 uppercase tracking-wider font-bold mb-2">Rollout Mode</div>
              <div className="flex gap-1">
                {[{id: 'ab_test', label: '🔀 A/B Test'}, {id: 'canary', label: '🐤 Canary'}, {id: 'shadow', label: '👻 Shadow'}].map(m => (
                  <button key={m.id} onClick={() => { setMode(m.id); setRequestLog([]); setStats({}); setEvents([]) }}
                    className={`text-[11px] px-3 py-1.5 rounded-md font-medium transition-all ${mode === m.id ? 'bg-orange-600 text-white' : 'text-gray-500 hover:text-gray-300 border border-gray-800/50'}`}>
                    {m.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Config file (collapsible, syntax highlighted) */}
            <details open className="border border-gray-800/40 rounded-lg overflow-hidden">
              <summary className="px-3 py-2 text-[10px] text-gray-500 hover:text-gray-300 bg-gray-900/40 cursor-pointer select-none flex items-center gap-1.5">
                <span>📄</span> Config File — <span className="text-orange-300">{mode}.jsonc</span>
              </summary>
              <pre className="text-[10px] bg-[#0d1117] p-3 overflow-x-auto font-mono leading-relaxed max-h-[220px] overflow-y-auto">
                <code dangerouslySetInnerHTML={{ __html:
                  '<span style="color:#c084fc">from</span> <span style="color:#93c5fd">bedrock_smart_router</span> <span style="color:#c084fc">import</span> <span style="color:#93c5fd">BedrockRouter</span>\n' +
                  '<span style="color:#c084fc">import</span> <span style="color:#93c5fd">json</span>\n\n' +
                  '<span style="color:#6b7280;font-style:italic">// Load config and create router</span>\n' +
                  '<span style="color:#9ca3af">config</span> = json.load(open(<span style="color:#86efac">"' + mode + '.jsonc"</span>))\n' +
                  '<span style="color:#9ca3af">router</span> = BedrockRouter.create(config)\n\n' +
                  '<span style="color:#6b7280;font-style:italic">// Config contents:</span>\n' +
                  syntaxHighlightJson(typeof currentConfig === 'string' ? currentConfig : JSON.stringify(currentConfig, null, 2))
                }} />
              </pre>
            </details>

            {/* Controls row */}
            <div className="flex items-center gap-3 flex-wrap">
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
              <div>
                <div className="text-[10px] text-gray-500 uppercase tracking-wider font-bold mb-1">Users</div>
                <div className="flex gap-1">
                  {[4, 8, 12].map(n => (
                    <button key={n} onClick={() => setNumUsers(n)}
                      className={`text-[10px] px-2 py-1 rounded-md font-medium transition-all ${numUsers === n ? 'bg-gray-600 text-white' : 'text-gray-500 hover:text-gray-300 border border-gray-800/50'}`}>
                      {n}
                    </button>
                  ))}
                </div>
              </div>
              <div>
                <div className="text-[10px] text-gray-500 uppercase tracking-wider font-bold mb-1">Reqs/User</div>
                <div className="flex gap-1">
                  {[20, 30, 50, 100].map(n => (
                    <button key={n} onClick={() => setRequestsPerUser(n)}
                      className={`text-[10px] px-2 py-1 rounded-md font-medium transition-all ${requestsPerUser === n ? 'bg-gray-600 text-white' : 'text-gray-500 hover:text-gray-300 border border-gray-800/50'}`}>
                      {n}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* Speed + buttons (same row as Budget Tracking) */}
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
              <div className="flex items-center gap-2 ml-auto">
                <button onClick={startSimulation} disabled={running}
                  className="px-5 py-2 bg-orange-600 hover:bg-orange-500 disabled:bg-gray-700 disabled:text-gray-500 text-white text-xs font-medium rounded-lg transition-all flex items-center gap-2">
                  {running ? <><svg className="w-3.5 h-3.5 animate-spin" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/></svg>Running...</> : '▶ Start Simulation'}
                </button>
                {running && <button onClick={stopSimulation} className="px-3 py-2 bg-red-600 hover:bg-red-500 text-white text-xs font-medium rounded-lg transition-all">⏹ Stop</button>}
                <button onClick={resetData} disabled={running} className="px-3 py-2 text-gray-500 hover:text-gray-300 text-xs border border-gray-800/50 rounded-lg transition-all disabled:opacity-50">Reset</button>
              </div>
            </div>
          </div>
        </div>

        {/* Section 2: Request Log (+ Shadow Log side by side in shadow mode) */}
        {requestLog.length > 0 && (
          <div className="border border-gray-800/60 rounded-xl overflow-hidden bg-gray-900/20">
            <div className="px-4 py-2.5 bg-gray-900/60 border-b border-gray-800/40">
              <div className="flex items-center gap-3">
                <span className="w-6 h-6 rounded-full bg-orange-600/20 border border-orange-500/40 flex items-center justify-center text-[11px] font-bold text-orange-400">2</span>
                <span className="text-xs font-medium text-gray-300">Request Log</span>
                <span className="text-[9px] text-gray-500 ml-auto">{requestLog.length} requests</span>
              </div>
            </div>
            <div className={`flex ${mode === 'shadow' && stats.shadow_log && stats.shadow_log.length > 0 ? '' : ''}`}>
              {/* Primary log */}
              <div ref={logRef} className={`${events.length > 0 ? 'max-h-[200px]' : 'max-h-[350px]'} overflow-y-auto divide-y divide-gray-800/20 ${mode === 'shadow' && stats.shadow_log && stats.shadow_log.length > 0 ? 'flex-1 border-r border-gray-800/30' : 'w-full'}`}>
              {requestLog.map((entry, i) => {
                const colors = COLOR_MAP[entry.variant] || COLOR_MAP.unknown
                return (
                  <div key={i} className={`px-3 py-2 text-[10px] hover:bg-gray-800/20`}>
                    <div className="flex items-center gap-2 mb-0.5">
                      <span className={`w-1.5 h-1.5 rounded-full ${colors.dot}`} />
                      <span className="text-gray-500 font-mono">{entry.user_id}</span>
                      <span className="text-gray-600">→</span>
                      <span className={`px-1.5 py-0.5 rounded text-[9px] font-medium ${colors.bg} ${colors.text} border ${colors.border}`}>{entry.variant}</span>
                      <span className="text-gray-400">{entry.model}</span>
                      <span className={`px-1 py-0.5 rounded text-[8px] ${entry.complexity === 'simple' ? 'bg-green-900/30 text-green-400' : entry.complexity === 'moderate' ? 'bg-yellow-900/30 text-yellow-400' : entry.complexity === 'complex' ? 'bg-orange-900/30 text-orange-400' : 'bg-red-900/30 text-red-400'}`}>{entry.complexity}</span>
                      <span className="text-gray-600 text-[9px]">{entry.latency_ms}ms</span>
                      <span className="text-gray-600 ml-auto">${entry.cost.toFixed(6)}</span>
                    </div>
                    <div className="text-gray-500 truncate pl-3">
                      <span className="text-gray-600">Q:</span> {entry.prompt}
                    </div>
                    <details className="pl-3">
                      <summary className="text-gray-400 truncate cursor-pointer hover:text-gray-300">
                        <span className="text-gray-600">A:</span> {(entry.response_text || '...').slice(0, 80)}{(entry.response_text || '').length > 80 ? '...' : ''}
                      </summary>
                      <div className="text-gray-400 text-[10px] mt-1 whitespace-pre-wrap pl-3">{entry.response_text || '...'}</div>
                    </details>
                  </div>
                )
              })}
              </div>

              {/* Shadow log (side by side, shadow mode only) */}
              {mode === 'shadow' && stats.shadow_log && stats.shadow_log.length > 0 && (
                <div className={`w-[380px] ${events.length > 0 ? 'max-h-[200px]' : 'max-h-[350px]'} overflow-y-auto bg-purple-950/5 border-l border-gray-800/30`}>
                  <div className="px-2 py-1.5 bg-purple-900/10 border-b border-gray-800/30 sticky top-0 z-10">
                    <span className="text-[9px] text-purple-400 font-bold uppercase">👻 Shadow (async background)</span>
                    <span className="text-[8px] text-gray-600 ml-1">({stats.shadow_log.length} mirrored)</span>
                  </div>
                  {stats.shadow_log.map((entry, i) => (
                    <div key={i} className="px-2 py-1.5 text-[9px] border-b border-gray-800/10">
                      <div className="flex items-center gap-1.5 mb-0.5">
                        <span className={`${entry.success ? 'text-green-400' : 'text-red-400'}`}>{entry.success ? '✓' : '✗'}</span>
                        <span className="text-purple-300">Nova Pro</span>
                        <span className="text-gray-600">{entry.latency_ms}ms</span>
                        <span className="text-gray-600 ml-auto">${entry.cost.toFixed(6)}</span>
                      </div>
                      {entry.prompt && <div className="text-gray-500 truncate pl-3"><span className="text-gray-600">Q:</span> {entry.prompt}</div>}
                      {entry.response_text && (
                        <details className="pl-3">
                          <summary className="text-purple-300/70 truncate cursor-pointer hover:text-purple-200">
                            <span className="text-gray-600">A:</span> {entry.response_text.slice(0, 60)}...
                          </summary>
                          <div className="text-purple-300/70 text-[9px] mt-0.5 whitespace-pre-wrap pl-3">{entry.response_text}</div>
                        </details>
                      )}
                      {entry.error && <div className="text-red-400 truncate pl-3">{entry.error.slice(0, 80)}</div>}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}

        {/* Section 3: Events */}
        {events.length > 0 && (
          <div className="border border-gray-800/60 rounded-xl overflow-hidden bg-gray-900/20">
            <div className="px-4 py-2.5 bg-gray-900/60 border-b border-gray-800/40">
              <div className="flex items-center gap-3">
                <span className="w-6 h-6 rounded-full bg-red-600/20 border border-red-500/40 flex items-center justify-center text-[11px] font-bold text-red-400">3</span>
                <span className="text-xs font-medium text-gray-300">Deployment Events</span>
              </div>
            </div>
            <div className="p-3 max-h-32 overflow-y-auto space-y-1">
              {events.map((e, i) => (
                <div key={i} className="text-[10px] flex items-center gap-2 px-2 py-1 rounded bg-gray-800/30 text-gray-400">
                  <span className="text-gray-600 font-mono">{e.time}</span>
                  <span>{e.message}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* ═══ Right Panel: Live Dashboard ═══ */}
      <div className="w-[420px] border-l border-gray-800/50 bg-[#0b1018] overflow-y-auto flex-shrink-0">
        <div className="p-4 space-y-4">
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium text-gray-300">📊 Rollout Dashboard</span>
            {running && <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />}
          </div>

          {/* Top metrics */}
          <div className="grid grid-cols-3 gap-2">
            <div className="bg-gray-900/40 border border-gray-800/40 rounded-lg p-2 text-center">
              <div className="text-[9px] text-gray-500 uppercase">Requests</div>
              <div className="text-lg font-bold text-gray-200">{stats.total_requests || 0}</div>
            </div>
            <div className="bg-gray-900/40 border border-gray-800/40 rounded-lg p-2 text-center">
              <div className="text-[9px] text-gray-500 uppercase">Mode</div>
              <div className="text-sm font-bold text-orange-400">{mode === 'ab_test' ? 'A/B' : mode === 'canary' ? 'Canary' : 'Shadow'}</div>
            </div>
            <div className="bg-gray-900/40 border border-gray-800/40 rounded-lg p-2 text-center">
              <div className="text-[9px] text-gray-500 uppercase">Variants</div>
              <div className="text-lg font-bold text-gray-200">{Object.keys(stats.variant_counts || {}).length}</div>
            </div>
          </div>

          {/* Traffic Split Bar */}
          {stats.variant_pcts && Object.keys(stats.variant_pcts).length > 0 && (
            <div className="border border-gray-800/40 rounded-lg overflow-hidden">
              <div className="px-3 py-2 bg-gray-900/60 border-b border-gray-800/40">
                <span className="text-[10px] text-gray-400 font-bold uppercase">Traffic Split</span>
              </div>
              <div className="p-3">
                <div className="flex h-4 rounded-full overflow-hidden bg-gray-800">
                  {Object.entries(stats.variant_pcts).map(([variant, pct]) => {
                    const colors = COLOR_MAP[variant] || COLOR_MAP.unknown
                    return <div key={variant} className={`${colors.dot} transition-all duration-500`} style={{ width: `${pct}%` }} />
                  })}
                </div>
                <div className="flex justify-between mt-2">
                  {Object.entries(stats.variant_pcts).map(([variant, pct]) => {
                    const colors = COLOR_MAP[variant] || COLOR_MAP.unknown
                    return (
                      <div key={variant} className="flex items-center gap-1.5 text-[10px]">
                        <span className={`w-2 h-2 rounded-full ${colors.dot}`} />
                        <span className={colors.text}>{variant}</span>
                        <span className="text-gray-500">{pct}%</span>
                      </div>
                    )
                  })}
                </div>
              </div>
            </div>
          )}

          {/* Per-Variant Stats */}
          {stats.variant_avg_cost && Object.keys(stats.variant_avg_cost).length > 0 && (
            <div className="border border-gray-800/40 rounded-lg overflow-hidden">
              <div className="px-3 py-2 bg-gray-900/60 border-b border-gray-800/40">
                <span className="text-[10px] text-gray-400 font-bold uppercase">Per-Variant Metrics</span>
              </div>
              <div className="divide-y divide-gray-800/30">
                {Object.entries(stats.variant_counts || {}).map(([variant, count]) => {
                  const colors = COLOR_MAP[variant] || COLOR_MAP.unknown
                  return (
                    <div key={variant} className="px-3 py-2.5">
                      <div className="flex items-center gap-2 mb-1">
                        <span className={`w-2 h-2 rounded-full ${colors.dot}`} />
                        <span className={`text-[11px] font-medium ${colors.text}`}>{variant}</span>
                        <span className="text-[9px] text-gray-600 ml-auto">{count} requests</span>
                      </div>
                      <div className="grid grid-cols-2 gap-2 text-[10px]">
                        <div><span className="text-gray-600">Avg Cost:</span> <span className="text-gray-300">${(stats.variant_avg_cost[variant] || 0).toFixed(6)}</span></div>
                        <div><span className="text-gray-600">Avg Latency:</span> <span className="text-gray-300">{(stats.variant_avg_latency[variant] || 0).toFixed(0)}ms</span></div>
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>
          )}

          {/* Sticky Verification (A/B mode only) */}
          {mode === 'ab_test' && stats.sticky_sample && Object.keys(stats.sticky_sample).length > 0 && (
            <div className="border border-gray-800/40 rounded-lg overflow-hidden">
              <div className="px-3 py-2 bg-gray-900/60 border-b border-gray-800/40">
                <span className="text-[10px] text-gray-400 font-bold uppercase">Sticky Assignment (same user → same variant)</span>
              </div>
              <div className="p-3 space-y-1">
                {Object.entries(stats.sticky_sample).map(([uid, variant]) => {
                  const colors = COLOR_MAP[variant] || COLOR_MAP.unknown
                  return (
                    <div key={uid} className="flex items-center gap-2 text-[10px]">
                      <span className="text-gray-500 font-mono">{uid}</span>
                      <span className="text-gray-600">→</span>
                      <span className={`px-1.5 py-0.5 rounded ${colors.bg} ${colors.text} border ${colors.border} text-[9px]`}>{variant}</span>
                    </div>
                  )
                })}
              </div>
            </div>
          )}

          {/* Canary Health (canary mode only) */}
          {mode === 'canary' && stats.canary_stats && (
            <div className="border border-gray-800/40 rounded-lg overflow-hidden">
              <div className="px-3 py-2 bg-gray-900/60 border-b border-gray-800/40">
                <span className="text-[10px] text-gray-400 font-bold uppercase">Canary Health</span>
              </div>
              <div className="p-3 text-[10px] space-y-1.5">
                <div className="flex justify-between"><span className="text-gray-500">Status:</span><span className={stats.canary_stats.rolled_back ? 'text-red-400 font-bold' : stats.canary_stats.promoted ? 'text-green-400 font-bold' : 'text-yellow-400'}>{stats.canary_stats.rolled_back ? '⛔ Rolled Back' : stats.canary_stats.promoted ? '✅ Promoted' : '🔄 Active'}</span></div>
                <div className="flex justify-between"><span className="text-gray-500">Canary Requests:</span><span className="text-gray-300">{stats.canary_stats.canary_requests}</span></div>
                <div className="flex justify-between"><span className="text-gray-500">Baseline Requests:</span><span className="text-gray-300">{stats.canary_stats.baseline_requests}</span></div>
                <div className="flex justify-between"><span className="text-gray-500">Error Rate:</span><span className={stats.canary_stats.canary_error_rate > 0.1 ? 'text-red-400' : 'text-gray-300'}>{(stats.canary_stats.canary_error_rate * 100).toFixed(1)}% <span className="text-gray-600">(threshold: 10%)</span></span></div>
                <div className="flex justify-between"><span className="text-gray-500">P95 Latency:</span><span className={stats.canary_stats.canary_p95_latency_ms > (stats.canary_stats.max_latency_threshold_ms || 5000) ? 'text-red-400' : 'text-gray-300'}>{stats.canary_stats.canary_p95_latency_ms || 0}ms <span className="text-gray-600">(threshold: {stats.canary_stats.max_latency_threshold_ms || 5000}ms)</span></span></div>
                {stats.canary_stats.rolled_back && (
                  <div className="mt-2 p-2 bg-red-900/10 border border-red-800/30 rounded text-[9px] text-red-300">
                    <strong>Rollback reason:</strong> {stats.canary_stats.canary_error_rate > 0.1
                      ? `Error rate ${(stats.canary_stats.canary_error_rate * 100).toFixed(1)}% exceeded 10% threshold`
                      : `P95 latency ${stats.canary_stats.canary_p95_latency_ms || '?'}ms exceeded ${stats.canary_stats.max_latency_threshold_ms || 5000}ms threshold`}
                    <div className="text-red-400/70 mt-1">All traffic reverted to baseline. Canary model no longer receives requests.</div>
                  </div>
                )}
                {stats.canary_stats.promoted && (
                  <div className="mt-2 p-2 bg-green-900/10 border border-green-800/30 rounded text-[9px] text-green-300">
                    <strong>Promoted!</strong> Canary passed all thresholds after {stats.canary_stats.canary_requests} requests. Ready for full rollout.
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Shadow Stats (shadow mode only) */}
          {mode === 'shadow' && stats.shadow_stats && (
            <div className="border border-gray-800/40 rounded-lg overflow-hidden">
              <div className="px-3 py-2 bg-gray-900/60 border-b border-gray-800/40">
                <span className="text-[10px] text-gray-400 font-bold uppercase">Shadow Mirroring</span>
              </div>
              <div className="p-3 text-[10px] space-y-1">
                <div className="flex justify-between"><span className="text-gray-500">Status:</span><span className="text-green-400">{stats.shadow_stats.active ? '🔄 Active' : '⏸ Inactive'}</span></div>
                <div className="flex justify-between"><span className="text-gray-500">Requests Mirrored:</span><span className="text-gray-300">{stats.shadow_stats.total || 0}</span></div>
                <div className="flex justify-between"><span className="text-gray-500">Sample Rate:</span><span className="text-gray-300">30%</span></div>
                <div className="text-[9px] text-gray-600 mt-2 pt-2 border-t border-gray-800/30">Shadow responses are logged in background — never returned to user. Compare quality offline.</div>
              </div>
            </div>
          )}

          {/* How It Works */}
          {stats.total_requests > 0 && (
            <div className="border border-gray-800/40 rounded-lg overflow-hidden">
              <div className="px-3 py-2 bg-gray-900/60 border-b border-gray-800/40">
                <span className="text-[10px] text-gray-400 font-bold uppercase">How It Works</span>
              </div>
              <div className="p-3 text-[10px] text-gray-400 space-y-2">
                {mode === 'ab_test' && <>
                  <div className="flex items-start gap-2"><span className="text-blue-400">1.</span><span>Traffic is split between <strong className="text-blue-300">control</strong> and <strong className="text-orange-300">treatment</strong> by weight (50/50)</span></div>
                  <div className="flex items-start gap-2"><span className="text-blue-400">2.</span><span>With <code className="text-orange-300">sticky: true</code>, the same user always gets the same variant (hash-based)</span></div>
                  <div className="flex items-start gap-2"><span className="text-blue-400">3.</span><span>Compare cost, latency, and quality between variants to decide which model wins</span></div>
                </>}
                {mode === 'canary' && <>
                  <div className="flex items-start gap-2"><span className="text-green-400">1.</span><span>Most traffic goes to <strong className="text-blue-300">baseline</strong>, only 20% goes to <strong className="text-green-300">canary</strong></span></div>
                  <div className="flex items-start gap-2"><span className="text-green-400">2.</span><span>If canary error rate exceeds threshold → <strong className="text-red-300">auto-rollback</strong></span></div>
                  <div className="flex items-start gap-2"><span className="text-green-400">3.</span><span>If canary performs well after N requests → <strong className="text-green-300">auto-promote</strong></span></div>
                </>}
                {mode === 'shadow' && <>
                  <div className="flex items-start gap-2"><span className="text-purple-400">1.</span><span>All traffic goes to <strong className="text-purple-300">primary</strong> (user sees this response)</span></div>
                  <div className="flex items-start gap-2"><span className="text-purple-400">2.</span><span>30% of requests are <strong className="text-gray-300">mirrored</strong> to the shadow model in background</span></div>
                  <div className="flex items-start gap-2"><span className="text-purple-400">3.</span><span>Shadow calls run <strong className="text-purple-300">async</strong> in a thread pool — zero impact on user-facing latency</span></div>
                  <div className="flex items-start gap-2"><span className="text-purple-400">4.</span><span>Shadow responses are logged but never returned — zero user impact</span></div>
                </>}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
