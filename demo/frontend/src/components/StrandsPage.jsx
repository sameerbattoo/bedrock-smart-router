import { useState, useEffect, useRef } from 'react'
import { Md, MetricWithDelta, STREAM_API, API } from './shared'
import AnalyticsPanel from './AnalyticsPanel'

export default function StrandsPage({ history, setHistory, onRun, restoreState }) {
  const [options, setOptions] = useState({ baseline_models: [], router_strategies: [] })
  const [baselineModel, setBaselineModel] = useState('sonnet')
  const [routerStrategy, setRouterStrategy] = useState('balanced')
  const [tools, setTools] = useState({ docs: true, diagram: true })
  const [toolStatus, setToolStatus] = useState({ docs: 'stopped', diagram: 'stopped' }) // stopped | starting | ready
  const [message, setMessage] = useState('')
  const [loading, setLoading] = useState(false)
  const [baselineSessionId, setBaselineSessionId] = useState('')
  const [routerSessionId, setRouterSessionId] = useState('')
  const [conversation, setConversation] = useState([]) // [{user, baseline, router, metrics}]

  const chatEndRef = useRef(null)

  useEffect(() => {
    fetch(`${API}/options`).then(r => r.json()).then(setOptions).catch(() => {})
    // MCP servers start at backend startup — poll status
    setToolStatus({ docs: 'starting', diagram: 'starting' })
  }, [])

  // Poll tool status until ready
  useEffect(() => {
    if (toolStatus.docs === 'ready' && (toolStatus.diagram === 'ready' || toolStatus.diagram === 'failed')) return
    const interval = setInterval(() => {
      fetch(`${STREAM_API}/strands-status`).then(r => r.json())
        .then(d => {
          setToolStatus({ docs: d.docs || 'stopped', diagram: d.diagram || 'stopped' })
          if (d.docs === 'ready' || d.docs === 'failed') clearInterval(interval)
        }).catch(() => {})
    }, 2000)
    return () => clearInterval(interval)
  }, [toolStatus.docs])

  // Restore from history
  useEffect(() => {
    if (restoreState && restoreState.use_case === 'strands') {
      const h = restoreState
      if (h.conversation) setConversation(h.conversation)
      if (h.baseline_session_id) setBaselineSessionId(h.baseline_session_id)
      if (h.router_session_id) setRouterSessionId(h.router_session_id)
      if (h.baseline_model) setBaselineModel(h.baseline_model)
      if (h.router_strategy) setRouterStrategy(h.router_strategy)
      if (h.tools) setTools(h.tools)
    }
  }, [restoreState])

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [conversation])

  function resetConversation() {
    // Save current conversation to history before resetting
    if (conversation.length > 0) {
      setHistory(prev => [...prev, {
        id: Date.now(), timestamp: Date.now(), use_case: 'strands',
        prompt: conversation.map(t => t.user).join(' → '),
        baseline_session_id: baselineSessionId,
        router_session_id: routerSessionId,
        conversation: conversation,
        baseline_model: baselineModel,
        router_strategy: routerStrategy,
        tools: { ...tools },
        router_model: conversation[conversation.length - 1]?.router?.model_used || 'unknown',
        complexity: conversation[conversation.length - 1]?.router?.complexity_detected || '',
        router_latency: Math.round(conversation.reduce((s, t) => s + (t.router?.latency_ms || 0), 0) / conversation.length),
        router_cost: conversation.reduce((s, t) => s + (t.router?.cost || 0), 0),
        baseline_latency: Math.round(conversation.reduce((s, t) => s + (t.baseline?.latency_ms || 0), 0) / conversation.length),
        baseline_cost: 0,
        savings_pct: 0,
        router_metrics: null, baseline_metrics: null,
      }])
    }
    // Start fresh conversation (new session IDs, keep backend sessions alive for history restore)
    setBaselineSessionId('')
    setRouterSessionId('')
    setConversation([])
  }

  async function sendMessage() {
    if (!message.trim() || loading) return
    if (onRun) onRun()
    setLoading(true)
    const userMsg = message.trim()
    setMessage('')

    // Add user message to conversation immediately
    const turnIdx = conversation.length
    setConversation(prev => [...prev, { user: userMsg, timestamp: new Date().toLocaleTimeString(), baseline: null, router: null }])

    const form = new FormData()
    form.append('message', userMsg)
    form.append('baseline_session_id', baselineSessionId)
    form.append('router_session_id', routerSessionId)
    form.append('baseline_model', baselineModel)
    form.append('router_strategy', routerStrategy)
    form.append('tools_enabled', JSON.stringify(tools))

    try {
      const res = await fetch(`${STREAM_API}/strands-chat`, { method: 'POST', body: form })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = '', currentEventType = null

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (line.startsWith('event: ')) { currentEventType = line.slice(7).trim() }
          else if (line.startsWith('data: ') && currentEventType) {
            const eventType = currentEventType; currentEventType = null
            const parsed = JSON.parse(line.slice(6))

            if (eventType === 'session_init') {
              if (!baselineSessionId) setBaselineSessionId(parsed.baseline_session_id)
              if (!routerSessionId) setRouterSessionId(parsed.router_session_id)
            } else if (eventType === 'baseline_complete') {
              setConversation(prev => prev.map((turn, i) => i === turnIdx ? { ...turn, baseline: parsed } : turn))
              setToolStatus({ docs: 'ready', diagram: 'ready' })
            } else if (eventType === 'baseline_error') {
              setConversation(prev => prev.map((turn, i) => i === turnIdx ? { ...turn, baseline: { error: parsed.error, latency_ms: parsed.latency_ms } } : turn))
            } else if (eventType === 'router_complete') {
              setConversation(prev => prev.map((turn, i) => i === turnIdx ? { ...turn, router: parsed } : turn))
              // Save to history
              setHistory(prev => [...prev, {
                id: Date.now(), timestamp: Date.now(), use_case: 'strands',
                prompt: userMsg, router_model: parsed.model_used,
                complexity: parsed.complexity_detected,
                router_latency: parsed.latency_ms, router_cost: parsed.cost || 0,
                baseline_latency: 0, baseline_cost: 0, savings_pct: 0,
                router_metrics: parsed, baseline_metrics: null,
              }])
            } else if (eventType === 'router_error') {
              setConversation(prev => prev.map((turn, i) => i === turnIdx ? { ...turn, router: { error: parsed.error, latency_ms: parsed.latency_ms } } : turn))
            } else if (eventType === 'done') {
              setLoading(false)
            }
          }
        }
      }
    } catch (err) { alert(`Error: ${err.message}`); setLoading(false) }
  }

  return (
    <div className="flex-1 flex min-w-0 overflow-hidden">
      <div className="flex-1 flex flex-col min-w-0">
        {/* ─── Agent Header ─── */}
        <div className="px-4 py-2 border-b border-gray-800/30 bg-gradient-to-r from-orange-950/20 to-purple-950/20">
          <div className="flex items-center gap-2">
            <span className="text-lg">☁️</span>
            <span className="text-sm font-bold text-gray-200">AWS Buddy</span>
            <span className="text-[10px] text-gray-500 bg-gray-800 px-2 py-0.5 rounded-full">AWS Tech Assistant</span>
            <span className="text-[9px] text-gray-600 ml-auto">Powered by Strands Agents + MCP</span>
          </div>
        </div>

        {/* ─── Control Panel (top) ─── */}
        <div className="px-4 py-3 border-b border-gray-800/50 bg-gray-900/40">
          <div className="flex gap-3 flex-wrap">
            {/* Baseline model (left half) */}
            <div className="flex-1 flex items-center gap-2">
              <span className="text-[10px] text-gray-500 font-bold uppercase">Baseline</span>
              <div className="flex bg-gray-900/80 rounded-lg p-0.5 border border-gray-800/50">
                {options.baseline_models.map(m => (
                  <button key={m.id} onClick={() => setBaselineModel(m.id)}
                    className={`text-[11px] px-2.5 py-1 rounded-md font-medium transition-all ${baselineModel === m.id ? 'bg-blue-600 text-white shadow-sm' : 'text-gray-500 hover:text-gray-300'}`}>
                    {m.label}
                  </button>
                ))}
              </div>
            </div>
            {/* Strategy (right half) */}
            <div className="flex-1 flex items-center gap-2">
              <span className="text-[10px] text-gray-500 font-bold uppercase">Strategy</span>
              <div className="flex bg-gray-900/80 rounded-lg p-0.5 border border-gray-800/50">
                {(options.router_strategies || []).map(s => (
                  <button key={s} onClick={() => setRouterStrategy(s)}
                    className={`text-[11px] px-2.5 py-1 rounded-md font-medium transition-all capitalize ${routerStrategy === s ? 'bg-orange-600 text-white shadow-sm' : 'text-gray-500 hover:text-gray-300'}`}>
                    {s.replace('-optimized','').replace('-',' ')}
                  </button>
                ))}
              </div>
              {/* Reset */}
              <button onClick={resetConversation}
                className="ml-auto text-[10px] text-red-400 hover:text-red-300 px-2 py-1 rounded hover:bg-red-900/20 border border-red-900/30 transition-all">
                Reset Chat
              </button>
            </div>
          </div>
        </div>

        {/* ─── Chat Area (scrollable) ─── */}
        <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4">
          {conversation.length === 0 && (
            <div className="text-center text-gray-600 mt-20">
              <div className="text-4xl mb-3">🤖</div>
              <div className="text-sm font-medium text-gray-400">AWS Tech Assistant</div>
              <div className="text-xs text-gray-600 mt-1">Ask anything about AWS — architecture, best practices, cost optimization</div>
              <div className="text-[10px] text-gray-700 mt-3">Both agents have access to AWS Documentation and Diagram tools</div>
            </div>
          )}

          {conversation.map((turn, i) => (
            <div key={i} className="space-y-3 animate-slideDown">
              {/* User message */}
              <div className="flex justify-center">
                <div className="bg-gray-800/60 border border-gray-700/50 rounded-xl px-4 py-2 max-w-[70%]">
                  <div className="flex items-center gap-2">
                    <span className="text-[9px] text-gray-500">You</span>
                    <span className="text-[9px] text-gray-600">{turn.timestamp}</span>
                  </div>
                  <div className="text-sm text-gray-200">{turn.user}</div>
                </div>
              </div>

              {/* Agent responses side-by-side */}
              <div className="flex gap-3">
                {/* Baseline response */}
                <div className="flex-1 border border-blue-900/30 rounded-lg overflow-hidden hover:border-blue-500/50 hover:shadow-lg hover:-translate-y-1 transition-all duration-200">
                  <div className="px-3 py-2 border-b border-blue-900/30 bg-blue-950/20">
                    <div className="flex items-center justify-between mb-1">
                      <div className="flex items-center gap-2">
                        <span className="text-[10px] font-medium text-blue-400">🧊 Baseline</span>
                        {turn.baseline && !turn.baseline.error && <span className="text-[9px] bg-blue-900/40 text-blue-300 px-1.5 py-0.5 rounded">{turn.baseline.model_used}</span>}
                      </div>
                    </div>
                    {turn.baseline && !turn.baseline.error && (
                      <div className="grid grid-cols-3 gap-2">
                        <div className="text-center"><div className="text-[9px] text-gray-500">Latency</div><div className="text-xs font-mono text-gray-300">{turn.baseline.latency_ms}ms</div></div>
                        <div className="text-center"><div className="text-[9px] text-gray-500">Tokens</div><div className="text-xs font-mono text-gray-300">{turn.baseline.input_tokens || '—'}↓ {turn.baseline.output_tokens || '—'}↑</div></div>
                        <div className="text-center"><div className="text-[9px] text-gray-500">Cost</div><div className="text-xs font-mono text-gray-300">{turn.baseline.cost != null ? `$${turn.baseline.cost}` : '—'}</div></div>
                      </div>
                    )}
                  </div>
                  <div className="p-3 text-sm text-gray-300 bg-[#080d18] min-h-[60px]">
                    {turn.baseline === null ? (
                      <div className="animate-pulse text-gray-600 text-xs">Generating...</div>
                    ) : turn.baseline.error ? (
                      <div className="text-red-400 text-xs">❌ {turn.baseline.error}</div>
                    ) : (
                      <Md variant="baseline">{turn.baseline.response_text}</Md>
                    )}
                  </div>
                </div>

                {/* Router response */}
                <div className="flex-1 border border-orange-900/30 rounded-lg overflow-hidden hover:border-orange-500/50 hover:shadow-lg hover:-translate-y-1 transition-all duration-200">
                  <div className="px-3 py-2 border-b border-orange-900/30 bg-orange-950/20">
                    <div className="flex items-center justify-between mb-1">
                      <div className="flex items-center gap-2">
                        <span className="text-[10px] font-medium text-orange-400">⚡ Smart Router</span>
                        {turn.router && !turn.router.error && <span className="text-[9px] bg-orange-900/40 text-orange-300 px-1.5 py-0.5 rounded">{turn.router.model_used}</span>}
                        {turn.router?.complexity_detected && <span className="text-[9px] bg-purple-900/40 text-purple-300 px-1.5 py-0.5 rounded">{turn.router.complexity_detected}</span>}
                      </div>
                    </div>
                    {turn.router && !turn.router.error && (
                      <div className="grid grid-cols-3 gap-2">
                        <MetricWithDelta label="Latency" value={`${turn.router.latency_ms}ms`} baseline={turn.baseline?.latency_ms} current={turn.router.latency_ms} lower />
                        <MetricWithDelta label="Tokens" value={`${turn.router.input_tokens || 0}↓ ${turn.router.output_tokens || 0}↑`} baseline={turn.baseline ? (turn.baseline.input_tokens||0)+(turn.baseline.output_tokens||0) : null} current={(turn.router.input_tokens||0)+(turn.router.output_tokens||0)} lower />
                        <MetricWithDelta label="Cost" value={`$${turn.router.cost || 0}`} baseline={turn.baseline?.cost} current={turn.router.cost} lower />
                      </div>
                    )}
                  </div>
                  <div className="p-3 text-sm text-gray-300 bg-[#0d1210] min-h-[60px]">
                    {turn.router === null ? (
                      <div className="animate-pulse text-gray-600 text-xs">Generating...</div>
                    ) : turn.router.error ? (
                      <div className="text-red-400 text-xs">❌ {turn.router.error}</div>
                    ) : (
                      <Md variant="router">{turn.router.response_text}</Md>
                    )}
                  </div>
                </div>
              </div>
            </div>
          ))}
          <div ref={chatEndRef} />
        </div>

        {/* ─── Chat Input (bottom, fixed) ─── */}
        <div className="px-4 py-3 border-t border-gray-800/50 bg-[#0e1420]">
          <div className="relative">
            <textarea value={message} onChange={e => setMessage(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage() }}}
              placeholder="Ask about AWS architecture, services, best practices..."
              rows={2}
              className="w-full bg-gray-900/60 border border-gray-700/50 rounded-xl px-4 py-3 pr-20 text-sm text-gray-200 placeholder-gray-500 focus:outline-none focus:border-orange-500/50 focus:ring-1 focus:ring-orange-500/20 resize-none" />
            <button onClick={sendMessage} disabled={loading || !message.trim()}
              className="absolute right-3 bottom-3 bg-orange-600 hover:bg-orange-700 disabled:opacity-40 text-white text-xs font-medium rounded-lg px-3 py-1.5 flex items-center gap-1.5 transition-all">
              {loading ? <svg className="animate-spin h-3 w-3" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"/><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/></svg> : null}
              {loading ? 'Thinking...' : 'Send'}
            </button>
          </div>
          <div className="flex items-center gap-2 mt-1.5 text-[9px] text-gray-600">
            <span className="text-[10px] text-gray-500 font-bold uppercase">Tools</span>
            <button onClick={() => setTools(t => ({...t, docs: !t.docs}))}
              className={`text-[10px] px-2 py-0.5 rounded-lg font-medium transition-all flex items-center gap-1 ${
                !tools.docs ? 'bg-gray-800 text-gray-600 border border-gray-700 line-through' :
                toolStatus.docs === 'ready' ? 'bg-green-600/20 text-green-400 border border-green-600/40' :
                'bg-orange-600/20 text-orange-400 border border-orange-600/40 animate-pulse'
              }`}>
              📚 AWS Docs {toolStatus.docs === 'starting' && '⏳'}
            </button>
            <button onClick={() => setTools(t => ({...t, diagram: !t.diagram}))}
              className={`text-[10px] px-2 py-0.5 rounded-lg font-medium transition-all flex items-center gap-1 ${
                !tools.diagram ? 'bg-gray-800 text-gray-600 border border-gray-700 line-through' :
                toolStatus.diagram === 'ready' ? 'bg-green-600/20 text-green-400 border border-green-600/40' :
                'bg-orange-600/20 text-orange-400 border border-orange-600/40 animate-pulse'
              }`}>
              🎨 Diagrams {toolStatus.diagram === 'starting' && '⏳'}
            </button>
            <span className="ml-2 text-gray-700">•</span>
            <span>Turn {conversation.length + 1}</span>
            <span>•</span>
            <span>Sessions: {baselineSessionId ? baselineSessionId.slice(0,8) : 'new'}... / {routerSessionId ? routerSessionId.slice(0,8) : 'new'}...</span>
          </div>
        </div>
      </div>

      {/* ─── Analytics Panel (right) ─── */}
      <AnalyticsPanel history={history} />
    </div>
  )
}
