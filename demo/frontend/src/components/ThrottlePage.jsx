import { useState, useEffect, useRef } from 'react'
import { Md, ExplainPopup, ResponseWithThinking, STREAM_API, API } from './shared'

// ─── Collapsible Step Section ──────────────────────────────────────
function StepSection({ number, title, visible, expanded, onToggle, children }) {
  if (!visible) return null
  return (
    <div className="border border-gray-800/60 rounded-xl overflow-hidden bg-gray-900/20 animate-slideDown">
      <button onClick={onToggle}
        className="w-full flex items-center gap-3 px-4 py-2.5 bg-gray-900/60 hover:bg-gray-800/60 transition-all border-b border-gray-800/40">
        <span className="w-6 h-6 rounded-full bg-orange-600/20 border border-orange-500/40 flex items-center justify-center text-[11px] font-bold text-orange-400">{number}</span>
        <span className="text-xs font-medium text-gray-300 flex-1 text-left">{title}</span>
        <svg className={`w-4 h-4 text-gray-500 transition-transform ${expanded ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7"/></svg>
      </button>
      {expanded && <div className="p-4">{children}</div>}
    </div>
  )
}

// ─── Timeline Entry ────────────────────────────────────────────────
function TimelineEntry({ status, model, attempt, maxRetries, delay, error, isLast }) {
  const isSuccess = status === 'success'
  const isFallback = status === 'fallback'
  return (
    <div className="flex items-start gap-3 relative">
      {/* Connector line */}
      {!isLast && <div className="absolute left-[11px] top-6 bottom-0 w-0.5 bg-gray-800" />}
      {/* Dot */}
      <div className={`w-6 h-6 rounded-full flex items-center justify-center flex-shrink-0 ${
        isSuccess ? 'bg-green-900/40 border border-green-500/50' :
        isFallback ? 'bg-yellow-900/40 border border-yellow-500/50' :
        'bg-red-900/40 border border-red-500/50'
      }`}>
        {isSuccess ? <span className="text-green-400 text-xs">✓</span> :
         isFallback ? <span className="text-yellow-400 text-xs">↪</span> :
         <span className="text-red-400 text-xs">✗</span>}
      </div>
      {/* Content */}
      <div className="flex-1 pb-4">
        <div className="flex items-center gap-2">
          <span className={`text-[11px] font-medium ${isSuccess ? 'text-green-400' : isFallback ? 'text-yellow-400' : 'text-red-400'}`}>
            {isSuccess ? 'Success' : isFallback ? 'Fallback' : `Attempt ${attempt}/${maxRetries}`}
          </span>
          <span className="text-[10px] text-gray-500">→</span>
          <span className="text-[11px] text-gray-300 font-mono">{model}</span>
          {delay > 0 && <span className="text-[9px] text-yellow-500 bg-yellow-900/20 px-1.5 py-0.5 rounded ml-auto">⏱ {delay >= 1000 ? `${(delay/1000).toFixed(1)}s` : `${delay}ms`} backoff</span>}
        </div>
        {error && <div className="text-[10px] text-red-400/80 mt-0.5 font-mono">HTTP 429 — {error}</div>}
        {isFallback && <div className="text-[10px] text-yellow-400/80 mt-0.5">Primary exhausted retries, switching to fallback model</div>}
        {isSuccess && <div className="text-[10px] text-green-400/80 mt-0.5">Response received successfully</div>}
      </div>
    </div>
  )
}

export default function ThrottlePage({ history, setHistory, onRun, restoreState }) {
  const [templates, setTemplates] = useState([])
  const [options, setOptions] = useState({ preferred_models: [] })
  const [prompt, setPrompt] = useState('')
  const [systemPrompt, setSystemPrompt] = useState('')
  const [difficulty, setDifficulty] = useState('simple')
  const [throttleModel, setThrottleModel] = useState('anthropic.claude-sonnet-4-6')
  const [loading, setLoading] = useState(false)
  const [routerText, setRouterText] = useState('')

  // Step visibility
  const [step2Visible, setStep2Visible] = useState(false)
  const [step3Visible, setStep3Visible] = useState(false)
  const [step4Visible, setStep4Visible] = useState(false)
  const [step1Expanded, setStep1Expanded] = useState(true)
  const [step2Expanded, setStep2Expanded] = useState(true)
  const [step3Expanded, setStep3Expanded] = useState(true)
  const [step4Expanded, setStep4Expanded] = useState(true)

  // Timeline events
  const [baselineTimeline, setBaselineTimeline] = useState([])
  const [routerTimeline, setRouterTimeline] = useState([])
  const [baselineFailed, setBaselineFailed] = useState(false)
  const [routerResult, setRouterResult] = useState(null)
  const [explainPopup, setExplainPopup] = useState(null)
  const [patchingOverlay, setPatchingOverlay] = useState(false)

  useEffect(() => {
    fetch(`${API}/templates`).then(r => r.json()).then(setTemplates).catch(() => {})
    fetch(`${API}/options`).then(r => r.json()).then(setOptions).catch(() => {})
  }, [])

  // Restore state from history navigation
  useEffect(() => {
    if (restoreState && restoreState.use_case === 'throttling') {
      const h = restoreState
      setPrompt(h.prompt || '')
      setSystemPrompt(h.system_prompt || '')
      setThrottleModel(h.throttle_model || 'anthropic.claude-sonnet-4-6')
      setStep1Expanded(false)
      setStep2Visible(true)
      setStep2Expanded(false)
      setStep3Visible(true)
      setStep3Expanded(false)
      setStep4Visible(true)
      setStep4Expanded(true)
      // Restore results
      if (h.baseline_timeline) setBaselineTimeline(h.baseline_timeline)
      if (h.baseline_failed != null) setBaselineFailed(h.baseline_failed)
      if (h.router_timeline) setRouterTimeline(h.router_timeline)
      if (h.router_result) { setRouterResult(h.router_result); setRouterText(h.router_result.response_text || '') }
    }
  }, [restoreState])

  const filtered = templates.filter(t => t.difficulty === difficulty)

  function selectTemplate(t) {
    setPrompt(t.prompt)
    setSystemPrompt(t.system_prompt || '')
    setStep2Visible(true)
    setStep2Expanded(true)
    setStep1Expanded(false)
    setStep3Visible(false)
    setStep4Visible(false)
    resetResults()
  }

  function handleManualEntry() {
    setPrompt('')
    setSystemPrompt('')
    setStep2Visible(true)
    setStep2Expanded(true)
    setStep1Expanded(false)
    setStep3Visible(false)
    setStep4Visible(false)
    resetResults()
  }

  function handlePromptReady() {
    if (!prompt.trim()) return
    setStep3Visible(true)
    setStep3Expanded(true)
    setStep2Expanded(false)
  }

  function resetResults() {
    setBaselineTimeline([])
    setRouterTimeline([])
    setBaselineFailed(false)
    setRouterResult(null)
    setRouterText('')
  }

  async function runThrottleDemo() {
    if (!prompt.trim() || !throttleModel || loading) return
    if (onRun) onRun()

    // Show patching overlay for 3 seconds
    setPatchingOverlay(true)
    await new Promise(resolve => setTimeout(resolve, 3000))
    setPatchingOverlay(false)

    setLoading(true)
    resetResults()
    setStep4Visible(true)
    setStep4Expanded(true)
    setStep3Expanded(false)
    setStep2Expanded(false)
    setStep1Expanded(false)

    try {
      const form = new FormData()
      form.append('prompt', prompt.trim())
      form.append('system_prompt', systemPrompt.trim())
      form.append('throttle_model', throttleModel)

      const res = await fetch(`${STREAM_API}/throttle-demo`, { method: 'POST', body: form })
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

            if (eventType === 'baseline_attempt') {
              setBaselineTimeline(prev => [...prev, { status: 'failed', model: parsed.model, attempt: parsed.attempt, maxRetries: parsed.max_retries, delay: parsed.delay_ms, error: parsed.error }])
            } else if (eventType === 'baseline_failed') {
              setBaselineFailed(true)
              setBaselineTimeline(prev => [...prev, { status: 'failed', model: parsed.model, attempt: 1, maxRetries: 1, delay: 0, error: parsed.error, request_id: parsed.request_id }])
            } else if (eventType === 'router_attempt') {
              setRouterTimeline(prev => [...prev, { status: 'failed', model: parsed.model, attempt: parsed.attempt, maxRetries: 4, delay: parsed.backoff_ms || 0, error: parsed.error }])
            } else if (eventType === 'router_fallback') {
              if (parsed.success) {
                // This is the successful fallback — will be followed by router_complete
              } else {
                setRouterTimeline(prev => [...prev, { status: 'fallback', model: `${parsed.to_model} (${parsed.to_model_id})`, attempt: 0, maxRetries: 0, delay: 0 }])
              }
            } else if (eventType === 'router_chunk') {
              setRouterText(prev => prev + parsed.text)
            } else if (eventType === 'router_complete') {
              setRouterResult(parsed)
              setRouterText(parsed.response_text || '')
              setLoading(false)
              // Add success entry with the actual fallback model
              const fallbackModel = parsed.fallback_to || parsed.model_used
              const fallbackId = parsed.fallback_model_id || parsed.model_id_full || ''
              setRouterTimeline(prev => [...prev, { status: 'success', model: `${fallbackModel}${fallbackId ? ` (${fallbackId})` : ''}`, attempt: 0, maxRetries: 0, delay: 0 }])
              // Save to history with full state
              setHistory(prev => [...prev, {
                id: Date.now(), timestamp: Date.now(), use_case: 'throttling',
                prompt: prompt.trim(), system_prompt: systemPrompt.trim(),
                throttle_model: throttleModel,
                router_model: parsed.model_used, complexity: parsed.complexity_detected,
                router_latency: parsed.latency_ms, router_cost: parsed.cost,
                baseline_cost: 0, savings_pct: 0, baseline_latency: 0,
                router_metrics: parsed, baseline_metrics: null,
                has_error: true, // Baseline intentionally throttled
                // Full state for history restore
                baseline_timeline: [...baselineTimeline],
                baseline_failed: true,
                router_timeline: [...routerTimeline, { status: 'success', model: `${fallbackModel}${fallbackId ? ` (${fallbackId})` : ''}`, attempt: 0, maxRetries: 0, delay: 0 }],
                router_result: parsed,
              }])
            } else if (eventType === 'router_timeline') {
              // Legacy: ignore if we already have real-time events
            } else if (eventType === 'router_error') {
              setRouterResult({ error: parsed.error })
              setLoading(false)
            } else if (eventType === 'done') {
              setLoading(false)
            }
          }
        }
      }
    } catch (err) { alert(`Error: ${err.message}`); setLoading(false) }
  }

  return (
    <div className="flex-1 overflow-y-auto p-4 space-y-3">
      {explainPopup && <ExplainPopup explanation={explainPopup} onClose={() => setExplainPopup(null)} />}

      {/* Patching overlay */}
      {patchingOverlay && (
        <div className="fixed inset-0 z-[200] bg-black/80 flex items-center justify-center">
          <div className="text-center animate-pulse">
            <div className="text-6xl mb-4">🔧</div>
            <div className="text-xl font-bold text-orange-400 mb-2">Patching boto3 client...</div>
            <div className="text-sm text-gray-400 max-w-md">
              Injecting ThrottlingException for <span className="text-red-400 font-mono">{throttleModel}</span> on both the baseline boto3 client and the Smart Router's internal client.
            </div>
            <div className="mt-4 flex justify-center gap-1">
              <div className="w-2 h-2 bg-orange-500 rounded-full animate-bounce" style={{animationDelay: '0ms'}}></div>
              <div className="w-2 h-2 bg-orange-500 rounded-full animate-bounce" style={{animationDelay: '150ms'}}></div>
              <div className="w-2 h-2 bg-orange-500 rounded-full animate-bounce" style={{animationDelay: '300ms'}}></div>
            </div>
          </div>
        </div>
      )}

      {/* ═══ Step 1: Choose Prompt Template ═══ */}
      <StepSection number={1} title="Choose Prompt Template" visible={true} expanded={step1Expanded} onToggle={() => setStep1Expanded(!step1Expanded)}>
        <div className="flex items-center gap-1 mb-3">
          {[
            { id: 'simple', icon: '🟢', label: 'Simple' },
            { id: 'medium', icon: '🔵', label: 'Medium' },
            { id: 'complex', icon: '🟣', label: 'Complex' },
            { id: 'reasoning', icon: '🧠', label: 'Reasoning' },
            { id: 'manual', icon: '✏️', label: 'Manual Entry' },
          ].map(d => (
            <button key={d.id} onClick={() => { if (d.id === 'manual') { handleManualEntry() } else { setDifficulty(d.id) } }}
              className={`text-[11px] px-3 py-1.5 rounded-lg font-medium transition-all ${
                d.id === 'manual'
                  ? 'text-green-400 border border-green-700/50 hover:bg-green-900/20'
                  : difficulty === d.id
                    ? 'bg-orange-600/20 text-orange-300 border border-orange-500/40'
                    : 'text-gray-500 hover:text-gray-300 border border-gray-800/50 hover:border-gray-700'
              }`}>
              {d.icon} {d.label}
            </button>
          ))}
        </div>
        {difficulty !== 'manual' && (
          <div className="grid grid-cols-4 gap-2">
            {filtered.map(t => (
              <button key={t.id} onClick={() => selectTemplate(t)}
                className={`p-3 rounded-lg border text-left transition-all hover:shadow-md ${prompt === t.prompt ? 'border-orange-500/60 bg-orange-950/20' : 'border-gray-800/50 bg-gray-900/30 hover:border-gray-700'}`}>
                <div className="text-[11px] text-gray-300 line-clamp-3 leading-relaxed">{t.prompt}</div>
              </button>
            ))}
          </div>
        )}
      </StepSection>

      {/* ═══ Step 2: System & User Prompt ═══ */}
      <StepSection number={2} title="System & User Prompt" visible={step2Visible} expanded={step2Expanded} onToggle={() => setStep2Expanded(!step2Expanded)}>
        <div className="mb-3">
          <div className="text-[10px] text-gray-500 uppercase font-bold mb-1">System Prompt</div>
          <div className="text-xs text-gray-400 bg-gray-900/60 rounded-lg px-3 py-2 border border-gray-800/50 font-mono">{systemPrompt || '(none)'}</div>
        </div>
        <div className="mb-3">
          <div className="text-[10px] text-gray-500 uppercase font-bold mb-1">User Prompt</div>
          <div className="text-xs text-gray-300 bg-gray-900/60 rounded-lg px-3 py-2 border border-gray-800/50">{prompt}</div>
        </div>
        <div className="flex justify-end">
          <button onClick={handlePromptReady} disabled={!prompt.trim()}
            className="bg-orange-600 hover:bg-orange-700 disabled:opacity-40 text-white text-xs font-bold rounded-lg px-5 py-2 flex items-center gap-2 transition-all shadow-lg shadow-orange-900/20">
            <span>⚡</span> Next →
          </button>
        </div>
      </StepSection>

      {/* ═══ Step 3: Select Model to Throttle ═══ */}
      <StepSection number={3} title="Select Model to Throttle" visible={step3Visible} expanded={step3Expanded} onToggle={() => setStep3Expanded(!step3Expanded)}>
        <div className="mb-3">
          <div className="text-[10px] text-gray-500 uppercase font-bold mb-1.5">Model to Simulate Throttling</div>
          <p className="text-[11px] text-gray-500 mb-3">We patch the boto3 client to throw ThrottlingException (HTTP 429) for this model. Both the baseline and Smart Router use the same patched client. The baseline fails immediately with no retry or fallback. The Smart Router retries then falls back to the next best model.</p>
          <select value={throttleModel} onChange={e => setThrottleModel(e.target.value)}
            className="w-full bg-gray-900/80 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-300 focus:outline-none focus:border-orange-600">
            <option value="">Select a model...</option>
            {(options.region_models || []).map(m => (
              <option key={m.id} value={m.id}>{m.label} — {m.profile_id}</option>
            ))}
          </select>
        </div>
        <button onClick={runThrottleDemo} disabled={loading || !throttleModel || !prompt.trim()}
          className="bg-red-600 hover:bg-red-700 disabled:opacity-40 text-white text-xs font-medium rounded-lg px-4 py-2 flex items-center gap-2 transition-all">
          {loading ? <svg className="animate-spin h-3.5 w-3.5" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"/><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/></svg> : <span>🛡️</span>}
          {loading ? 'Simulating...' : 'Simulate Throttle & Run'}
        </button>
      </StepSection>

      {/* ═══ Step 4: Results ═══ */}
      <StepSection number={4} title="Results: Throttle Handling" visible={step4Visible} expanded={step4Expanded} onToggle={() => setStep4Expanded(!step4Expanded)}>
        <div className="flex gap-4 min-h-[400px]">
          {/* Baseline Panel */}
          <div className="flex-1 border border-red-900/30 rounded-lg overflow-hidden flex flex-col hover:border-red-500/50 hover:shadow-lg hover:-translate-y-1 transition-all duration-200">
            <div className="px-4 py-2.5 border-b border-red-900/30 bg-red-950/20">
              <div className="flex items-center gap-2">
                <span className="text-xs font-medium text-red-400">🧊 Baseline (Direct Call)</span>
                <span className="text-[10px] bg-red-900/40 text-red-300 px-1.5 py-0.5 rounded">{throttleModel ? (options.preferred_models?.find(m => m.id === throttleModel)?.label || throttleModel) : '—'}</span>
              </div>
            </div>
            <div className="flex-1 p-4 bg-[#0d0a0a]">
              {baselineTimeline.length === 0 && loading && (
                <div className="text-[11px] text-gray-500 animate-pulse">Attempting direct call...</div>
              )}
              {/* Timeline */}
              <div className="space-y-0">
                {baselineTimeline.map((entry, i) => (
                  <TimelineEntry key={i} {...entry} isLast={i === baselineTimeline.length - 1 && baselineFailed} />
                ))}
              </div>
              {baselineFailed && (
                <div className="mt-4 p-3 bg-red-950/30 border border-red-800/40 rounded-lg">
                  <div className="text-xs font-medium text-red-400 mb-1">❌ Request Failed (HTTP 429)</div>
                  <div className="text-[11px] text-red-300/80">ThrottlingException — no retry logic, no fallback. Your application is down.</div>
                  {baselineTimeline[0]?.error && <div className="text-[10px] text-red-400/60 font-mono mt-1.5">{baselineTimeline[0].error}</div>}
                  {baselineTimeline[0]?.request_id && <div className="text-[9px] text-gray-600 mt-1">RequestId: {baselineTimeline[0].request_id}</div>}
                </div>
              )}
            </div>
          </div>

          {/* Smart Router Panel */}
          <div className="flex-1 border border-green-900/30 rounded-lg overflow-hidden flex flex-col hover:border-green-500/50 hover:shadow-lg hover:-translate-y-1 transition-all duration-200">
            <div className="px-4 py-2.5 border-b border-green-900/30 bg-green-950/10">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-medium text-green-400">⚡ Smart Router (Retry + Fallback)</span>
                  {routerResult?.complexity_detected && <span className="text-[10px] bg-purple-900/40 text-purple-300 px-1.5 py-0.5 rounded">{routerResult.complexity_detected}</span>}
                  {routerResult?.strategy_used && <span className="text-[10px] bg-orange-900/40 text-orange-300 px-1.5 py-0.5 rounded">preferred: {options.region_models?.find(m => m.id === throttleModel)?.label || throttleModel}</span>}
                </div>
                <div className="flex items-center gap-2">
                  {routerResult?.routing_overhead_ms != null && <span className="text-[10px] text-gray-500">Routing Overhead: {routerResult.routing_overhead_ms}ms</span>}
                  {routerResult?.explanation && <button onClick={() => setExplainPopup({...routerResult.explanation, hide_reason: true})} className="text-[10px] text-green-400 hover:text-green-300 bg-green-900/20 hover:bg-green-900/40 px-1.5 py-0.5 rounded transition-all">ⓘ Explain</button>}
                </div>
              </div>
            </div>
            <div className="flex-1 p-4 bg-[#0a0d0a] overflow-y-auto">
              {routerTimeline.length === 0 && loading && (
                <div className="text-[11px] text-gray-500 animate-pulse">Router processing...</div>
              )}
              {/* Metrics (on top, like use-case 1) */}
              {routerResult && !routerResult.error && (
                <div className="mb-4 pb-3 border-b border-green-900/30 grid grid-cols-4 gap-3 text-center">
                  <div><div className="text-[9px] text-gray-500">Latency</div><div className="text-xs font-mono text-gray-300">{routerResult.latency_ms}ms</div></div>
                  <div><div className="text-[9px] text-gray-500">Tokens</div><div className="text-xs font-mono text-gray-300">{routerResult.input_tokens}↓ {routerResult.output_tokens}↑</div></div>
                  <div><div className="text-[9px] text-gray-500">Cost</div><div className="text-xs font-mono text-gray-300">${routerResult.cost}</div></div>
                  <div><div className="text-[9px] text-gray-500">Fallback</div><div className="text-xs font-mono text-green-400">✓ Used</div></div>
                </div>
              )}
              {/* Timeline */}
              <div className="space-y-0 mb-4">
                {routerTimeline.map((entry, i) => (
                  <TimelineEntry key={i} {...entry} isLast={i === routerTimeline.length - 1} />
                ))}
              </div>
              {/* Response */}
              {routerText && (
                <div className="border-t border-green-900/30 pt-3">
                  <div className="text-[10px] text-green-400/60 uppercase font-bold mb-2">Response from fallback model</div>
                  <div className="text-sm text-gray-300">
                    <ResponseWithThinking text={routerText} variant="router" streaming={loading} />
                  </div>
                </div>
              )}
              {/* Error */}
              {routerResult?.error && (
                <div className="mt-3 p-3 bg-red-950/30 border border-red-800/40 rounded-lg">
                  <div className="text-xs font-medium text-red-400 mb-1">❌ Router Error</div>
                  <div className="text-[11px] text-red-300/80 font-mono break-all">{routerResult.error}</div>
                </div>
              )}
            </div>
          </div>
        </div>
      </StepSection>
    </div>
  )
}
