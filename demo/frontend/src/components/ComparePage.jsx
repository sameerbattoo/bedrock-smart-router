import { useState, useEffect, useRef } from 'react'
import { Md, MetricWithDelta, ExplainPopup, AccuracyPopup, STREAM_API, API } from './shared'
import AnalyticsPanel from './AnalyticsPanel'

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

export default function ComparePage({ history, setHistory, restoreState, onRun }) {
  const round = (n, d) => Math.round(n * 10**d) / 10**d

  const [templates, setTemplates] = useState([])
  const [options, setOptions] = useState({ baseline_models: [], router_strategies: [], preferred_models: [] })
  const [prompt, setPrompt] = useState('')
  const [systemPrompt, setSystemPrompt] = useState('')
  const [file, setFile] = useState(null)
  const [baselineModel, setBaselineModel] = useState('sonnet')
  const [routerStrategy, setRouterStrategy] = useState('balanced')
  const [preferredModel, setPreferredModel] = useState('')
  const [preferredSearch, setPreferredSearch] = useState('')
  const [showPreferredDropdown, setShowPreferredDropdown] = useState(false)
  const [difficulty, setDifficulty] = useState('simple')
  const [loading, setLoading] = useState(false)
  const [baselineText, setBaselineText] = useState('')
  const [routerText, setRouterText] = useState('')
  const [baselineMetrics, setBaselineMetrics] = useState(null)
  const [routerMetrics, setRouterMetrics] = useState(null)
  const [judgeScores, setJudgeScores] = useState(null)
  const [judgePending, setJudgePending] = useState(false)
  const [popup, setPopup] = useState(null)
  const [explainPopup, setExplainPopup] = useState(null)
  const [historyFileId, setHistoryFileId] = useState('')
  const [historyFileName, setHistoryFileName] = useState('')
  const [historyFileExpired, setHistoryFileExpired] = useState(false)
  const fileRef = useRef(null)
  const runIdRef = useRef(0)

  // Step visibility & expansion
  const [step2Visible, setStep2Visible] = useState(false)
  const [step3Visible, setStep3Visible] = useState(false)
  const [step1Expanded, setStep1Expanded] = useState(true)
  const [step2Expanded, setStep2Expanded] = useState(true)
  const [step3Expanded, setStep3Expanded] = useState(true)

  useEffect(() => {
    fetch(`${API}/templates`).then(r => r.json()).then(setTemplates).catch(() => {})
    fetch(`${API}/options`).then(r => r.json()).then(setOptions).catch(() => {})
  }, [])

  // Restore state from history navigation
  useEffect(() => {
    if (restoreState) {
      const h = restoreState
      setPrompt(h.prompt || '')
      setSystemPrompt(h.system_prompt || '')
      setHistoryFileId(h.file_id || '')
      setHistoryFileName(h.file || '')
      setHistoryFileExpired(false)
      if (h.file_id) {
        setFile(null)
        // Check if file still exists on backend
        fetch(`${API}/check-file?file_id=${h.file_id}`).then(r => r.json())
          .then(data => { if (!data.exists) setHistoryFileExpired(true) })
          .catch(() => setHistoryFileExpired(true))
      }
      if (h.baseline_model) setBaselineModel(h.baseline_model)
      if (h.router_strategy) setRouterStrategy(h.router_strategy)
      setPreferredModel(h.preferred_model || '')
      setStep2Visible(true)
      setStep2Expanded(false)
      setStep1Expanded(false)
      if (h.baseline_metrics || h.router_metrics) {
        if (h.baseline_metrics) { setBaselineMetrics(h.baseline_metrics); setBaselineText(h.baseline_metrics.response_text || '') }
        if (h.router_metrics) { setRouterMetrics(h.router_metrics); setRouterText(h.router_metrics.response_text || '') }
        if (h.judge_scores) { setJudgeScores(h.judge_scores); setJudgePending(false) }
        else { setJudgeScores(null); setJudgePending(false) }
        setStep3Visible(true)
        setStep3Expanded(true)
      }
    }
  }, [restoreState])

  const filtered = templates.filter(t => t.difficulty === difficulty)

  function selectTemplate(t) {
    setPrompt(t.prompt)
    setSystemPrompt(t.system_prompt || '')
    setStep2Visible(true)
    setStep2Expanded(true)
    setStep1Expanded(false)
    // Clear previous results
    setStep3Visible(false)
    setBaselineText('')
    setRouterText('')
    setBaselineMetrics(null)
    setRouterMetrics(null)
    setJudgeScores(null)
  }

  function handleManualEntry() {
    setPrompt('')
    setSystemPrompt('')
    setStep2Visible(true)
    setStep2Expanded(true)
    setStep1Expanded(false)
    setStep3Visible(false)
    setBaselineText('')
    setRouterText('')
    setBaselineMetrics(null)
    setRouterMetrics(null)
    setJudgeScores(null)
  }

  async function run() {
    if (!prompt.trim() || loading) return
    if (onRun) onRun()
    setLoading(true)
    setBaselineText('')
    setRouterText('')
    setBaselineMetrics(null)
    setRouterMetrics(null)
    setJudgeScores(null)
    setJudgePending(true)
    setStep3Visible(true)
    setStep3Expanded(true)
    setStep2Expanded(false)
    setStep1Expanded(false)
    runIdRef.current += 1
    const currentRunId = runIdRef.current
    let currentFileId = ''

    const form = new FormData()
    form.append('prompt', prompt.trim())
    form.append('system_prompt', systemPrompt.trim())
    form.append('run_judge', 'true')
    form.append('selected_tools', '[]')
    form.append('baseline_model', baselineModel)
    form.append('router_strategy', preferredModel ? 'balanced' : routerStrategy)
    form.append('preferred_model', preferredModel || '')
    if (file) form.append('file', file)
    if (!file && historyFileId) { form.append('file_id', historyFileId); currentFileId = historyFileId }

    try {
      function addHistoryEntry(bl, rt) {
        const savingsPct = bl.cost > 0 ? round((1 - rt.cost / bl.cost) * 100, 1) : 0
        setHistory(prev => [...prev, {
          id: Date.now(), timestamp: Date.now(), use_case: 'compare',
          prompt: prompt.trim(), system_prompt: systemPrompt.trim(),
          file: file?.name, file_id: currentFileId,
          baseline_model: baselineModel, router_strategy: routerStrategy, preferred_model: preferredModel,
          baseline_metrics: bl, router_metrics: rt, judge_scores: null,
          baseline_latency: bl.latency_ms, router_latency: rt.latency_ms,
          baseline_ttft: bl.ttft_ms, router_ttft: rt.ttft_ms,
          baseline_cost: bl.cost, router_cost: rt.cost,
          baseline_tokens: (bl.input_tokens||0)+(bl.output_tokens||0),
          router_tokens: (rt.input_tokens||0)+(rt.output_tokens||0),
          savings_pct: savingsPct, router_model: rt.model_used, complexity: rt.complexity_detected,
        }])
      }

      const res = await fetch(`${STREAM_API}/compare-stream`, { method: 'POST', body: form })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = '', baselineData = null, routerData = null, currentEventType = null

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (currentRunId !== runIdRef.current) break
          if (line.startsWith('event: ')) { currentEventType = line.slice(7).trim() }
          else if (line.startsWith('data: ') && currentEventType) {
            const eventType = currentEventType; currentEventType = null
            const parsed = JSON.parse(line.slice(6))
            if (eventType === 'baseline_chunk') setBaselineText(prev => prev + parsed.text)
            else if (eventType === 'router_chunk') setRouterText(prev => prev + parsed.text)
            else if (eventType === 'file_stored') { currentFileId = parsed.file_id }
            else if (eventType === 'baseline_complete') {
              baselineData = parsed; setBaselineMetrics(parsed)
              if (routerData) { addHistoryEntry(baselineData, routerData); setLoading(false) }
            } else if (eventType === 'router_complete') {
              routerData = parsed; setRouterMetrics(parsed)
              if (baselineData) { addHistoryEntry(baselineData, routerData); setLoading(false) }
            } else if (eventType === 'judge_scores') {
              setJudgeScores(parsed); setJudgePending(false)
              setHistory(prev => { if (!prev.length) return prev; return [...prev.slice(0,-1), {...prev[prev.length-1], baseline_score: parsed.baseline_score, router_score: parsed.router_score, baseline_reasoning: parsed.baseline_reasoning, router_reasoning: parsed.router_reasoning, judge_scores: parsed}] })
            } else if (eventType === 'done') { setJudgePending(false); setLoading(false) }
          }
        }
      }
    } catch (err) { alert(`Error: ${err.message}`); setLoading(false); setJudgePending(false) }
  }

  const hasResult = baselineText || routerText || baselineMetrics || routerMetrics

  return (
    <div className="flex-1 flex min-w-0 overflow-hidden">
      {popup && <AccuracyPopup {...popup} onClose={() => setPopup(null)} />}
      {explainPopup && <ExplainPopup explanation={explainPopup} onClose={() => setExplainPopup(null)} />}

      {/* Main scrollable area with steps */}
      <div className="flex-1 overflow-y-auto p-4 space-y-3">

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
                  className={`p-3 rounded-lg border text-left transition-all hover:shadow-md ${prompt === t.prompt ? 'border-orange-500/60 bg-orange-950/20 shadow-sm shadow-orange-900/20' : 'border-gray-800/50 bg-gray-900/30 hover:border-gray-700 hover:bg-gray-900/50'}`}>
                  <div className="text-[11px] text-gray-300 line-clamp-3 leading-relaxed">{t.prompt}</div>
                </button>
              ))}
            </div>
          )}
        </StepSection>

        {/* ═══ Step 2: System & User Prompt ═══ */}
        <StepSection number={2} title="System & User Prompt" visible={step2Visible} expanded={step2Expanded} onToggle={() => setStep2Expanded(!step2Expanded)}>
          {/* System Prompt */}
          <div className="mb-3">
            <div className="text-[10px] text-gray-500 uppercase font-bold mb-1.5 flex items-center gap-1.5">
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.066 2.573c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.573 1.066c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.066-2.573c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"/><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"/></svg>
              System Prompt
            </div>
            <textarea value={systemPrompt} onChange={e => setSystemPrompt(e.target.value)}
              placeholder="You are a helpful assistant..."
              rows={2} className="w-full bg-gray-900/40 border border-gray-800/50 rounded-lg px-3 py-2 text-sm text-gray-300 placeholder-gray-600 focus:outline-none focus:border-orange-600/40 resize-none" />
          </div>
          {/* User Prompt */}
          <div className="mb-3">
            <div className="text-[10px] text-gray-500 uppercase font-bold mb-1.5 flex items-center gap-1.5">
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"/></svg>
              User Prompt
            </div>
            <div className="relative">
              <textarea value={prompt} onChange={e => setPrompt(e.target.value)}
                onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); run() }}}
                placeholder="Ask anything..." rows={3}
                className="w-full bg-gray-900/60 border border-gray-700/50 rounded-xl px-4 py-3 text-sm text-gray-200 placeholder-gray-500 focus:outline-none focus:border-orange-500/50 focus:ring-1 focus:ring-orange-500/20 resize-none" />
              <div className="absolute right-3 bottom-2.5 flex items-center gap-2">
                {file && <span className="text-[9px] text-gray-400 bg-gray-800 px-1.5 py-0.5 rounded">📎{file.name}<button onClick={()=>{setFile(null);setHistoryFileId('');setHistoryFileName('')}} className="text-red-400 ml-1">&times;</button></span>}
                {!file && historyFileId && (
                  historyFileExpired
                    ? <span className="text-[9px] text-red-300 bg-red-900/40 border border-red-700/50 px-1.5 py-0.5 rounded">📎{historyFileName} <span className="text-red-400">(expired — re-upload)</span><button onClick={()=>{setHistoryFileId('');setHistoryFileName('');setHistoryFileExpired(false)}} className="text-red-400 ml-1">&times;</button></span>
                    : <span className="text-[9px] text-gray-400 bg-gray-800 px-1.5 py-0.5 rounded">📎{historyFileName || '(from history)'}<button onClick={()=>{setHistoryFileId('');setHistoryFileName('')}} className="text-red-400 ml-1">&times;</button></span>
                )}
                <button onClick={() => fileRef.current?.click()} className="text-gray-600 hover:text-gray-400 p-1" title="Attach file">
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13" /></svg>
                </button>
                <button onClick={run} disabled={loading || !prompt.trim()}
                  className="bg-orange-600 hover:bg-orange-700 disabled:opacity-40 text-white text-xs font-medium rounded-lg px-4 py-1.5 flex items-center gap-1.5 transition-all shadow-lg shadow-orange-900/20">
                  {loading ? <svg className="animate-spin h-3.5 w-3.5" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"/><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/></svg> : null}
                  {loading ? 'Running...' : 'Run'}
                </button>
              </div>
              <input ref={fileRef} type="file" accept=".pdf,.png,.jpg,.jpeg,.gif,.webp" onChange={e => { setFile(e.target.files?.[0]); setHistoryFileId('') }} className="hidden" />
            </div>
          </div>
        </StepSection>

        {/* ═══ Step 3: Response ═══ */}
        <StepSection number={3} title="Response Comparison" visible={step3Visible} expanded={step3Expanded} onToggle={() => setStep3Expanded(!step3Expanded)}>
          {/* Baseline & Strategy controls */}
          <div className="flex gap-3 mb-4 pb-3 border-b border-gray-800/40">
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
            <div className="flex-1 flex items-center gap-2">
              <span className="text-[10px] text-gray-500 font-bold uppercase">Strategy</span>
              <div className="flex bg-gray-900/80 rounded-lg p-0.5 border border-gray-800/50">
                {options.router_strategies.map(s => (
                  <button key={s} onClick={() => { setRouterStrategy(s); setPreferredModel('') }}
                    className={`text-[11px] px-2.5 py-1 rounded-md font-medium transition-all capitalize ${routerStrategy === s && !preferredModel ? 'bg-orange-600 text-white shadow-sm' : 'text-gray-500 hover:text-gray-300'}`}>
                    {s.replace('-optimized','').replace('-',' ')}
                  </button>
                ))}
                <button onClick={() => { setPreferredModel(preferredModel || 'us.anthropic.claude-sonnet-4-6'); setShowPreferredDropdown(true) }}
                  className={`text-[11px] px-2.5 py-1 rounded-md font-medium transition-all ${preferredModel ? 'bg-green-600 text-white shadow-sm' : 'text-gray-500 hover:text-gray-300'}`}>
                  Preferred
                </button>
              </div>
              {preferredModel && (
                <div className="relative">
                  <input value={preferredSearch} onChange={e => { setPreferredSearch(e.target.value); setShowPreferredDropdown(true) }}
                    onFocus={() => setShowPreferredDropdown(true)}
                    placeholder={options.preferred_models?.find(m => m.id === preferredModel)?.label || 'Sonnet 4.6'}
                    className="w-36 bg-gray-900/80 border border-gray-700 rounded-md px-2 py-1 text-[11px] text-gray-300 placeholder-gray-500 focus:outline-none focus:border-green-600" />
                  {showPreferredDropdown && (
                    <div className="absolute top-full left-0 mt-1 w-52 bg-gray-900 border border-gray-700 rounded-lg shadow-xl z-50 max-h-48 overflow-y-auto">
                      {(options.preferred_models || []).filter(m => m.id && m.label.toLowerCase().includes(preferredSearch.toLowerCase())).map(m => (
                        <button key={m.id} onClick={() => { setPreferredModel(m.id); setPreferredSearch(''); setShowPreferredDropdown(false) }}
                          className={`w-full text-left px-3 py-1.5 text-[11px] hover:bg-gray-800 ${preferredModel === m.id ? 'text-green-400 bg-green-950/30' : 'text-gray-400'}`}>
                          {m.label}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              )}
              {/* Re-run button */}
              <button onClick={run} disabled={loading || !prompt.trim()}
                className="ml-auto bg-orange-600 hover:bg-orange-700 disabled:opacity-40 text-white text-[11px] font-medium rounded-lg px-3 py-1.5 flex items-center gap-1.5 transition-all">
                {loading ? <svg className="animate-spin h-3 w-3" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"/><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/></svg> : <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"/></svg>}
                Re-run
              </button>
            </div>
          </div>

          {/* Split panel responses */}
          {hasResult ? (
            <div className="flex gap-3 min-h-[400px]">
              {/* Baseline */}
              <div className="flex-1 border border-blue-900/30 rounded-lg overflow-hidden flex flex-col hover:border-blue-500/50 hover:shadow-lg hover:shadow-blue-900/20 hover:-translate-y-1.5 transition-all duration-200">
                <div className="px-4 py-2 border-b border-blue-900/30 bg-blue-950/20">
                  <div className="flex items-center gap-2 mb-1.5">
                    <span className="text-xs font-medium text-blue-400">Baseline</span>
                    <span className="text-[10px] bg-blue-900/40 text-blue-300 px-1.5 py-0.5 rounded">{baselineMetrics?.model_used}</span>
                  </div>
                  {baselineMetrics ? (
                    <div className="grid grid-cols-5 gap-2">
                      <div className="text-center"><div className="text-[9px] text-gray-500">TTFT</div><div className="text-xs font-mono text-gray-300">{baselineMetrics.ttft_ms}ms</div></div>
                      <div className="text-center"><div className="text-[9px] text-gray-500">Latency</div><div className="text-xs font-mono text-gray-300">{baselineMetrics.latency_ms}ms</div></div>
                      <div className="text-center"><div className="text-[9px] text-gray-500">Tokens</div><div className="text-xs font-mono text-gray-300">{baselineMetrics.input_tokens}↓ {baselineMetrics.output_tokens}↑</div></div>
                      <div className="text-center"><div className="text-[9px] text-gray-500">Cost</div><div className="text-xs font-mono text-gray-300">${baselineMetrics.cost}</div></div>
                      <div className="text-center cursor-pointer" onClick={e => judgeScores && setPopup({side:'baseline',score:judgeScores.baseline_score,reasoning:judgeScores.baseline_reasoning,position:{x:e.clientX,y:e.clientY}})}>
                        <div className="text-[9px] text-gray-500">Accuracy</div>
                        <div className="text-xs font-mono text-blue-300 underline decoration-dotted">{judgeScores ? `${judgeScores.baseline_score}/10` : judgePending ? <span className="animate-pulse text-gray-600">...</span> : '—'}</div>
                      </div>
                    </div>
                  ) : <div className="text-[10px] text-gray-600 animate-pulse">Waiting...</div>}
                </div>
                <div className="flex-1 overflow-y-auto p-4 text-sm text-gray-300 bg-[#080d18]">
                  {baselineText ? <Md variant="baseline">{baselineText}</Md> : loading ? <div className="animate-pulse text-gray-600">Generating...</div> : null}
                </div>
              </div>
              {/* Router */}
              <div className="flex-1 border border-orange-900/30 rounded-lg overflow-hidden flex flex-col hover:border-orange-500/50 hover:shadow-lg hover:shadow-orange-900/20 hover:-translate-y-1.5 transition-all duration-200">
                <div className="px-4 py-2 border-b border-orange-900/30 bg-orange-950/20">
                  <div className="flex items-center justify-between mb-1.5">
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-medium text-orange-400">Smart Router</span>
                      <span className="text-[10px] bg-orange-900/40 text-orange-300 px-1.5 py-0.5 rounded cursor-help" title={routerMetrics?.model_id_full || ''}>{routerMetrics?.model_used}</span>
                      {routerMetrics?.complexity_detected && <span className="text-[10px] bg-purple-900/40 text-purple-300 px-1.5 py-0.5 rounded">{routerMetrics.complexity_detected}</span>}
                    </div>
                    <div className="flex items-center gap-2">
                      {routerMetrics?.routing_overhead_ms != null && <span className="text-[10px] text-gray-500">Routing Overhead: {routerMetrics.routing_overhead_ms}ms</span>}
                      {routerMetrics?.explanation && <button onClick={() => setExplainPopup(routerMetrics.explanation)} className="text-[10px] text-orange-400 hover:text-orange-300 bg-orange-900/20 hover:bg-orange-900/40 px-1.5 py-0.5 rounded transition-all">ⓘ Explain</button>}
                    </div>
                  </div>
                  {routerMetrics ? (
                    <div className="grid grid-cols-5 gap-2">
                      <MetricWithDelta label="TTFT" value={`${routerMetrics.ttft_ms}ms`} baseline={baselineMetrics?.ttft_ms} current={routerMetrics.ttft_ms} lower />
                      <MetricWithDelta label="Latency" value={`${routerMetrics.latency_ms}ms`} baseline={baselineMetrics?.latency_ms} current={routerMetrics.latency_ms} lower />
                      <MetricWithDelta label="Tokens" value={`${routerMetrics.input_tokens}↓ ${routerMetrics.output_tokens}↑`} baseline={baselineMetrics ? baselineMetrics.input_tokens+baselineMetrics.output_tokens : null} current={routerMetrics.input_tokens+routerMetrics.output_tokens} lower />
                      <MetricWithDelta label="Cost" value={`$${routerMetrics.cost}`} baseline={baselineMetrics?.cost} current={routerMetrics.cost} lower />
                      <div className="text-center cursor-pointer" onClick={e => judgeScores && setPopup({side:'router',score:judgeScores.router_score,reasoning:judgeScores.router_reasoning,position:{x:e.clientX,y:e.clientY}})}>
                        <div className="text-[9px] text-gray-500">Accuracy</div>
                        <div className="text-xs font-mono text-orange-300 underline decoration-dotted">{judgeScores ? `${judgeScores.router_score}/10` : judgePending ? <span className="animate-pulse text-gray-600">...</span> : '—'}</div>
                      </div>
                    </div>
                  ) : <div className="text-[10px] text-gray-600 animate-pulse">Waiting...</div>}
                </div>
                <div className="flex-1 overflow-y-auto p-4 text-sm text-gray-300 bg-[#0d1210]">
                  {routerText ? <Md variant="router">{routerText}</Md> : loading ? <div className="animate-pulse text-gray-600">Generating...</div> : null}
                </div>
              </div>
            </div>
          ) : (
            <div className="flex items-center justify-center min-h-[200px]">
              <div className="flex items-center gap-2 text-sm text-gray-500">
                <svg className="animate-spin h-4 w-4 text-orange-500" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"/><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/></svg>
                Running both models in parallel...
              </div>
            </div>
          )}
        </StepSection>
      </div>

      {/* Analytics Panel (right sidebar) */}
      <AnalyticsPanel history={history} />
    </div>
  )
}
