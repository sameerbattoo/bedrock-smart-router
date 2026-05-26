import { useState, useEffect, useRef } from 'react'
import { Md, MetricWithDelta, ExplainPopup, AccuracyPopup, ResponseWithThinking, STREAM_API, API } from './shared'
import AnalyticsPanel from './AnalyticsPanel'
import { useSpeechToText } from '../hooks/useSpeechToText'
import AudioWaveform from './AudioWaveform'

export default function StrandsPage({ history, setHistory, onRun, restoreState, prewarmedSessionIds, conversation, setConversation }) {
  const [options, setOptions] = useState({ baseline_models: [], router_strategies: [] })
  const [baselineModel, setBaselineModel] = useState('sonnet')
  const [routerStrategy, setRouterStrategy] = useState('quality-optimized')
  const [classifier, setClassifier] = useState('heuristic')
  const [preferredModel, setPreferredModel] = useState('')
  const [preferredSearch, setPreferredSearch] = useState('')
  const [showPreferredDropdown, setShowPreferredDropdown] = useState(false)
  const [tools, setTools] = useState({ docs: true, diagram: true })
  const [toolStatus, setToolStatus] = useState({ docs: 'stopped', diagram: 'stopped' }) // stopped | starting | ready
  const [message, setMessage] = useState('')
  const [sendTarget, setSendTarget] = useState('both') // 'both' | 'baseline' | 'router'
  const [loading, setLoading] = useState(false)
  const [baselineSessionId, setBaselineSessionId] = useState('')
  const [routerSessionId, setRouterSessionId] = useState('')
  const [explainPopup, setExplainPopup] = useState(null)
  const [accuracyPopup, setAccuracyPopup] = useState(null)
  const [showSystemPrompt, setShowSystemPrompt] = useState(false)
  const [systemPromptText, setSystemPromptText] = useState('')
  const [initializing, setInitializing] = useState(true)

  const chatEndRef = useRef(null)
  const chatContainerRef = useRef(null)
  const userScrolledRef = useRef(false)

  // Speech-to-text (Whisper tiny.en, runs in browser)
  const [pendingVoiceSubmit, setPendingVoiceSubmit] = useState(false)
  const {
    isListening,
    isLoading: isSpeechLoading,
    isModelLoading,
    modelProgress,
    transcript: speechTranscript,
    error: speechError,
    recordingDuration,
    startListening,
    stopListening,
    resetTranscript,
    isSupported: isSpeechSupported,
  } = useSpeechToText({
    model: 'Xenova/whisper-tiny.en',
    silenceThreshold: 0.01,
    silenceTimeout: 2000,
    onSilenceDetected: () => {
      stopListening()
      setPendingVoiceSubmit(true)
    },
  })

  // Auto-submit voice transcript when recording stops
  useEffect(() => {
    if (speechTranscript && pendingVoiceSubmit && !loading) {
      setPendingVoiceSubmit(false)
      const text = speechTranscript.trim()
      resetTranscript()
      if (text) {
        setMessage(text)
        // Trigger send after state update
        setTimeout(() => {
          document.getElementById('strands-send-btn')?.click()
        }, 50)
      }
    }
  }, [speechTranscript, pendingVoiceSubmit, loading])

  const handleMicClick = async () => {
    if (isListening) {
      stopListening()
      setPendingVoiceSubmit(true)
    } else {
      resetTranscript()
      setMessage('')
      setPendingVoiceSubmit(false)
      await startListening()
    }
  }

  const formatDuration = (seconds) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
  }

  useEffect(() => {
    fetch(`${API}/options`).then(r => r.json()).then(setOptions).catch(() => {})
    fetch(`${API}/strands-system-prompt`).then(r => r.json()).then(d => setSystemPromptText(d.system_prompt || '')).catch(() => {})
    // Use pre-warmed session IDs from App.jsx (agents already initialized on app load)
    if (prewarmedSessionIds?.baseline) {
      setBaselineSessionId(prewarmedSessionIds.baseline)
      setRouterSessionId(prewarmedSessionIds.router)
      setToolStatus({ docs: 'ready', diagram: 'ready' })
      setInitializing(false)
    } else {
      // Fallback: init on page mount if not pre-warmed
      setToolStatus({ docs: 'starting', diagram: 'starting' })
      _initAgents()
    }
  }, [])

  async function _initAgents() {
    const form = new FormData()
    form.append('message', 'Hi! Please introduce yourself briefly.')
    form.append('baseline_session_id', '')
    form.append('router_session_id', '')
    form.append('baseline_model', baselineModel)
    form.append('router_strategy', routerStrategy)
    form.append('classifier', classifier)
    form.append('skip_judge', 'true')
    try {
      const res = await fetch(`${STREAM_API}/strands-chat`, { method: 'POST', body: form })
      if (!res.ok) { setInitializing(false); return }
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
              setBaselineSessionId(parsed.baseline_session_id)
              setRouterSessionId(parsed.router_session_id)
            } else if (eventType === 'baseline_complete') {
              setConversation(prev => [...prev.filter(t => t.user !== '__init__'), { user: '__init__', timestamp: '', baseline: parsed, router: prev.find(t => t.user === '__init__')?.router || null, isWelcome: true }])
              setToolStatus(ts => ({ ...ts, docs: 'ready' }))
            } else if (eventType === 'router_complete') {
              setConversation(prev => [...prev.filter(t => t.user !== '__init__'), { user: '__init__', timestamp: '', baseline: prev.find(t => t.user === '__init__')?.baseline || null, router: parsed, isWelcome: true }])
              setToolStatus(ts => ({ ...ts, diagram: 'ready' }))
            } else if (eventType === 'done') {
              setInitializing(false)
              setToolStatus({ docs: 'ready', diagram: 'ready' })
            }
          }
        }
      }
    } catch { setInitializing(false) }
  }

  // Tool status is set by the init response (no polling needed)

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
    // Only auto-scroll if user hasn't manually scrolled up
    if (!userScrolledRef.current) {
      chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }
  }, [conversation])

  function resetConversation() {
    if (conversation.length > 0) {
      // Per-turn metrics are already stored individually in history.
    }
    // Clear conversation and baseline session (allows model switching)
    // Keep router session alive (MCP tools stay warm)
    setConversation([])
    if (baselineSessionId || routerSessionId) {
      const form = new FormData()
      form.append('baseline_session_id', baselineSessionId)
      form.append('router_session_id', routerSessionId)
      fetch(`${API}/strands-reset`, { method: 'POST', body: form }).catch(() => {})
    }
    setBaselineSessionId('')  // Allows baseline model switching
  }

  async function sendMessage() {
    if (!message.trim() || loading) return
    if (onRun) onRun()
    userScrolledRef.current = false  // Reset scroll lock on new message
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
    form.append('router_strategy', preferredModel ? 'balanced' : routerStrategy)
    form.append('classifier', classifier)
    form.append('preferred_model', preferredModel || '')
    form.append('send_target', sendTarget)

    try {
      const res = await fetch(`${STREAM_API}/strands-chat`, { method: 'POST', body: form })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = '', currentEventType = null
      let baselineResult = null, routerResult = null
      let judgeScoresData = null

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
            } else if (eventType === 'baseline_token') {
              setConversation(prev => prev.map((turn, i) => i === turnIdx ? { ...turn, baseline: { ...turn.baseline, response_text: (turn.baseline?.response_text || '') + parsed.text, _streaming: true } } : turn))
            } else if (eventType === 'baseline_progress') {
              setConversation(prev => prev.map((turn, i) => i === turnIdx ? { ...turn, baseline: { ...turn.baseline, _progress: parsed.message, _tools_used: [...new Set([...(turn.baseline?._tools_used || []), parsed.message.replace('🔧 Using tool: ', '')])] } } : turn))
            } else if (eventType === 'baseline_complete') {
              setConversation(prev => prev.map((turn, i) => i === turnIdx ? { ...turn, baseline: { ...parsed, _tools_used: turn.baseline?._tools_used || [] } } : turn))
              setToolStatus(ts => ({ ...ts, docs: 'ready' }))
              baselineResult = parsed
            } else if (eventType === 'baseline_error') {
              setConversation(prev => prev.map((turn, i) => i === turnIdx ? { ...turn, baseline: { error: parsed.error, latency_ms: parsed.latency_ms } } : turn))
            } else if (eventType === 'router_token') {
              setConversation(prev => prev.map((turn, i) => i === turnIdx ? { ...turn, router: { ...turn.router, response_text: (turn.router?.response_text || '') + parsed.text, _streaming: true } } : turn))
            } else if (eventType === 'router_progress') {
              setConversation(prev => prev.map((turn, i) => i === turnIdx ? { ...turn, router: { ...turn.router, _progress: parsed.message, _tools_used: [...new Set([...(turn.router?._tools_used || []), parsed.message.replace('🔧 Using tool: ', '')])] } } : turn))
            } else if (eventType === 'router_complete') {
              setConversation(prev => prev.map((turn, i) => i === turnIdx ? { ...turn, router: { ...parsed, _tools_used: turn.router?._tools_used || [] } } : turn))
              routerResult = parsed
            } else if (eventType === 'router_error') {
              setConversation(prev => prev.map((turn, i) => i === turnIdx ? { ...turn, router: { error: parsed.error, latency_ms: parsed.latency_ms } } : turn))
            } else if (eventType === 'judge_scores') {
              judgeScoresData = parsed
              setConversation(prev => prev.map((turn, i) => i === turnIdx ? {
                ...turn,
                baseline: turn.baseline ? { ...turn.baseline, accuracy_score: parsed.baseline_score, accuracy_reasoning: parsed.baseline_reasoning } : turn.baseline,
                router: turn.router ? { ...turn.router, accuracy_score: parsed.router_score, accuracy_reasoning: parsed.router_reasoning } : turn.router,
              } : turn))
              // Update the last history entry with judge scores (only if sent to both)
              if (sendTarget === 'both') {
                setHistory(prev => { if (!prev.length) return prev; return [...prev.slice(0,-1), {...prev[prev.length-1], baseline_score: parsed.baseline_score, router_score: parsed.router_score}] })
              }
            } else if (eventType === 'done') {
              // Only save to history when sent to both agents (meaningful comparison)
              if (sendTarget === 'both' && routerResult) {
                const blCost = baselineResult?.cost || 0
                const rtCost = routerResult.cost || 0
                const savings = blCost > 0 ? Math.round((1 - rtCost / blCost) * 1000) / 10 : 0
                setHistory(prev => [...prev, {
                  id: Date.now(), timestamp: Date.now(), use_case: 'strands',
                  prompt: userMsg, router_model: routerResult.model_used,
                  complexity: routerResult.complexity_detected,
                  router_latency: routerResult.latency_ms, router_cost: rtCost,
                  baseline_latency: baselineResult?.latency_ms || 0,
                  baseline_cost: blCost,
                  savings_pct: savings,
                  baseline_score: null,
                  router_score: null,
                  has_error: false,
                }])
              }
              setLoading(false)
            }
          }
        }
      }
    } catch (err) { alert(`Error: ${err.message}`); setLoading(false) }
  }

  return (
    <div className="flex-1 flex min-w-0 overflow-hidden">
      {explainPopup && <ExplainPopup explanation={explainPopup} onClose={() => setExplainPopup(null)} />}
      {accuracyPopup && <AccuracyPopup {...accuracyPopup} onClose={() => setAccuracyPopup(null)} />}
      <div className="flex-1 flex flex-col min-w-0">
        {/* ─── Agent Header ─── */}
        <div className="px-4 py-2 border-b border-gray-800/30 bg-gradient-to-r from-orange-950/20 to-purple-950/20">
          <div className="flex items-center gap-2 cursor-pointer" onClick={() => setShowSystemPrompt(!showSystemPrompt)}>
            <span className="text-lg">☁️</span>
            <span className="text-sm font-bold text-gray-200">AWS Buddy</span>
            <span className="text-[10px] text-gray-500 bg-gray-800 px-2 py-0.5 rounded-full">AWS Tech Assistant</span>
            <svg className={`w-3 h-3 text-gray-500 transition-transform ml-1 ${showSystemPrompt ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7"/></svg>
            <span className="text-[9px] text-gray-600 ml-auto">Powered by Strands Agents + MCP</span>
          </div>
          {showSystemPrompt && (
            <div className="mt-2 p-3 bg-gray-900/60 border border-gray-800/50 rounded-lg text-[11px] text-gray-400 leading-relaxed max-h-40 overflow-y-auto font-mono whitespace-pre-wrap">
              {systemPromptText || 'Loading...'}
            </div>
          )}
        </div>

        {/* ─── Control Panel (top) ─── */}
        <div className="px-4 py-3 border-b border-gray-800/50 bg-gray-900/40">
          <div className="flex gap-3 flex-wrap">
            {/* Baseline model (left half) */}
            <div className="flex-1 flex items-center gap-2">
              <span className="text-[10px] text-gray-500 font-bold uppercase">Baseline</span>
              <div className="flex bg-gray-900/80 rounded-lg p-0.5 border border-gray-800/50">
                {options.baseline_models.map(m => (
                  <button key={m.id} onClick={() => !baselineSessionId && setBaselineModel(m.id)}
                    disabled={!!baselineSessionId}
                    className={`text-[11px] px-2.5 py-1 rounded-md font-medium transition-all ${baselineModel === m.id ? 'bg-blue-600 text-white shadow-sm' : 'text-gray-500 hover:text-gray-300'} ${baselineSessionId ? 'opacity-50 cursor-not-allowed' : ''}`}>
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
                    onBlur={() => setTimeout(() => setShowPreferredDropdown(false), 200)}
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
              {/* Classifier toggle */}
              <div className="flex items-center gap-1.5 ml-2">
                <span className="text-[9px] text-gray-500 font-bold uppercase">Classifier</span>
                <div className="flex bg-gray-900/80 rounded-lg p-0.5 border border-gray-800/50">
                  <button onClick={() => setClassifier('heuristic')}
                    className={`text-[10px] px-2 py-0.5 rounded-md font-medium transition-all ${classifier === 'heuristic' ? 'bg-purple-600 text-white shadow-sm' : 'text-gray-500 hover:text-gray-300'}`}>
                    Heuristic
                  </button>
                  <button onClick={() => setClassifier('ml')}
                    className={`text-[10px] px-2 py-0.5 rounded-md font-medium transition-all ${classifier === 'ml' ? 'bg-purple-600 text-white shadow-sm' : 'text-gray-500 hover:text-gray-300'}`}>
                    ML
                  </button>
                </div>
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
        <div ref={chatContainerRef} className="flex-1 overflow-y-auto px-4 py-4 space-y-4"
          onScroll={e => {
            const el = e.currentTarget
            const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 80
            userScrolledRef.current = !atBottom
          }}>
          {conversation.length === 0 && (
            <div className="text-center text-gray-600 mt-20">
              <div className="text-4xl mb-3">🤖</div>
              <div className="text-sm font-medium text-gray-400">AWS Buddy is ready</div>
              <div className="text-xs text-gray-600 mt-1">Ask anything about AWS — architecture, best practices, cost optimization</div>
              <div className="text-[10px] text-gray-700 mt-3">Both agents have access to AWS Documentation and Diagram tools</div>
              {initializing && <div className="text-[10px] text-orange-400 mt-2 animate-pulse">⏳ Warming up MCP tools...</div>}
            </div>
          )}

          {conversation.map((turn, i) => (
            <div key={i} className="space-y-3 animate-slideDown">
              {/* User message (skip for welcome/init messages) */}
              {!turn.isWelcome && (
                <div className="flex justify-center">
                  <div className="bg-gray-800/60 border border-gray-700/50 rounded-xl px-4 py-2 max-w-[70%]">
                    <div className="flex items-center gap-2">
                      <span className="text-[9px] text-gray-500">You</span>
                      <span className="text-[9px] text-gray-600">{turn.timestamp}</span>
                    </div>
                    <div className="text-sm text-gray-200">{turn.user}</div>
                  </div>
                </div>
              )}

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
                    {turn.baseline && !turn.baseline.error && !turn.isWelcome && !turn.baseline._streaming && (
                      <div className="grid grid-cols-5 gap-2">
                        <div className="text-center"><div className="text-[9px] text-gray-500">TTFT</div><div className="text-xs font-mono text-gray-300">{turn.baseline.ttft_ms}ms</div></div>
                        <div className="text-center"><div className="text-[9px] text-gray-500">Latency</div><div className="text-xs font-mono text-gray-300">{turn.baseline.latency_ms}ms</div></div>
                        <div className="text-center"><div className="text-[9px] text-gray-500">Tokens</div><div className="text-xs font-mono text-gray-300">{turn.baseline.input_tokens || '—'}↓ {turn.baseline.output_tokens || '—'}↑</div></div>
                        <div className="text-center"><div className="text-[9px] text-gray-500">Cost</div><div className="text-xs font-mono text-gray-300">{turn.baseline.cost != null ? `$${turn.baseline.cost}` : '—'}</div></div>
                        <div className="text-center cursor-pointer" onClick={e => turn.baseline?.accuracy_score != null && setAccuracyPopup({side:'baseline',score:turn.baseline.accuracy_score,reasoning:turn.baseline.accuracy_reasoning,position:{x:e.clientX,y:e.clientY}})}>
                          <div className="text-[9px] text-gray-500">Accuracy</div>
                          <div className="text-xs font-mono text-blue-300 underline decoration-dotted">{turn.baseline.accuracy_score != null ? `${turn.baseline.accuracy_score}/10` : <span className="animate-pulse text-gray-600">...</span>}</div>
                        </div>
                      </div>
                    )}
                  </div>
                  <div className="p-3 text-sm text-gray-300 bg-[#080d18] min-h-[60px]">
                    {turn.baseline === null ? (
                      <div className="animate-pulse text-gray-600 text-xs">Generating...</div>
                    ) : turn.baseline.error ? (
                      <div className="text-red-400 text-xs">❌ {turn.baseline.error}</div>
                    ) : turn.baseline._streaming ? (
                      <div>
                        {turn.baseline._progress && <div className="text-[10px] text-yellow-400 mb-2 animate-pulse">{turn.baseline._progress}</div>}
                        <ResponseWithThinking text={turn.baseline.response_text} variant="baseline" streaming={true} />
                      </div>
                    ) : (
                      <div>
                        {turn.baseline._tools_used?.length > 0 && (
                          <div className="flex gap-1 mb-2 pb-2 border-b border-blue-900/20">
                            {turn.baseline._tools_used.map((t, i) => <span key={i} className="text-[9px] bg-blue-900/30 text-blue-300 px-1.5 py-0.5 rounded">🔧 {t}</span>)}
                          </div>
                        )}
                        <ResponseWithThinking text={turn.baseline.response_text} variant="baseline" streaming={false} />
                      </div>
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
                      <div className="flex items-center gap-2">
                        {turn.router?.routing_overhead_ms != null && <span className="text-[9px] text-gray-500">Overhead: {turn.router.routing_overhead_ms}ms</span>}
                        {turn.router?.explanation && <button onClick={() => setExplainPopup({...turn.router.explanation, _fallback_used: turn.router.fallback_used, _actual_model: turn.router.model_used})} className="text-[10px] text-orange-400 hover:text-orange-300 bg-orange-900/20 hover:bg-orange-900/40 px-1.5 py-0.5 rounded transition-all">ⓘ Explain</button>}
                      </div>
                    </div>
                    {turn.router && !turn.router.error && !turn.isWelcome && !turn.router._streaming && (
                      <div className="grid grid-cols-5 gap-2">
                        <MetricWithDelta label="TTFT" value={`${turn.router.ttft_ms}ms`} baseline={turn.baseline?.ttft_ms} current={turn.router.ttft_ms} lower />
                        <MetricWithDelta label="Latency" value={`${turn.router.latency_ms}ms`} baseline={turn.baseline?.latency_ms} current={turn.router.latency_ms} lower />
                        <MetricWithDelta label="Tokens" value={`${turn.router.input_tokens || 0}↓ ${turn.router.output_tokens || 0}↑`} baseline={turn.baseline ? (turn.baseline.input_tokens||0)+(turn.baseline.output_tokens||0) : null} current={(turn.router.input_tokens||0)+(turn.router.output_tokens||0)} lower />
                        <MetricWithDelta label="Cost" value={`$${turn.router.cost || 0}`} baseline={turn.baseline?.cost} current={turn.router.cost} lower />
                        <div className="text-center cursor-pointer" onClick={e => turn.router?.accuracy_score != null && setAccuracyPopup({side:'router',score:turn.router.accuracy_score,reasoning:turn.router.accuracy_reasoning,position:{x:e.clientX,y:e.clientY}})}>
                          <div className="text-[9px] text-gray-500">Accuracy</div>
                          <div className="text-xs font-mono text-orange-300 underline decoration-dotted">{turn.router.accuracy_score != null ? `${turn.router.accuracy_score}/10` : <span className="animate-pulse text-gray-600">...</span>}</div>
                        </div>
                      </div>
                    )}
                  </div>
                  <div className="p-3 text-sm text-gray-300 bg-[#0d1210] min-h-[60px]">
                    {turn.router === null ? (
                      <div className="animate-pulse text-gray-600 text-xs">Generating...</div>
                    ) : turn.router.error ? (
                      <div className="text-red-400 text-xs">❌ {turn.router.error}</div>
                    ) : turn.router._streaming ? (
                      <div>
                        {turn.router._progress && <div className="text-[10px] text-yellow-400 mb-2 animate-pulse">{turn.router._progress}</div>}
                        <ResponseWithThinking text={turn.router.response_text} variant="router" streaming={true} />
                      </div>
                    ) : (
                      <div>
                        {turn.router._tools_used?.length > 0 && (
                          <div className="flex gap-1 mb-2 pb-2 border-b border-orange-900/20">
                            {turn.router._tools_used.map((t, i) => <span key={i} className="text-[9px] bg-orange-900/30 text-orange-300 px-1.5 py-0.5 rounded">🔧 {t}</span>)}
                          </div>
                        )}
                        <ResponseWithThinking text={turn.router.response_text} variant="router" streaming={false} />
                      </div>
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
          {/* Voice recording indicator */}
          {isListening && (
            <div className="flex items-center gap-3 mb-2 px-3 py-2 bg-red-950/30 border border-red-900/40 rounded-lg">
              <AudioWaveform isActive={isListening} color="#ef4444" />
              <span className="text-xs text-red-400 font-medium">Listening...</span>
              <span className="text-xs text-red-300 font-mono">{formatDuration(recordingDuration)}</span>
              <span className="text-[10px] text-gray-500 ml-auto">Speak now — auto-sends on silence</span>
            </div>
          )}
          {/* Model loading progress */}
          {isModelLoading && (
            <div className="flex items-center gap-2 mb-2 px-3 py-2 bg-orange-950/20 border border-orange-900/30 rounded-lg">
              <svg className="animate-spin h-3 w-3 text-orange-400" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"/><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/></svg>
              <span className="text-xs text-orange-400">Loading Whisper model... {modelProgress}%</span>
              <div className="flex-1 h-1.5 bg-gray-800 rounded-full overflow-hidden">
                <div className="h-full bg-orange-500 rounded-full transition-all duration-300" style={{ width: `${modelProgress}%` }} />
              </div>
            </div>
          )}
          {/* Transcription loading */}
          {isSpeechLoading && (
            <div className="flex items-center gap-2 mb-2 px-3 py-2 bg-blue-950/20 border border-blue-900/30 rounded-lg">
              <svg className="animate-spin h-3 w-3 text-blue-400" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"/><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/></svg>
              <span className="text-xs text-blue-400">Transcribing audio...</span>
            </div>
          )}
          {/* Speech error */}
          {speechError && (
            <div className="flex items-center gap-2 mb-2 px-3 py-1.5 bg-red-950/20 border border-red-900/30 rounded-lg">
              <span className="text-xs text-red-400">⚠️ {speechError}</span>
            </div>
          )}
          <div className="relative">
            <textarea value={message} onChange={e => setMessage(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage() }}}
              placeholder={isListening ? 'Listening...' : 'Ask about AWS architecture, services, best practices...'}
              rows={2}
              disabled={isListening}
              className="w-full bg-gray-900/60 border border-gray-700/50 rounded-xl px-4 py-3 pr-28 text-sm text-gray-200 placeholder-gray-500 focus:outline-none focus:border-orange-500/50 focus:ring-1 focus:ring-orange-500/20 resize-none disabled:opacity-50" />
            <div className="absolute right-3 bottom-3 flex items-center gap-2">
              {/* Mic button */}
              {isSpeechSupported && (
                <button
                  onClick={handleMicClick}
                  disabled={loading || isModelLoading || isSpeechLoading}
                  title={isListening ? 'Stop recording' : 'Voice input (Whisper)'}
                  className={`p-1.5 rounded-lg transition-all disabled:opacity-40 ${
                    isListening
                      ? 'bg-red-600 hover:bg-red-700 text-white animate-mic-pulse'
                      : 'bg-gray-700 hover:bg-gray-600 text-gray-300 hover:text-white'
                  }`}
                >
                  {isListening ? (
                    <svg className="h-4 w-4" fill="currentColor" viewBox="0 0 24 24">
                      <rect x="6" y="6" width="12" height="12" rx="2" />
                    </svg>
                  ) : (
                    <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M12 18.75a6 6 0 006-6v-1.5m-6 7.5a6 6 0 01-6-6v-1.5m6 7.5v3.75m-3.75 0h7.5M12 15.75a3 3 0 01-3-3V4.5a3 3 0 116 0v8.25a3 3 0 01-3 3z" />
                    </svg>
                  )}
                </button>
              )}
              {/* Send button */}
              <button id="strands-send-btn" onClick={sendMessage} disabled={loading || !message.trim() || isListening}
                className="bg-orange-600 hover:bg-orange-700 disabled:opacity-40 text-white text-xs font-medium rounded-lg px-3 py-1.5 flex items-center gap-1.5 transition-all">
                {loading ? <svg className="animate-spin h-3 w-3" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"/><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/></svg> : null}
                {loading ? 'Thinking...' : 'Send'}
              </button>
            </div>
          </div>
          <div className="flex items-center gap-2 mt-1.5 text-[9px] text-gray-600">
            <span className="text-[10px] text-gray-500 font-bold uppercase">Send to</span>
            {['both', 'baseline', 'router'].map(t => (
              <button key={t} onClick={() => setSendTarget(t)}
                className={`text-[10px] px-2 py-0.5 rounded-lg font-medium transition-all capitalize ${
                  sendTarget === t ? 'bg-purple-600/20 text-purple-300 border border-purple-500/40' : 'text-gray-600 hover:text-gray-400 border border-gray-800/50'
                }`}>
                {t}
              </button>
            ))}
            <span className="text-gray-700">•</span>
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
