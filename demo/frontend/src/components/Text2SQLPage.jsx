import { useState, useRef, useEffect, useCallback } from 'react'
import { useSpeechToText } from '../hooks/useSpeechToText'
import AudioWaveform from './AudioWaveform'
import { ExplainPopup, ResponseWithThinking, STREAM_API } from './shared'

/**
 * Text2SQL Page — Chat-style agent for querying e-commerce data.
 */

// Semantic cache demo: original questions mapped to semantically similar variants
const CACHE_DEMO_QUESTIONS = {
  'Show me the month-over-month growth rate of orders for each category in 2025': [
    'What is the percentage change in orders per category each month in 2025?',
    'Display the monthly order growth by product category for 2025',
    'How did order volumes change month to month across categories in 2025?',
  ],
  'Show monthly order trends for 2025': [
    'Visualize the order trends for the year 2025',
    'Show the order count per month for 2025',
    'Give me the 2025 monthly order trend data',
  ],
  'For customers who placed orders in both Q1 and Q2 of 2025, what was their average order value change between quarters?': [
    'Compare average order values between Q1 and Q2 2025 for repeat customers',
    'For repeat customers in Q1 and Q2 of 2025, how did their average order value change?',
    'What was the change in average order value between Q1 and Q2 2025 for customers active in both?',
  ],
  'Show me a chart of sales by category': [
    'Show sales chart grouped by category',
    'Give me a sales by category chart',
    'Chart the sales distribution across categories',
  ],
}

export default function Text2SQLPage({ onRun }) {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [sessionId] = useState(() => crypto.randomUUID())
  const [explainPopup, setExplainPopup] = useState(null)
  const [pendingVoiceSubmit, setPendingVoiceSubmit] = useState(false)
  const [showSystemPrompt, setShowSystemPrompt] = useState(false)
  const [systemPromptText, setSystemPromptText] = useState('')
  const [strategy, setStrategy] = useState('quality-optimized')
  const [lastOriginalQuestion, setLastOriginalQuestion] = useState(null)

  useEffect(() => {
    fetch(`${STREAM_API}/text2sql/system-prompt`).then(r => r.json()).then(d => setSystemPromptText(d.system_prompt || '')).catch(() => {})
  }, [])

  const messagesEndRef = useRef(null)
  const chatContainerRef = useRef(null)
  const textareaRef = useRef(null)
  const userScrolledRef = useRef(false)

  // Voice input
  const {
    isListening, isLoading: isSpeechLoading, isModelLoading, modelProgress,
    transcript: speechTranscript, error: speechError, recordingDuration,
    startListening, stopListening, resetTranscript, isSupported: isSpeechSupported,
  } = useSpeechToText({
    model: 'Xenova/whisper-tiny.en',
    silenceThreshold: 0.01,
    silenceTimeout: 2000,
    onSilenceDetected: () => { stopListening(); setPendingVoiceSubmit(true) },
  })

  // Auto-submit voice transcript
  useEffect(() => {
    if (speechTranscript && pendingVoiceSubmit && !loading) {
      setPendingVoiceSubmit(false)
      const text = speechTranscript.trim()
      resetTranscript()
      if (text) sendMessage(text)
    }
  }, [speechTranscript, pendingVoiceSubmit, loading])

  const handleMicClick = async () => {
    if (isListening) { stopListening(); setPendingVoiceSubmit(true) }
    else { resetTranscript(); setInput(''); setPendingVoiceSubmit(false); await startListening() }
  }

  const formatDuration = (s) => `${Math.floor(s/60).toString().padStart(2,'0')}:${(s%60).toString().padStart(2,'0')}`

  // Auto-scroll
  useEffect(() => {
    if (!userScrolledRef.current) chatContainerRef.current?.scrollTo({ top: chatContainerRef.current.scrollHeight, behavior: 'smooth' })
  }, [messages])

  // Auto-resize textarea
  useEffect(() => {
    const el = textareaRef.current
    if (el) { el.style.height = 'auto'; el.style.height = Math.min(el.scrollHeight, 120) + 'px' }
  }, [input])

  const sendMessage = useCallback(async (text) => {
    const msg = (text || input).trim()
    if (!msg || loading) return
    if (onRun) onRun()

    setMessages(prev => [...prev, { role: 'user', content: msg }])
    setInput('')
    setLoading(true)
    userScrolledRef.current = false

    // Track if this is an original demo question (for showing cache demo pills)
    if (CACHE_DEMO_QUESTIONS[msg]) {
      setLastOriginalQuestion(msg)
    }

    // Add streaming placeholder
    setMessages(prev => [...prev, { role: 'assistant', content: '', _streaming: true, _tools: [] }])

    const form = new FormData()
    form.append('message', msg)
    form.append('session_id', sessionId)
    form.append('strategy', strategy)

    try {
      const res = await fetch(`${STREAM_API}/text2sql/chat`, { method: 'POST', body: form })
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
            const parsed = JSON.parse(line.slice(6))
            if (eventType === 'token') {
              setMessages(prev => {
                const updated = [...prev]
                const last = updated[updated.length - 1]
                if (last?.role === 'assistant') updated[updated.length - 1] = { ...last, content: last.content + parsed.text }
                return updated
              })
            } else if (eventType === 'status') {
              setMessages(prev => {
                const updated = [...prev]
                const last = updated[updated.length - 1]
                if (last?.role === 'assistant') updated[updated.length - 1] = { ...last, _status: parsed.message }
                return updated
              })
            } else if (eventType === 'tool_use') {
              setMessages(prev => {
                const updated = [...prev]
                const last = updated[updated.length - 1]
                if (last?.role === 'assistant') updated[updated.length - 1] = { ...last, _tools: [...(last._tools || []), parsed.name] }
                return updated
              })
            } else if (eventType === 'metrics') {
              setMessages(prev => {
                const updated = [...prev]
                const last = updated[updated.length - 1]
                if (last?.role === 'assistant') updated[updated.length - 1] = { ...last, metrics: parsed, _streaming: false }
                return updated
              })
            } else if (eventType === 'error') {
              setMessages(prev => {
                const updated = [...prev]
                const last = updated[updated.length - 1]
                if (last?.role === 'assistant') updated[updated.length - 1] = { ...last, content: last.content + `\n❌ ${parsed.error}`, _streaming: false }
                return updated
              })
            } else if (eventType === 'done') {
              setLoading(false)
            }
            eventType = null
          }
        }
      }
    } catch (err) {
      setMessages(prev => {
        const updated = [...prev]
        const last = updated[updated.length - 1]
        if (last?.role === 'assistant') updated[updated.length - 1] = { ...last, content: `Error: ${err.message}`, _streaming: false }
        return updated
      })
    } finally {
      setLoading(false)
    }
  }, [input, loading, sessionId, onRun, strategy])

  const handleKeyDown = (e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage() } }

  const handleSuggestion = (text) => { if (!loading) sendMessage(text) }

  const handleReset = async () => {
    setMessages([])
    const form = new FormData()
    form.append('session_id', sessionId)
    fetch(`${STREAM_API}/text2sql/reset`, { method: 'POST', body: form }).catch(() => {})
  }

  return (
    <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
      {explainPopup && <ExplainPopup explanation={explainPopup} onClose={() => setExplainPopup(null)} />}

      {/* Header */}
      <div className="px-4 py-2 border-b border-gray-800/30 bg-gradient-to-r from-orange-950/20 to-purple-950/20">
        <div className="flex items-center gap-2">
          <span className="text-lg cursor-pointer" onClick={() => setShowSystemPrompt(!showSystemPrompt)}>🛒</span>
          <span className="text-sm font-bold text-gray-200 cursor-pointer" onClick={() => setShowSystemPrompt(!showSystemPrompt)}>E-Commerce Data Assistant</span>
          <span className="text-[10px] text-gray-500 bg-gray-800 px-2 py-0.5 rounded-full cursor-pointer" onClick={() => setShowSystemPrompt(!showSystemPrompt)}>Text2SQL + Semantic Cache</span>
          <svg className={`w-3 h-3 text-gray-500 transition-transform cursor-pointer ${showSystemPrompt ? 'rotate-180' : ''}`} onClick={() => setShowSystemPrompt(!showSystemPrompt)} fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7"/></svg>

          {/* Strategy selector */}
          <div className="flex items-center gap-1.5 ml-4">
            <span className="text-[9px] text-gray-500 font-bold uppercase">Strategy</span>
            <div className="flex bg-gray-900/80 rounded-lg p-0.5 border border-gray-800/50">
              {['balanced', 'cost-optimized', 'quality-optimized', 'latency-optimized'].map(s => (
                <button key={s} onClick={() => setStrategy(s)}
                  className={`text-[10px] px-2 py-0.5 rounded-md font-medium transition-all capitalize ${strategy === s ? 'bg-orange-600 text-white shadow-sm' : 'text-gray-500 hover:text-gray-300'}`}>
                  {s.replace('-optimized','').replace('-',' ')}
                </button>
              ))}
            </div>
          </div>

          <span className="text-[9px] text-gray-600 ml-auto">Powered by Strands Agents + Smart Router</span>
          <button onClick={handleReset} className="text-[10px] text-red-400 hover:text-red-300 px-2 py-1 rounded hover:bg-red-900/20 border border-red-900/30 transition-all ml-2">Reset Chat</button>
        </div>
        {showSystemPrompt && (
          <div className="mt-2 p-3 bg-gray-900/60 border border-gray-800/50 rounded-lg text-[11px] text-gray-400 leading-relaxed max-h-40 overflow-y-auto font-mono whitespace-pre-wrap">
            {systemPromptText || 'Loading...'}
          </div>
        )}
      </div>

      {/* Chat messages */}
      <div ref={chatContainerRef} className="flex-1 overflow-y-auto px-4 py-4 space-y-4"
        onScroll={e => { const el = e.currentTarget; userScrolledRef.current = el.scrollHeight - el.scrollTop - el.clientHeight > 80 }}>

        {/* Empty state */}
        {messages.length === 0 && (
          <div className="text-center mt-20">
            <div className="text-4xl mb-3">🛒</div>
            <div className="text-sm font-medium text-gray-400">How can I help you today?</div>
            <div className="text-xs text-gray-600 mt-1 max-w-md mx-auto">Ask about products, orders, customers, shipments, or sales data. I can generate SQL, create charts, and provide insights.</div>
            {/* Suggestion chips */}
            <div className="grid grid-cols-2 gap-3 mt-6 max-w-2xl mx-auto">
              {[
                { icon: '📈', q: 'Show me the month-over-month growth rate of orders for each category in 2025' },
                { icon: '📊', q: 'Show monthly order trends for 2025' },
                { icon: '🔄', q: 'For customers who placed orders in both Q1 and Q2 of 2025, what was their average order value change between quarters?' },
                { icon: '🛒', q: 'Show me a chart of sales by category' },
              ].map(({ icon, q }) => (
                <button key={q} onClick={() => handleSuggestion(q)} disabled={loading}
                  className="flex items-start gap-2.5 text-left text-[11px] px-4 py-3 rounded-xl border border-gray-700/50 bg-gray-900/60 text-gray-300 hover:border-orange-500/50 hover:bg-gray-800/80 hover:text-orange-200 transition-all disabled:opacity-40 min-h-[56px]">
                  <span className="text-base flex-shrink-0 mt-0.5">{icon}</span>
                  <span className="leading-relaxed">{q}</span>
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Messages */}
        {messages.map((msg, i) => (
          <ChatBubble key={i} msg={msg} onExplain={setExplainPopup}
            cacheDemoPills={
              // Show cache demo pills on the last assistant message that's done streaming
              !msg._streaming && msg.role === 'assistant' && msg.metrics && i === messages.length - 1 && lastOriginalQuestion
                ? CACHE_DEMO_QUESTIONS[lastOriginalQuestion] || null
                : null
            }
            onCacheDemoClick={(q) => { if (!loading) sendMessage(q) }}
          />
        ))}

        <div ref={messagesEndRef} />
      </div>

      {/* Input area */}
      <div className="px-4 py-3 border-t border-gray-800/50 bg-[#0e1420]">
        {/* Voice indicators */}
        {isListening && (
          <div className="flex items-center gap-3 mb-2 px-3 py-2 bg-red-950/30 border border-red-900/40 rounded-lg">
            <AudioWaveform isActive={isListening} color="#ef4444" />
            <span className="text-xs text-red-400 font-medium">Listening...</span>
            <span className="text-xs text-red-300 font-mono">{formatDuration(recordingDuration)}</span>
            <span className="text-[10px] text-gray-500 ml-auto">Auto-sends on silence</span>
          </div>
        )}
        {isModelLoading && (
          <div className="flex items-center gap-2 mb-2 px-3 py-2 bg-orange-950/20 border border-orange-900/30 rounded-lg">
            <span className="text-xs text-orange-400">Loading Whisper... {modelProgress}%</span>
            <div className="flex-1 h-1.5 bg-gray-800 rounded-full overflow-hidden"><div className="h-full bg-orange-500 rounded-full transition-all" style={{ width: `${modelProgress}%` }} /></div>
          </div>
        )}

        <div className="relative">
          <textarea ref={textareaRef} value={input} onChange={e => setInput(e.target.value)} onKeyDown={handleKeyDown}
            placeholder={isListening ? 'Listening...' : 'Ask about your e-commerce data...'} rows={1} disabled={isListening}
            className="w-full bg-gray-900/60 border border-gray-700/50 rounded-xl px-4 py-3 pr-28 text-sm text-gray-200 placeholder-gray-500 focus:outline-none focus:border-orange-500/50 focus:ring-1 focus:ring-orange-500/20 resize-none disabled:opacity-50" />
          <div className="absolute right-3 bottom-2.5 flex items-center gap-2">
            {isSpeechSupported && (
              <button onClick={handleMicClick} disabled={loading || isModelLoading || isSpeechLoading}
                title={isListening ? 'Stop' : 'Voice input'}
                className={`p-1.5 rounded-lg transition-all disabled:opacity-40 ${isListening ? 'bg-red-600 hover:bg-red-700 text-white animate-mic-pulse' : 'bg-gray-700 hover:bg-gray-600 text-gray-300 hover:text-white'}`}>
                {isListening ? <svg className="h-4 w-4" fill="currentColor" viewBox="0 0 24 24"><rect x="6" y="6" width="12" height="12" rx="2" /></svg>
                  : <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M12 18.75a6 6 0 006-6v-1.5m-6 7.5a6 6 0 01-6-6v-1.5m6 7.5v3.75m-3.75 0h7.5M12 15.75a3 3 0 01-3-3V4.5a3 3 0 116 0v8.25a3 3 0 01-3 3z" /></svg>}
              </button>
            )}
            <button onClick={() => sendMessage()} disabled={loading || !input.trim() || isListening}
              className="bg-orange-600 hover:bg-orange-700 disabled:opacity-40 text-white text-xs font-medium rounded-lg px-3 py-1.5 flex items-center gap-1.5 transition-all">
              {loading ? <svg className="animate-spin h-3 w-3" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"/><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/></svg> : null}
              {loading ? 'Thinking...' : 'Send'}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

// ── Chat Bubble Component ──────────────────────────────────────────

function ChatBubble({ msg, onExplain, cacheDemoPills, onCacheDemoClick }) {
  const isUser = msg.role === 'user'
  const metrics = msg.metrics

  return (
    <div className={`flex gap-3 ${isUser ? 'justify-end' : 'justify-start'}`}>
      {/* Agent avatar */}
      {!isUser && <div className="w-8 h-8 rounded-lg bg-green-900/40 flex items-center justify-center text-sm flex-shrink-0 mt-1">🤖</div>}

      <div className={`${isUser ? 'max-w-[70%]' : 'max-w-[85%]'} flex flex-col`}>
        {/* Tool use indicators */}
        {!isUser && msg._tools && msg._tools.length > 0 && (
          <div className="flex gap-1 mb-1">
            {[...new Set(msg._tools)].map((t, i) => <span key={i} className="text-[9px] bg-orange-900/30 text-orange-300 px-1.5 py-0.5 rounded">🔧 {t}</span>)}
          </div>
        )}

        {/* Metrics header (assistant only, after streaming completes) */}
        {!isUser && metrics && <MetricsHeader metrics={metrics} onExplain={onExplain} />}

        {/* Message bubble */}
        <div className={`rounded-xl px-4 py-3 text-sm leading-relaxed ${isUser ? 'bg-orange-600 text-white' : 'bg-gray-800/60 border border-gray-700/50 text-gray-200'}`}>
          {isUser ? <p>{msg.content}</p> : <AssistantContent content={msg.content} isStreaming={msg._streaming} />}
        </div>

        {/* Status indicator (while processing) */}
        {!isUser && msg._streaming && msg._status && (
          <div className="mt-1.5 flex items-center gap-2 px-3 py-1.5 bg-gray-900/40 border border-gray-800/30 rounded-lg">
            <svg className="animate-spin h-3 w-3 text-orange-400" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"/><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/></svg>
            <span className="text-[10px] text-gray-400">{msg._status}</span>
          </div>
        )}

        {/* Semantic cache demo pills */}
        {cacheDemoPills && cacheDemoPills.length > 0 && (
          <div className="mt-2 p-2 bg-blue-950/20 border border-blue-900/30 rounded-lg">
            <div className="text-[9px] text-blue-400 font-medium mb-1.5">🧪 Try a semantically similar question (should hit cache):</div>
            <div className="flex flex-wrap gap-1.5">
              {cacheDemoPills.map((q, i) => (
                <button key={i} onClick={() => onCacheDemoClick(q)}
                  className="text-[10px] px-2.5 py-1.5 rounded-lg border border-blue-800/40 bg-blue-950/30 text-blue-300 hover:border-blue-500/60 hover:bg-blue-900/40 hover:text-blue-200 transition-all">
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* User avatar */}
      {isUser && <div className="w-8 h-8 rounded-lg bg-orange-600 flex items-center justify-center text-sm flex-shrink-0 mt-1 text-white">👤</div>}
    </div>
  )
}

// ── Assistant Content (renders markdown + HTML via ResponseWithThinking) ────────

function stripHtmlFences(text) {
  if (!text) return text
  // Remove ```html ... ``` fences — unwrap the HTML so rehype-raw can parse it
  return text.replace(/```html\s*\n?([\s\S]*?)```/g, (_, html) => html.trim())
}

function AssistantContent({ content, isStreaming }) {
  const processed = stripHtmlFences(content)
  return (
    <div className="agent-response text-sm leading-relaxed">
      <ResponseWithThinking text={processed} variant="router" streaming={isStreaming} />
      {isStreaming && <span className="inline-block w-2 h-4 bg-orange-400 animate-pulse ml-0.5" />}
    </div>
  )
}

// ── Metrics Header ─────────────────────────────────────────────────

function MetricsHeader({ metrics, onExplain }) {
  const cacheHits = metrics.cache_hits || 0
  const cacheRead = metrics.prompt_cache_read || metrics.cache_read_tokens || 0
  const cacheWrite = metrics.prompt_cache_write || metrics.cache_write_tokens || 0
  const totalInput = (metrics.total_input_tokens || 0) + cacheRead
  const cacheEfficiency = totalInput > 0 ? Math.round((cacheRead / totalInput) * 100) : 0

  return (
    <div className="mb-1.5 px-3 py-2 bg-gray-900/60 border border-gray-800/50 rounded-lg">
      {/* Row 1: Model + complexity + explain */}
      <div className="flex items-center gap-2 mb-1.5">
        <span className="text-[9px] bg-orange-900/40 text-orange-300 px-1.5 py-0.5 rounded">{metrics.model_used || 'unknown'}</span>
        {metrics.complexity && <span className="text-[9px] bg-purple-900/40 text-purple-300 px-1.5 py-0.5 rounded">{metrics.complexity}</span>}
        {metrics.routing_overhead_ms != null && <span className="text-[9px] text-gray-500">Overhead: {metrics.routing_overhead_ms}ms</span>}
        {onExplain && metrics.explanation && (
          <button onClick={() => onExplain({...metrics.explanation, _fallback_used: metrics.fallback_used, _actual_model: metrics.model_used})}
            className="text-[9px] text-orange-400 hover:text-orange-300 bg-orange-900/20 hover:bg-orange-900/40 px-1.5 py-0.5 rounded transition-all ml-auto">ⓘ Explain</button>
        )}
      </div>

      {/* Row 2: Core metrics */}
      <div className="flex items-center gap-3 text-[10px]">
        <span><span className="text-gray-500">TTFT</span> <span className="text-gray-300 font-mono">{metrics.ttft_ms}ms</span></span>
        <span><span className="text-gray-500">Latency</span> <span className="text-gray-300 font-mono">{metrics.latency_ms}ms</span></span>
        <span><span className="text-gray-500">Tokens</span> <span className="text-gray-300 font-mono">{metrics.total_input_tokens || 0}↓ {metrics.total_output_tokens || 0}↑</span></span>
        <span><span className="text-gray-500">Cost</span> <span className="text-gray-300 font-mono">${metrics.cost || 0}</span></span>
        <span><span className="text-gray-500">Steps</span> <span className="text-gray-300 font-mono">{metrics.steps || 0}</span></span>
      </div>

      {/* Row 3: Cache savings (prompt cache + semantic cache) */}
      {(cacheRead > 0 || cacheHits > 0) && (
        <div className="flex items-center gap-3 mt-1.5 pt-1.5 border-t border-gray-800/40 text-[10px]">
          {cacheRead > 0 && (
            <span className="flex items-center gap-1 text-green-400">
              <span>📦</span>
              <span>Prompt Cache: {cacheRead.toLocaleString()} tokens read ({cacheEfficiency}% hit rate)</span>
              {cacheWrite > 0 && <span className="text-gray-500 ml-1">| {cacheWrite.toLocaleString()} written</span>}
            </span>
          )}
          {cacheHits > 0 && (
            <span className="flex items-center gap-1 text-blue-400">
              <span>⚡</span>
              <span>Semantic Cache Hit — skipped DB + Chart generation call</span>
            </span>
          )}
        </div>
      )}
    </div>
  )
}
