import { useState, useEffect } from 'react'
import { Md, ExplainPopup, AccuracyPopup, ResponseWithThinking, API, STREAM_API } from './shared'

// ── Step Section (same pattern as ComparePage) ─────────────────────
function StepSection({ number, title, visible, expanded, onToggle, children }) {
  if (!visible) return null
  return (
    <div className="bg-gray-900/30 border border-gray-800/50 rounded-xl overflow-hidden animate-slideDown">
      <button onClick={onToggle} className="w-full flex items-center gap-3 px-4 py-3 hover:bg-gray-800/20 transition-colors">
        <span className="text-xs font-bold text-orange-400 bg-orange-900/30 w-6 h-6 rounded-full flex items-center justify-center">{number}</span>
        <span className="text-sm font-medium text-gray-300 flex-1 text-left">{title}</span>
        <svg className={`w-4 h-4 text-gray-500 transition-transform ${expanded ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7"/></svg>
      </button>
      {expanded && <div className="px-4 pb-4 border-t border-gray-800/30 pt-3">{children}</div>}
    </div>
  )
}

// ── Syntax-highlighted Python code ─────────────────────────────────
function PythonCode({ code, className = '' }) {
  // Simple Python syntax highlighting — order matters to avoid double-matching
  let highlighted = code
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')

  // Split into lines and highlight each line to handle comments correctly
  highlighted = highlighted.split('\n').map(line => {
    // Find comment position (# not inside a string)
    const commentIdx = line.search(/(?<![&\w])#/)
    let codePart = line
    let commentPart = ''
    if (commentIdx >= 0) {
      codePart = line.slice(0, commentIdx)
      commentPart = `<span class="text-gray-500 italic">${line.slice(commentIdx)}</span>`
    }

    // Highlight the code part
    codePart = codePart
      // Strings (double-quoted)
      .replace(/("(?:[^"\\]|\\.)*")/g, '<span class="text-green-400">$1</span>')
      // Keywords
      .replace(/\b(from|import|True|False|None)\b/g, '<span class="text-purple-400">$1</span>')
      // Function/method calls
      .replace(/\b(BedrockRouter|RoutingConfig|router)\b/g, '<span class="text-blue-300">$1</span>')
      .replace(/\.(create|converse)\b/g, '.<span class="text-yellow-300">$1</span>')
      // Named parameters
      .replace(/\b(strategy|preferred_model|max_cost_per_request|exclude_models|metadata|tags|region|aip|enabled|auto_create|tag_keys|messages|routing|excluded_models|explain|classifier)\b(?=\s*=)/g, '<span class="text-orange-300">$1</span>')

    return codePart + commentPart
  }).join('\n')

  return (
    <pre className={`text-[10px] bg-[#0d1117] border border-gray-800/60 rounded-lg p-3 overflow-x-auto font-mono leading-relaxed ${className}`}>
      <code dangerouslySetInnerHTML={{ __html: highlighted }} />
    </pre>
  )
}

// ── Tenant definitions ─────────────────────────────────────────────
// ── Tenant color mappings (by color name from backend) ─────────────
const TENANT_STYLES = {
  purple: { borderColor: 'border-purple-900/40', headerBg: 'bg-purple-950/20', textColor: 'text-purple-400', badgeColor: 'bg-purple-900/40 text-purple-300' },
  orange: { borderColor: 'border-orange-900/40', headerBg: 'bg-orange-950/20', textColor: 'text-orange-400', badgeColor: 'bg-orange-900/40 text-orange-300' },
  green: { borderColor: 'border-green-900/40', headerBg: 'bg-green-950/20', textColor: 'text-green-400', badgeColor: 'bg-green-900/40 text-green-300' },
}

// ── Main Component ─────────────────────────────────────────────────
export default function MultiTenantPage({ onRun }) {
  const [templates, setTemplates] = useState([])
  const [tenants, setTenants] = useState([])
  const [routerSetupCode, setRouterSetupCode] = useState('')
  const [difficulty, setDifficulty] = useState('medium')
  const [prompt, setPrompt] = useState('')
  const [systemPrompt, setSystemPrompt] = useState('')
  const [isManual, setIsManual] = useState(false)
  const [file, setFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [classifier, setClassifier] = useState('heuristic')
  const [results, setResults] = useState({})
  const [scores, setScores] = useState({})
  const [explainPopup, setExplainPopup] = useState(null)
  const [accuracyPopup, setAccuracyPopup] = useState(null)
  const [codeExpanded, setCodeExpanded] = useState({ enterprise: true, free: true })

  // Step visibility (same pattern as ComparePage)
  const [step1Expanded, setStep1Expanded] = useState(true)
  const [step2Visible, setStep2Visible] = useState(false)
  const [step2Expanded, setStep2Expanded] = useState(true)
  const [step3Visible, setStep3Visible] = useState(false)
  const [step3Expanded, setStep3Expanded] = useState(true)

  useEffect(() => {
    fetch(`${API}/templates`).then(r => r.json()).then(setTemplates).catch(() => {})
    fetch(`${API}/multi-tenant/tenants`).then(r => r.json()).then(data => {
      setTenants(data.tenants || [])
      setRouterSetupCode(data.router_setup_code || '')
    }).catch(() => {})
  }, [])

  const filtered = templates.filter(t => t.difficulty === difficulty)

  function selectTemplate(t) {
    setPrompt(t.prompt)
    setSystemPrompt(t.system_prompt || '')
    setIsManual(false)
    setFile(null)
    setStep2Visible(true)
    setStep2Expanded(true)
    setStep1Expanded(false)
    setStep3Visible(false)
    setResults({})
    setScores({})
  }

  function handleManualEntry() {
    setPrompt('')
    setSystemPrompt('')
    setIsManual(true)
    setFile(null)
    setStep2Visible(true)
    setStep2Expanded(true)
    setStep1Expanded(false)
    setStep3Visible(false)
    setResults({})
    setScores({})
  }

  async function runAll() {
    if (!prompt.trim() || loading) return
    if (onRun) onRun()
    setLoading(true)
    setResults({})
    setScores({})
    setCodeExpanded({ enterprise: true, free: true })
    setStep3Visible(true)
    setStep3Expanded(true)
    setStep2Expanded(false)

    const form = new FormData()
    form.append('prompt', prompt.trim())
    form.append('system_prompt', systemPrompt.trim())
    form.append('classifier', classifier)

    try {
      const res = await fetch(`${STREAM_API}/multi-tenant/run`, { method: 'POST', body: form })
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
            if (eventType === 'tenant_token') {
              setResults(prev => ({
                ...prev,
                [parsed.tenant_id]: {
                  ...(prev[parsed.tenant_id] || {}),
                  response_text: (prev[parsed.tenant_id]?.response_text || '') + parsed.text,
                  _streaming: true,
                }
              }))
            } else if (eventType === 'tenant_complete') {
              setResults(prev => ({ ...prev, [parsed.tenant_id]: { ...parsed, _streaming: false } }))
            } else if (eventType === 'tenant_error') {
              setResults(prev => ({ ...prev, [parsed.tenant_id]: { error: parsed.error } }))
            } else if (eventType === 'judge_score') {
              setScores(prev => ({ ...prev, [parsed.tenant_id]: parsed }))
            } else if (eventType === 'done') {
              setLoading(false)
              setCodeExpanded({ enterprise: false, free: false })
            }
            eventType = null
          }
        }
      }
    } catch (err) {
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex-1 flex min-w-0 overflow-hidden">
      {explainPopup && <ExplainPopup explanation={explainPopup} onClose={() => setExplainPopup(null)} />}
      {accuracyPopup && <AccuracyPopup {...accuracyPopup} onClose={() => setAccuracyPopup(null)} />}

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
              <button key={d.id} onClick={() => { if (d.id === 'manual') { handleManualEntry() } else { setDifficulty(d.id); setIsManual(false) } }}
                className={`text-[11px] px-3 py-1.5 rounded-lg font-medium transition-all ${
                  d.id === 'manual'
                    ? (isManual ? 'bg-green-600/20 text-green-300 border border-green-500/40' : 'text-green-400 border border-green-700/50 hover:bg-green-900/20')
                    : difficulty === d.id && !isManual
                      ? 'bg-orange-600/20 text-orange-300 border border-orange-500/40'
                      : 'text-gray-500 hover:text-gray-300 border border-gray-800/50 hover:border-gray-700'
                }`}>
                {d.icon} {d.label}
              </button>
            ))}
          </div>
          {!isManual && (
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

        {/* ═══ Step 2: System & User Prompt + Code + Run ═══ */}
        <StepSection number={2} title="Prompt & Router Configuration" visible={step2Visible} expanded={step2Expanded} onToggle={() => setStep2Expanded(!step2Expanded)}>
          {/* System prompt */}
          <div className="mb-3">
            <div className="text-[10px] text-gray-500 uppercase font-bold mb-1">System Prompt</div>
            {isManual ? (
              <textarea value={systemPrompt} onChange={e => setSystemPrompt(e.target.value)} rows={2} placeholder="You are a helpful assistant..."
                className="w-full text-xs text-gray-300 bg-gray-900/40 border border-gray-800/50 rounded-lg px-3 py-2 font-mono resize-none focus:outline-none focus:border-orange-600/40" />
            ) : (
              <div className="text-xs text-gray-400 bg-gray-900/60 rounded-lg px-3 py-2 border border-gray-800/50 font-mono">{systemPrompt || '(none)'}</div>
            )}
          </div>
          {/* User prompt */}
          <div className="mb-3">
            <div className="text-[10px] text-gray-500 uppercase font-bold mb-1">User Prompt</div>
            {isManual ? (
              <div className="relative">
                <textarea value={prompt} onChange={e => setPrompt(e.target.value)}
                  onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); runAll() }}}
                  placeholder="Ask anything..." rows={3}
                  className="w-full bg-gray-900/60 border border-gray-700/50 rounded-xl px-4 py-3 text-sm text-gray-200 placeholder-gray-500 focus:outline-none focus:border-orange-500/50 focus:ring-1 focus:ring-orange-500/20 resize-none" />
                <div className="absolute right-3 bottom-2.5 flex items-center gap-2">
                  {file && <span className="text-[9px] text-gray-400 bg-gray-800 px-1.5 py-0.5 rounded">📎{file.name}<button onClick={() => setFile(null)} className="text-red-400 ml-1">&times;</button></span>}
                  <button onClick={() => document.getElementById('mt-file-input')?.click()} className="text-gray-600 hover:text-gray-400 p-1" title="Attach file">
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13" /></svg>
                  </button>
                  <button onClick={runAll} disabled={loading || !prompt.trim()}
                    className="bg-orange-600 hover:bg-orange-700 disabled:opacity-40 text-white text-xs font-medium rounded-lg px-4 py-1.5 flex items-center gap-1.5 transition-all shadow-lg shadow-orange-900/20">
                    {loading ? <svg className="animate-spin h-3.5 w-3.5" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"/><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/></svg> : null}
                    {loading ? 'Running...' : 'Run All'}
                  </button>
                </div>
                <input id="mt-file-input" type="file" accept=".pdf,.png,.jpg,.jpeg,.gif,.webp" onChange={e => setFile(e.target.files?.[0] || null)} className="hidden" />
              </div>
            ) : (
              <div className="text-xs text-gray-300 bg-gray-900/60 rounded-lg px-3 py-2 border border-gray-800/50">{prompt}</div>
            )}
          </div>
          {/* File upload section removed — it's inline in the textarea now */}
          {/* Router setup code (expanded by default) */}
          <div className="mb-4">
            <div className="text-[10px] text-gray-500 uppercase font-bold mb-1.5">Router Setup (shared across all tenants)</div>
            <PythonCode code={routerSetupCode.replace('"heuristic"', `"${classifier}"`)} />
            <div className="text-[9px] text-gray-600 mt-1.5 flex items-center gap-1">
              <span>☁️</span> Each tenant auto-gets a dedicated Application Inference Profile → per-tenant metrics in CloudWatch + cost breakdown in Cost Explorer
            </div>
          </div>
          {/* Classifier toggle + Run button */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-1.5">
              <span className="text-[10px] text-gray-500 font-bold uppercase">Classifier</span>
              <div className="flex bg-gray-900/80 rounded-lg p-0.5 border border-gray-800/50">
                <button onClick={() => setClassifier('heuristic')}
                  className={`text-[11px] px-2 py-1 rounded-md font-medium transition-all ${classifier === 'heuristic' ? 'bg-purple-600 text-white shadow-sm' : 'text-gray-500 hover:text-gray-300'}`}>
                  Heuristic
                </button>
                <button onClick={() => setClassifier('ml')}
                  className={`text-[11px] px-2 py-1 rounded-md font-medium transition-all ${classifier === 'ml' ? 'bg-purple-600 text-white shadow-sm' : 'text-gray-500 hover:text-gray-300'}`}>
                  ML
                </button>
              </div>
            </div>
            <button onClick={runAll} disabled={loading}
              className="bg-orange-600 hover:bg-orange-700 disabled:opacity-40 text-white text-sm font-bold rounded-lg px-5 py-2.5 flex items-center gap-2 transition-all shadow-lg shadow-orange-900/20">
              {loading ? <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"/><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/></svg> : <span>⚡</span>}
              {loading ? 'Running tenants...' : 'Run Both Tenants'}
            </button>
          </div>
        </StepSection>

        {/* ═══ Step 3: Results per Tenant ═══ */}
        <StepSection number={3} title="Results — Same Prompt, Different Routing" visible={step3Visible} expanded={step3Expanded} onToggle={() => setStep3Expanded(!step3Expanded)}>
          {(() => {
            // Compute best/worst per metric across all tenants
            const completedTenants = tenants.filter(t => results[t.id] && !results[t.id].error)
            const getBest = (key, lower = true) => {
              if (completedTenants.length < 2) return null
              const vals = completedTenants.map(t => ({ id: t.id, val: results[t.id]?.[key] })).filter(v => v.val != null)
              if (vals.length < 2) return null
              return lower ? vals.reduce((a, b) => a.val < b.val ? a : b).id : vals.reduce((a, b) => a.val > b.val ? a : b).id
            }
            const getWorst = (key, lower = true) => {
              if (completedTenants.length < 2) return null
              const vals = completedTenants.map(t => ({ id: t.id, val: results[t.id]?.[key] })).filter(v => v.val != null)
              if (vals.length < 2) return null
              return lower ? vals.reduce((a, b) => a.val > b.val ? a : b).id : vals.reduce((a, b) => a.val < b.val ? a : b).id
            }
            const bestTtft = getBest('ttft_ms')
            const worstTtft = getWorst('ttft_ms')
            const bestLatency = getBest('latency_ms')
            const worstLatency = getWorst('latency_ms')
            const bestCost = getBest('cost')
            const worstCost = getWorst('cost')
            const bestScore = scores ? (() => {
              const vals = tenants.filter(t => scores[t.id]).map(t => ({ id: t.id, val: scores[t.id].score }))
              return vals.length >= 2 ? vals.reduce((a, b) => a.val > b.val ? a : b).id : null
            })() : null
            const worstScore = scores ? (() => {
              const vals = tenants.filter(t => scores[t.id]).map(t => ({ id: t.id, val: scores[t.id].score }))
              return vals.length >= 2 ? vals.reduce((a, b) => a.val < b.val ? a : b).id : null
            })() : null

            const metricIcon = (tenantId, metricBest, metricWorst) => {
              if (tenantId === metricBest) return <span className="text-green-400" title="Best">🏆</span>
              return null
            }

            return (
          <div className="grid grid-cols-2 gap-3">
            {tenants.map(tenant => {
              const r = results[tenant.id]
              const score = scores[tenant.id]
              const expanded = codeExpanded[tenant.id]
              const styles = TENANT_STYLES[tenant.color] || TENANT_STYLES.orange

              return (
                <div key={tenant.id} className={`border ${styles.borderColor} rounded-xl overflow-hidden flex flex-col`}>
                  {/* Tenant Header — compact single row */}
                  <div className={`px-3 py-2 ${styles.headerBg} border-b ${styles.borderColor}`}>
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <span className="text-base">{tenant.icon}</span>
                        <span className={`text-xs font-bold ${styles.textColor}`}>{tenant.name}</span>
                        <span className={`text-[9px] px-1.5 py-0.5 rounded ${styles.badgeColor}`}>{tenant.tier}</span>
                        {r && !r.error && <span className={`text-[9px] px-1.5 py-0.5 rounded ${styles.badgeColor}`}>{r.model_used}</span>}
                        {r && !r.error && r.complexity_detected && <span className="text-[9px] bg-gray-800 text-gray-400 px-1.5 py-0.5 rounded">{r.complexity_detected}</span>}
                      </div>
                      <div className="flex items-center gap-1.5">
                        {r && !r.error && r.explanation && (
                          <button onClick={() => setExplainPopup({...r.explanation, _fallback_used: r.fallback_used, _actual_model: r.model_used})}
                            className={`text-[9px] ${styles.textColor} hover:opacity-80 bg-gray-900/40 px-1.5 py-0.5 rounded transition-all`}>
                            ⓘ Explain
                          </button>
                        )}
                      </div>
                    </div>

                    {/* Metrics row */}
                    {r && !r.error && (
                      <div className="flex items-center gap-3 mt-1.5 text-center">
                        <div className="flex items-center gap-0.5">{metricIcon(tenant.id, bestTtft, worstTtft)}<span className="text-[8px] text-gray-500">TTFT </span><span className="text-[10px] font-mono text-gray-300">{r.ttft_ms}ms</span></div>
                        <div className="flex items-center gap-0.5">{metricIcon(tenant.id, bestLatency, worstLatency)}<span className="text-[8px] text-gray-500">Latency </span><span className="text-[10px] font-mono text-gray-300">{r.latency_ms}ms</span></div>
                        <div><span className="text-[8px] text-gray-500">Tokens </span><span className="text-[10px] font-mono text-gray-300">{r.input_tokens}↓{r.output_tokens}↑</span></div>
                        <div className="flex items-center gap-0.5">{metricIcon(tenant.id, bestCost, worstCost)}<span className="text-[8px] text-gray-500">Cost </span><span className="text-[10px] font-mono text-gray-300">${r.cost}</span>{(() => {
                          const otherTenant = tenants.find(t => t.id !== tenant.id)
                          const otherCost = results[otherTenant?.id]?.cost
                          if (otherCost && otherCost > 0 && r.cost !== otherCost) {
                            const pct = Math.round(((r.cost - otherCost) / otherCost) * 100)
                            const cheaper = r.cost < otherCost
                            return <span className={`text-[9px] font-medium ml-1 ${cheaper ? 'text-green-400' : 'text-red-400'}`}>{cheaper ? '↓' : '↑'}{Math.abs(pct)}%</span>
                          }
                          return null
                        })()}</div>
                        <div className="flex items-center gap-0.5 cursor-pointer" onClick={e => score && setAccuracyPopup({side: tenant.name, score: score.score, reasoning: score.reasoning, position: {x: e.clientX, y: e.clientY}})}>{metricIcon(tenant.id, bestScore, worstScore)}<span className="text-[8px] text-gray-500">Accuracy </span><span className={`text-[10px] font-mono font-bold underline decoration-dotted ${score ? (score.score >= 8 ? 'text-green-400' : score.score >= 6 ? 'text-yellow-400' : 'text-red-400') : 'text-gray-600'}`}>{score ? `${score.score}/10` : '...'}</span>{(() => {
                          const otherTenant = tenants.find(t => t.id !== tenant.id)
                          const otherScore = scores?.[otherTenant?.id]?.score
                          if (score && otherScore && score.score !== otherScore) {
                            const better = score.score > otherScore
                            const delta = Math.abs(score.score - otherScore)
                            return <span className={`text-[9px] font-medium ml-1 ${better ? 'text-green-400' : 'text-red-400'}`}>{better ? '↑' : '↓'}{delta}pt</span>
                          }
                          return null
                        })()}</div>
                      </div>
                    )}
                    {r && r.error && <div className="text-[10px] text-red-400 mt-1">❌ {r.error}</div>}
                    {!r && loading && <div className="text-[10px] text-gray-500 animate-pulse mt-1">⏳ Generating...</div>}
                  </div>

                  {/* Collapsible Code — router.converse() call */}
                  <div className={`border-b ${styles.borderColor}`}>
                    <button onClick={() => setCodeExpanded(prev => ({ ...prev, [tenant.id]: !prev[tenant.id] }))}
                      className="w-full text-left px-3 py-1.5 text-[9px] text-gray-500 hover:text-gray-300 hover:bg-gray-900/40 flex items-center gap-1 transition-all">
                      <svg className={`w-3 h-3 transition-transform ${expanded ? 'rotate-90' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7"/></svg>
                      <code className="font-mono">router.converse()</code> config
                    </button>
                    {expanded && (
                      <div className="px-2 pb-2">
                        <PythonCode code={tenant.code.replace(
                          'explain=True,',
                          `classifier="${classifier}",\n        explain=True,`
                        )} className="text-[9px]" />
                      </div>
                    )}
                  </div>

                  {/* Response */}
                  <div className="flex-1 p-3 text-[11px] text-gray-300 overflow-y-auto max-h-[300px] bg-[#080d18]">
                    {r && !r.error ? (
                      <ResponseWithThinking text={r.response_text} variant={tenant.id === 'enterprise' ? 'baseline' : 'router'} />
                    ) : r && r.error ? (
                      <div className="text-red-400 text-xs">Error: {r.error}</div>
                    ) : loading ? (
                      <div className="text-gray-600 animate-pulse">Waiting for response...</div>
                    ) : (
                      <div className="text-gray-700 text-center mt-8">Run to see response</div>
                    )}
                  </div>
                </div>
              )
            })}
          </div>
            )
          })()}
        </StepSection>
      </div>
    </div>
  )
}
