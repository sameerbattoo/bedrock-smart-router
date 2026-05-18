import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeRaw from 'rehype-raw'

// ─── Constants ─────────────────────────────────────────────────────
export const API = '/api'
export const STREAM_API = '/api'

export const USE_CASES = [
  { id: 'compare', label: 'Baseline vs Smart Router', icon: '⚡', description: 'Compare responses, cost, latency and accuracy side-by-side' },
  { id: 'throttling', label: 'Throttle Handling', icon: '🛡️', description: 'Automatic fallback when models are throttled' },
  { id: 'strands', label: 'Strands Agents', icon: '🤖', description: 'Use Smart Router as a model provider in Strands' },
  { id: 'multi-tenant', label: 'Multi-Tenant Routing', icon: '🏢', description: 'Per-tenant tracking, budgets and model segregation' },
  { id: 'semantic-cache', label: 'Semantic Caching', icon: '💾', description: 'Text2SQL with semantic cache, FAISS vectors, and chart generation' },
]

export const TIME_FILTERS = [
  { id: 'all', label: 'All Time' },
  { id: '10m', label: '10 min', ms: 10 * 60 * 1000 },
  { id: '30m', label: '30 min', ms: 30 * 60 * 1000 },
  { id: '1h', label: '1 hour', ms: 60 * 60 * 1000 },
  { id: '6h', label: '6 hours', ms: 6 * 60 * 60 * 1000 },
  { id: '24h', label: '24 hours', ms: 24 * 60 * 60 * 1000 },
]

// ─── Markdown ──────────────────────────────────────────────────────
export function Md({ children, variant = 'baseline' }) {
  const codeBg = variant === 'router' ? 'bg-[#1a1510] border-orange-900/30' : 'bg-[#0a1020] border-blue-900/30'
  const inlineBg = variant === 'router' ? 'bg-orange-950/40 text-orange-300' : 'bg-blue-950/40 text-blue-300'
  const tableBorder = variant === 'router' ? 'border-orange-800/40' : 'border-blue-800/40'
  const thBg = variant === 'router' ? 'bg-orange-950/50 text-orange-200' : 'bg-blue-950/50 text-blue-200'
  const tdBorder = variant === 'router' ? 'border-orange-900/30' : 'border-blue-900/30'
  return (
    <Markdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeRaw]} components={{
      code({ inline, children, ...props }) {
        return inline
          ? <code className={`${inlineBg} px-1.5 py-0.5 rounded text-xs`} {...props}>{children}</code>
          : <pre className={`${codeBg} border rounded-lg p-3 overflow-x-auto text-xs my-2`}><code {...props}>{children}</code></pre>
      },
      h1: ({children}) => <h1 className="text-lg font-bold mt-4 mb-2">{children}</h1>,
      h2: ({children}) => <h2 className="text-base font-semibold mt-3 mb-1.5">{children}</h2>,
      h3: ({children}) => <h3 className="text-sm font-semibold mt-2 mb-1">{children}</h3>,
      p: ({children}) => <p className="mb-2 leading-relaxed">{children}</p>,
      ul: ({children}) => <ul className="list-disc pl-5 mb-2 space-y-0.5">{children}</ul>,
      ol: ({children}) => <ol className="list-decimal pl-5 mb-2 space-y-0.5">{children}</ol>,
      li: ({children}) => <li className="text-sm">{children}</li>,
      table: ({children}) => <table className={`border-collapse border ${tableBorder} my-2 text-xs w-full rounded-lg overflow-hidden`}>{children}</table>,
      th: ({children}) => <th className={`border ${tdBorder} px-3 py-1.5 ${thBg} font-medium text-left`}>{children}</th>,
      td: ({children}) => <td className={`border ${tdBorder} px-3 py-1.5`}>{children}</td>,
    }}>{children}</Markdown>
  )
}

// ─── Thinking Block Detection & Display ────────────────────────────

/**
 * Parse <think>...</think> or <thinking>...</thinking> tags from response text.
 * Returns { thinking: string|null, content: string }
 */
export function parseThinking(text) {
  if (!text) return { thinking: null, content: text || '' }
  const match = text.match(/<think(?:ing)?>([\s\S]*?)<\/think(?:ing)?>/)
  if (!match) return { thinking: null, content: text }
  const thinking = match[1].trim()
  const content = text.replace(/<think(?:ing)?>[\s\S]*?<\/think(?:ing)?>\s*/, '').trim()
  return { thinking, content }
}

/**
 * Collapsible thinking section.
 * - `open`: whether the details element is open (controlled externally)
 * - When `streaming` is true, it stays open; when streaming ends, parent collapses it.
 */
export function ThinkingBlock({ thinking, open = false }) {
  if (!thinking) return null
  return (
    <details open={open} className="mb-2 border border-purple-900/30 rounded-lg overflow-hidden">
      <summary className="text-[10px] text-purple-400 bg-purple-950/20 px-2 py-1 cursor-pointer hover:bg-purple-950/30 select-none">
        🧠 Thinking...
      </summary>
      <div className="px-3 py-2 text-[11px] text-gray-500 italic leading-relaxed whitespace-pre-wrap">
        {thinking}
      </div>
    </details>
  )
}

/**
 * Render response text with thinking block support.
 * - `streaming`: if true, thinking block stays open
 * - `variant`: 'baseline' or 'router' for Md styling
 */
export function ResponseWithThinking({ text, variant = 'baseline', streaming = false }) {
  const { thinking, content } = parseThinking(text)
  return (
    <div>
      <ThinkingBlock thinking={thinking} open={streaming} />
      <Md variant={variant}>{content}</Md>
    </div>
  )
}

// ─── Small Components ──────────────────────────────────────────────
export function MetricWithDelta({ label, value, baseline, current, lower = true }) {
  let delta = null, color = 'text-gray-500'
  if (baseline != null && current != null && baseline !== 0) {
    const pct = ((current - baseline) / baseline * 100).toFixed(0)
    const absPct = Math.abs(Number(pct))
    if (absPct === 0) {
      color = 'text-orange-400'
      delta = '0%'
    } else {
      const improved = lower ? current < baseline : current > baseline
      color = improved ? 'text-green-400' : 'text-red-400'
      delta = `${improved ? '↓' : '↑'}${absPct}%`
    }
  }
  return (
    <div className="text-center">
      <div className="text-[9px] text-gray-500">{label}</div>
      <div className="text-xs font-mono text-gray-300">{value}</div>
      {delta && <div className={`text-[9px] font-medium ${color}`}>{delta}</div>}
    </div>
  )
}

export function StatCard({ label, value, color = 'text-white' }) {
  return (
    <div className="bg-gray-900/60 border border-gray-800/60 rounded-xl p-3 text-center">
      <div className="text-[10px] text-gray-500 uppercase tracking-wider mb-1">{label}</div>
      <div className={`text-xl font-bold ${color}`}>{value}</div>
    </div>
  )
}

// ─── Explain Popup ─────────────────────────────────────────────────
export function ExplainPopup({ explanation, onClose }) {
  if (!explanation) return null
  const cx = explanation.complexity || {}
  const strat = explanation.strategy || {}
  const candidates = explanation.top5_candidates || []
  const thresholds = cx.classification_thresholds || {}
  const payload = cx.multimodal_payload
  const dimScores = cx.dimension_scores || {}
  const tiers = ['micro','lite','mid','heavy','reasoning']
  const fallbackUsed = explanation._fallback_used
  const actualModel = explanation._actual_model

  return (
    <>
      <div className="fixed inset-0 z-[99] bg-black/50" onClick={onClose} />
      <div className="fixed z-[100] top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[92vw] max-w-[1200px] max-h-[88vh] overflow-y-auto bg-gray-900 border border-gray-700 rounded-xl p-6 shadow-2xl">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-bold text-orange-400">Routing Decision Explained</h3>
          <button onClick={onClose} className="text-gray-500 hover:text-white text-lg">&times;</button>
        </div>

        {/* Fallback notice */}
        {fallbackUsed && (
          <div className="mb-4 p-3 bg-yellow-900/20 border border-yellow-700/40 rounded-lg flex items-center gap-2">
            <span className="text-yellow-400 text-sm">⚡</span>
            <div>
              <div className="text-xs text-yellow-300 font-medium">Fallback triggered</div>
              <div className="text-[10px] text-yellow-400/80">The top-ranked model was throttled or unavailable. The router fell back to <span className="font-medium text-yellow-300">{actualModel}</span>.</div>
            </div>
          </div>
        )}

        {/* Step 1 */}
        <div className="mb-5">
          <div className="text-[10px] text-gray-500 uppercase tracking-wider mb-2 font-bold">Step 1: Complexity Classification</div>
          <div className="bg-gray-800/60 rounded-lg p-4">
            <div className="flex items-center gap-3 mb-3">
              <span className="text-xs text-gray-400">Final Score:</span>
              <span className="text-sm font-mono font-bold text-white">{cx.score?.toFixed(4)}</span>
              {cx.score_before_boost != null && cx.score_before_boost !== cx.score && <span className="text-[9px] text-gray-500">(base: {cx.score_before_boost?.toFixed(4)})</span>}
              {payload && <span className="text-[9px] text-yellow-400 bg-yellow-900/30 px-1.5 py-0.5 rounded">+{payload.complexity_boost} payload boost ({(payload.bytes/1024).toFixed(0)}KB)</span>}
              <span className="text-xs text-gray-400">&rarr;</span>
              <span className={`text-xs font-bold px-2 py-0.5 rounded ${cx.classification==='simple'?'bg-green-900/40 text-green-400':cx.classification==='moderate'?'bg-blue-900/40 text-blue-400':cx.classification==='complex'?'bg-purple-900/40 text-purple-400':'bg-red-900/40 text-red-400'}`}>{cx.classification?.toUpperCase()}</span>
            </div>
            <div className="mb-3">
              <div className="text-[9px] text-gray-500 mb-1">Classification Thresholds:</div>
              <table className="w-full text-[10px]"><tbody>
                {Object.entries(thresholds).map(([k,v])=>(<tr key={k} className={cx.classification===k?'text-orange-300 font-medium':'text-gray-500'}><td className="py-0.5 pr-3 capitalize w-24">{k}</td><td className="py-0.5 font-mono">{v}</td><td className="py-0.5 pl-2 text-right">{cx.classification===k?'◀ current':''}</td></tr>))}
              </tbody></table>
            </div>
            <div className="mb-3">
              <div className="text-[9px] text-gray-500 mb-1">Dimension Scores (15 dimensions, weighted sum = final score):</div>
              <div className="grid grid-cols-3 gap-x-4 gap-y-0.5 text-[10px]">
                {Object.entries(dimScores).map(([dim, score]) => (
                  <div key={dim} className="flex items-center gap-1">
                    <span className="text-gray-500 w-32 truncate">{dim.replace(/_/g,' ')}</span>
                    <div className="flex-1 h-1.5 bg-gray-700 rounded-full overflow-hidden">
                      <div className="h-full bg-orange-500 rounded-full" style={{width:`${Math.min(score*100,100)}%`}}/>
                    </div>
                    <span className="text-gray-400 font-mono w-8 text-right">{score?.toFixed(2)}</span>
                  </div>
                ))}
              </div>
            </div>
            {/* Score Breakdown: user message vs system prompt floor */}
            {cx.user_message_score != null && (
              <div className="mb-3">
                <div className="text-[9px] text-gray-500 mb-1">Score Breakdown:</div>
                <div className="flex items-center gap-4 text-[10px]">
                  <span className="text-gray-400">User Message Score: <span className="font-mono text-white">{cx.user_message_score?.toFixed(4)}</span></span>
                  <span className="text-gray-400">System Prompt Floor: <span className="font-mono text-white">{cx.system_prompt_floor?.toFixed(4)}</span></span>
                  <span className="text-gray-400">&rarr; Final: max(user_msg, floor) = <span className="font-mono font-bold text-white">{cx.score?.toFixed(4)}</span></span>
                  {cx.floor_applied && <span className="text-[9px] bg-orange-900/40 text-orange-300 px-1.5 py-0.5 rounded font-medium">⬆ Floor applied</span>}
                </div>
              </div>
            )}
            {/* System Floor Markers */}
            {cx.system_floor_markers && Object.keys(cx.system_floor_markers).length > 0 && (
              <div className="mb-3">
                <div className="text-[9px] text-gray-500 mb-1">System Prompt Floor Markers:</div>
                <div className="flex flex-wrap gap-1">
                  {Object.entries(cx.system_floor_markers).flatMap(([cat, markers]) => markers.map((m, i) => (
                    <span key={`sys-${cat}-${i}`} className="text-[9px] bg-teal-900/40 text-teal-300 px-1.5 py-0.5 rounded border border-teal-700/30"><span className="text-teal-400/70">System:</span> <span className="text-teal-500/70">{cat}:</span> {m}</span>
                  )))}
                </div>
              </div>
            )}
            <div>
              <div className="text-[9px] text-gray-500 mb-1">User Prompt Markers (Last Message):</div>
              <div className="flex flex-wrap gap-1">
                {Object.entries(cx.markers_hit||{}).filter(([,v])=>v&&v.length>0).flatMap(([cat,markers])=>markers.map((m,i)=>(<span key={`${cat}-${i}`} className="text-[9px] bg-gray-700 text-gray-300 px-1.5 py-0.5 rounded"><span className="text-orange-400/70">{cat}:</span> {m}</span>)))}
                {Object.values(cx.markers_hit||{}).every(v=>!v||v.length===0)&&<span className="text-[10px] text-gray-600 italic">No keyword markers detected</span>}
              </div>
            </div>
            {payload && <div className="mt-3 text-[10px] text-yellow-300 bg-yellow-900/20 border border-yellow-800/30 rounded p-2">
              📎 Document/Image attached ({(payload.bytes/1024).toFixed(0)}KB) added +{payload.complexity_boost} to complexity score.
            </div>}
          </div>
        </div>

        {/* Step 2 */}
        <div className="mb-5">
          <div className="text-[10px] text-gray-500 uppercase tracking-wider mb-2 font-bold">Step 2: Eligible Model Tiers</div>
          <div className="bg-gray-800/60 rounded-lg p-4">
            <div className="text-[10px] text-gray-400 mb-2">Classification <span className="font-bold text-white">{cx.classification?.toUpperCase()}</span> maps to tier range <span className="font-bold text-orange-300">{cx.tier_range?.min?.toUpperCase()}</span> → <span className="font-bold text-orange-300">{cx.tier_range?.max?.toUpperCase()}</span>:</div>
            <div className="flex items-center gap-2 mb-2">
              {tiers.map(t=>{const inRange=cx.tier_range&&tiers.indexOf(t)>=tiers.indexOf(cx.tier_range.min)&&tiers.indexOf(t)<=tiers.indexOf(cx.tier_range.max);return<span key={t} className={`text-[10px] px-3 py-1.5 rounded ${inRange?'bg-orange-600/30 text-orange-300 border border-orange-500/50 font-medium':'bg-gray-800 text-gray-600 border border-gray-700'}`}>{t.toUpperCase()}</span>})}
              <span className="text-[10px] text-gray-500 ml-3">&rarr; <span className="font-bold text-white">{explanation.candidates_evaluated}</span> of 65 models eligible</span>
            </div>
            <div className="text-[9px] text-gray-500 mt-2">Mapping: simple→MICRO-LITE | moderate→LITE-MID | complex→MID-HEAVY | reasoning→REASONING</div>
          </div>
        </div>

        {/* Step 3 */}
        <div className="mb-5">
          <div className="text-[10px] text-gray-500 uppercase tracking-wider mb-2 font-bold">Step 3: Strategy Scoring ({strat.name})</div>
          <div className="bg-gray-800/60 rounded-lg p-4">
            {strat.preferred_model && (
              <div className="mb-3 p-2.5 bg-blue-900/20 border border-blue-700/40 rounded-lg">
                <div className="text-[10px] text-blue-300 font-medium mb-1">🎯 Preferred Model Override</div>
                <div className="text-[9px] text-blue-400/80">
                  <span className="font-mono font-medium text-blue-300">{strat.preferred_model}</span> is used directly — strategy scoring below determines the <span className="font-medium text-blue-300">fallback chain</span> (used if preferred model is throttled/unavailable).
                </div>
              </div>
            )}
            {strat.weights&&<div className="flex gap-4 mb-3 text-[10px]"><span className="text-gray-400">Weights:</span><span className="text-blue-400 font-mono">cost={strat.weights.cost}</span><span className="text-green-400 font-mono">latency={strat.weights.latency}</span><span className="text-purple-400 font-mono">quality={strat.weights.quality}</span></div>}
            {!strat.weights&&!strat.preferred_model&&<div className="text-[10px] text-gray-400 mb-3">Single-dimension strategy: composite = <span className="font-mono text-white">{strat.name?.replace('-optimized','')}_score</span> only</div>}
            {!strat.weights&&strat.preferred_model&&<div className="text-[10px] text-gray-400 mb-3">Fallback ranking by: <span className="font-mono text-white">{strat.name?.replace('-optimized','')}_score</span></div>}
            <table className="w-full text-[10px]"><thead><tr className="text-gray-500 border-b border-gray-700">
              <th className="text-left py-1.5">Model</th>
              {strat.weights&&<th className="text-right py-1.5 group relative cursor-help">Composite <span className="invisible group-hover:visible absolute bottom-full right-0 mb-1 w-52 p-1.5 bg-gray-800 border border-gray-600 rounded text-[9px] text-gray-300 font-normal z-[200] shadow-lg">Weighted sum: cost×w + latency×w + quality×w</span></th>}
              <th className={`text-right py-1.5 group relative cursor-help ${!strat.weights && strat.name==='cost-optimized'?'text-orange-400 font-bold':''}`}>Cost ⓘ<span className="invisible group-hover:visible absolute bottom-full right-0 mb-1 w-56 p-1.5 bg-gray-800 border border-gray-600 rounded text-[9px] text-gray-300 font-normal z-[200] shadow-lg">min_cost / model_cost (ratio). Cheapest = 1.0, floor = 0.10.</span></th>
              <th className={`text-right py-1.5 group relative cursor-help ${!strat.weights && strat.name==='latency-optimized'?'text-orange-400 font-bold':''}`}>Latency ⓘ<span className="invisible group-hover:visible absolute bottom-full right-0 mb-1 w-56 p-1.5 bg-gray-800 border border-gray-600 rounded text-[9px] text-gray-300 font-normal z-[200] shadow-lg">fastest_latency / model_latency (ratio). Fastest = 1.0, floor = 0.10.</span></th>
              <th className={`text-right py-1.5 group relative cursor-help ${!strat.weights && strat.name==='quality-optimized'?'text-orange-400 font-bold':''}`}>Quality ⓘ<span className="invisible group-hover:visible absolute bottom-full right-0 mb-1 w-56 p-1.5 bg-gray-800 border border-gray-600 rounded text-[9px] text-gray-300 font-normal z-[200] shadow-lg">quality_baseline / 60 (AA Intelligence Index). Higher = smarter.</span></th>
            </tr></thead><tbody>
              {candidates.map((c,i)=>(<tr key={i} className={`${i===0?'text-orange-300 font-medium bg-orange-900/10': actualModel && c.model === actualModel ? 'text-green-300 font-medium bg-green-900/10' : 'text-gray-400'} border-b border-gray-800/50`}>
                <td className="py-1.5">{i===0 ? (fallbackUsed ? '⚠ ' : '★ ') : (actualModel && c.model === actualModel ? '✓ ' : '')}{c.model}{actualModel && c.model === actualModel && fallbackUsed ? ' (served)' : ''}</td>
                {strat.weights&&<td className="text-right py-1.5 font-mono">{c.composite?.toFixed(4)}</td>}
                <td className={`text-right py-1.5 font-mono ${!strat.weights && strat.name==='cost-optimized'?'text-orange-300 bg-orange-900/10':''}`}>{c.cost?.toFixed(4)}</td>
                <td className={`text-right py-1.5 font-mono ${!strat.weights && strat.name==='latency-optimized'?'text-orange-300 bg-orange-900/10':''}`}>{c.latency?.toFixed(4)}</td>
                <td className={`text-right py-1.5 font-mono ${!strat.weights && strat.name==='quality-optimized'?'text-orange-300 bg-orange-900/10':''}`}>{c.quality?.toFixed(4)}</td>
              </tr>))}
            </tbody></table>
          </div>
        </div>

        {explanation.reason && !explanation.hide_reason && <div className="text-xs text-gray-300 bg-gray-800/40 rounded-lg p-3 border-l-2 border-orange-500">{explanation.reason}</div>}
      </div>
    </>
  )
}

// ─── Accuracy Popup ────────────────────────────────────────────────
export function AccuracyPopup({ score, reasoning, side, position, onClose }) {
  return (
    <>
      <div className="fixed inset-0 z-[99]" onClick={onClose} />
      <div className="fixed z-[100] w-72 bg-gray-900 border border-gray-700 rounded-lg p-3 shadow-xl"
        style={{ top: position.y + 8, left: Math.min(position.x - 100, window.innerWidth - 300) }}>
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-medium text-gray-300">{side === 'baseline' ? 'Baseline' : 'Smart Router'} Accuracy</span>
          <span className={`text-lg font-bold ${score >= 8 ? 'text-green-400' : score >= 6 ? 'text-yellow-400' : 'text-red-400'}`}>{score}/10</span>
        </div>
        <p className="text-xs text-gray-400 leading-relaxed">{reasoning || 'No reasoning provided.'}</p>
        <div className="text-[9px] text-gray-600 mt-2">Scored by Claude Opus 4.7</div>
      </div>
    </>
  )
}
