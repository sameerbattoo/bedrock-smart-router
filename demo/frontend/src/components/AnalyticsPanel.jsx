import { useState, useRef, useCallback } from 'react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, LineChart, Line, CartesianGrid, ScatterChart, Scatter, Cell, PieChart, Pie, Legend } from 'recharts'
import { StatCard } from './shared'

export default function AnalyticsPanel({ history }) {
  const [expanded, setExpanded] = useState(false)
  const [animKey, setAnimKey] = useState(0)
  const MIN_WIDTH = 256 // 16rem = w-64
  const MAX_WIDTH = 384 // 1.5x
  const [panelWidth, setPanelWidth] = useState(MIN_WIDTH)
  const dragging = useRef(false)
  const startX = useRef(0)
  const startWidth = useRef(MIN_WIDTH)

  const onMouseDown = useCallback((e) => {
    dragging.current = true
    startX.current = e.clientX
    startWidth.current = panelWidth
    document.body.style.cursor = 'col-resize'
    document.body.style.userSelect = 'none'

    const onMouseMove = (e) => {
      if (!dragging.current) return
      const delta = startX.current - e.clientX // dragging left = increase width
      const newWidth = Math.min(MAX_WIDTH, Math.max(MIN_WIDTH, startWidth.current + delta))
      setPanelWidth(newWidth)
    }
    const onMouseUp = () => {
      dragging.current = false
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
      document.removeEventListener('mousemove', onMouseMove)
      document.removeEventListener('mouseup', onMouseUp)
    }
    document.addEventListener('mousemove', onMouseMove)
    document.addEventListener('mouseup', onMouseUp)
  }, [panelWidth])

  const totalRuns = history.length
  // Only use successful runs (no errors on either side) for metric calculations
  const validRuns = history.filter(h => !h.has_error)
  const validCount = validRuns.length
  const totalBaselineCost = validRuns.reduce((s,h) => s + (h.baseline_cost||0), 0)
  const totalRouterCost = validRuns.reduce((s,h) => s + (h.router_cost||0), 0)
  const cumulativeSavingsPct = totalBaselineCost > 0 ? ((1 - totalRouterCost / totalBaselineCost) * 100).toFixed(1) : '—'
  const avgBaselineLatency = validCount > 0 ? (validRuns.reduce((s,h) => s + (h.baseline_latency||0), 0) / validCount).toFixed(0) : '—'
  const avgRouterLatency = validCount > 0 ? (validRuns.reduce((s,h) => s + (h.router_latency||0), 0) / validCount).toFixed(0) : '—'
  const avgLatencyImprovement = validCount > 0 && avgBaselineLatency > 0 ? ((1 - avgRouterLatency / avgBaselineLatency) * 100).toFixed(1) : '—'
  const scoredRuns = validRuns.filter(h => h.baseline_score && h.router_score)
  const avgBaselineScore = scoredRuns.length > 0 ? (scoredRuns.reduce((s,h) => s + h.baseline_score, 0) / scoredRuns.length).toFixed(1) : '—'
  const avgRouterScore = scoredRuns.length > 0 ? (scoredRuns.reduce((s,h) => s + h.router_score, 0) / scoredRuns.length).toFixed(1) : '—'
  const avgAccuracyDelta = scoredRuns.length > 0 ? ((avgRouterScore - avgBaselineScore)).toFixed(1) : '—'

  const chartData = validRuns.slice(-20).map((h, i) => ({ ...h, runNum: i + 1 }))
  const tooltipStyle = {background:'#1f2937',border:'1px solid #374151',borderRadius:8,fontSize:11}

  return (
    <>
      {expanded && <div className="fixed inset-0 z-[95] bg-black/60" onClick={() => setExpanded(false)} />}
      <div className={`${expanded ? 'fixed inset-0 z-[96] bg-[#0a0e17] m-0' : 'border-l border-gray-800/50 bg-[#0b1018] relative'} flex flex-col overflow-y-auto transition-all duration-300`}
        style={expanded ? {} : { width: panelWidth }}>
        {/* Drag handle (left edge) */}
        {!expanded && <div onMouseDown={onMouseDown} className="absolute left-0 top-0 bottom-0 w-1.5 cursor-col-resize hover:bg-orange-500/30 active:bg-orange-500/50 transition-colors z-10" />}
      <div className="flex items-center justify-between px-3 pt-3 pb-1">
        <div className="text-[10px] text-gray-400 font-bold uppercase tracking-wider">{expanded ? 'Session Scorecard' : 'Session Scorecard'}</div>
        <div className="flex items-center gap-2">
          <button onClick={() => { setExpanded(!expanded); setAnimKey(k => k + 1) }} className="text-gray-500 hover:text-orange-400 p-1 rounded hover:bg-gray-800">
          {expanded
            ? <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12"/></svg>
            : <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4"/></svg>}
        </button>
        </div>
      </div>

      {/* Stats */}
      <div className={`p-3 border-b border-gray-800/50 ${expanded ? 'px-8 py-6' : ''}`}>
        <div className={`grid ${expanded ? 'grid-cols-4' : 'grid-cols-2'} gap-2`}>
          <div className="bg-gray-900/60 border border-gray-800/60 rounded-xl p-3 text-center hover:border-gray-600 hover:shadow-lg hover:shadow-gray-900/30 hover:-translate-y-1.5 transition-all duration-200">
            <div className="text-[10px] text-gray-500 uppercase tracking-wider mb-1">Runs</div>
            <div className={`${expanded ? 'text-3xl' : 'text-xl'} font-bold text-white`}>{validCount}</div>
          </div>
          {/* Cost Savings card */}
          <div className="bg-gray-900/60 border border-gray-800/60 rounded-xl p-3 text-center hover:border-gray-600 hover:shadow-lg hover:shadow-gray-900/30 hover:-translate-y-1.5 transition-all duration-200">
            <div className="text-[10px] text-gray-500 uppercase tracking-wider mb-1">Cost Savings</div>
            <div className={`${expanded ? 'text-3xl' : 'text-xl'} font-bold text-green-400`}>{cumulativeSavingsPct}%</div>
            {expanded && totalBaselineCost > 0 && <div className="text-sm text-green-300 mt-0.5">Saved ${(totalBaselineCost - totalRouterCost).toFixed(4)} for {validCount} runs</div>}
            <div className={`flex justify-center gap-3 mt-1.5 ${expanded ? 'text-xs' : 'text-[9px]'}`}>
              <span className="text-blue-400">{expanded ? 'Baseline' : 'B'}: ${totalBaselineCost.toFixed(4)}</span>
              <span className="text-orange-400">{expanded ? 'Smart Router' : 'R'}: ${totalRouterCost.toFixed(4)}</span>
            </div>
          </div>
          {/* Latency Improvement card */}
          <div className="bg-gray-900/60 border border-gray-800/60 rounded-xl p-3 text-center hover:border-gray-600 hover:shadow-lg hover:shadow-gray-900/30 hover:-translate-y-1.5 transition-all duration-200">
            <div className="text-[10px] text-gray-500 uppercase tracking-wider mb-1">Avg Latency</div>
            <div className={`${expanded ? 'text-3xl' : 'text-xl'} font-bold ${Number(avgLatencyImprovement) > 0 ? 'text-green-400' : Number(avgLatencyImprovement) < 0 ? 'text-red-400' : 'text-gray-300'}`}>{avgLatencyImprovement !== '—' ? `${avgLatencyImprovement}%` : '—'}</div>
            {expanded && validCount > 0 && (() => { const totalSavedMs = validRuns.reduce((s,h) => s + ((h.baseline_latency||0) - (h.router_latency||0)), 0); return totalSavedMs > 0 ? <div className="text-sm text-green-300 mt-0.5">Saved {(totalSavedMs/1000).toFixed(1)}s for {validCount} runs</div> : null })()}
            <div className={`flex justify-center gap-3 mt-1.5 ${expanded ? 'text-xs' : 'text-[9px]'}`}>
              <span className="text-blue-400">{expanded ? 'Baseline' : 'B'}: {(avgBaselineLatency/1000).toFixed(1)}s</span>
              <span className="text-orange-400">{expanded ? 'Smart Router' : 'R'}: {(avgRouterLatency/1000).toFixed(1)}s</span>
            </div>
          </div>
          {/* Accuracy card */}
          <div className="bg-gray-900/60 border border-gray-800/60 rounded-xl p-3 text-center hover:border-gray-600 hover:shadow-lg hover:shadow-gray-900/30 hover:-translate-y-1.5 transition-all duration-200">
            <div className="text-[10px] text-gray-500 uppercase tracking-wider mb-1">Avg Accuracy</div>
            <div className={`${expanded ? 'text-3xl' : 'text-xl'} font-bold ${Number(avgAccuracyDelta) > 0 ? 'text-green-400' : Number(avgAccuracyDelta) < 0 ? 'text-red-400' : 'text-orange-400'}`}>{avgAccuracyDelta !== '—' ? `${Number(avgAccuracyDelta) >= 0 ? '+' : ''}${avgAccuracyDelta}` : '—'}</div>
            <div className={`flex justify-center gap-3 mt-1.5 ${expanded ? 'text-xs' : 'text-[9px]'}`}>
              <span className="text-blue-400">{expanded ? 'Baseline' : 'B'}: {avgBaselineScore}/10</span>
              <span className="text-orange-400">{expanded ? 'Smart Router' : 'R'}: {avgRouterScore}/10</span>
            </div>
          </div>
        </div>
      </div>

      {/* Charts */}
      {history.length > 0 && (
        <div key={animKey} className={`${expanded ? 'flex-1 p-6 overflow-y-auto' : 'flex flex-col'}`}>
          {/* ─── Collapsed view (sidebar) ─── */}
          {!expanded && <>
            <div className="p-3 border-b border-gray-800/50 hover:bg-gray-800/20 hover:-translate-y-0.5 transition-all duration-200 rounded-lg">
              <div className="text-[10px] text-gray-400 font-bold uppercase tracking-wider mb-2">Latency (ms)</div>
              <ResponsiveContainer width="100%" height={90}>
                <BarChart data={chartData} barGap={1}><XAxis dataKey="runNum" tick={{fontSize:9,fill:'#6b7280'}} label={{value:"Runs",position:"bottom",fontSize:9,fill:"#6b7280",offset:-5}}/><YAxis tick={{fontSize:9,fill:'#6b7280'}} width={35}/><Tooltip cursor={false} contentStyle={tooltipStyle}/><Bar dataKey="baseline_latency" fill="#3b82f6" name="Baseline" radius={[2,2,0,0]} animationDuration={800} animationBegin={100}/><Bar dataKey="router_latency" fill="#f97316" name="Router" radius={[2,2,0,0]} animationDuration={800} animationBegin={300}/></BarChart>
              </ResponsiveContainer>
            </div>
            <div className="p-3 border-b border-gray-800/50 hover:bg-gray-800/20 hover:-translate-y-0.5 transition-all duration-200 rounded-lg">
              <div className="text-[10px] text-gray-400 font-bold uppercase tracking-wider mb-2">Cost ($)</div>
              <ResponsiveContainer width="100%" height={90}>
                <BarChart data={chartData} barGap={1}><XAxis dataKey="runNum" tick={{fontSize:9,fill:'#6b7280'}} label={{value:"Runs",position:"bottom",fontSize:9,fill:"#6b7280",offset:-5}}/><YAxis tick={{fontSize:9,fill:'#6b7280'}} width={40} tickFormatter={v=>`$${Number(v).toFixed(4)}`}/><Tooltip cursor={false} contentStyle={tooltipStyle} formatter={v=>`$${Number(v).toFixed(6)}`}/><Bar dataKey="baseline_cost" fill="#3b82f6" name="Baseline" radius={[2,2,0,0]} animationDuration={800} animationBegin={100}/><Bar dataKey="router_cost" fill="#f97316" name="Router" radius={[2,2,0,0]} animationDuration={800} animationBegin={300}/></BarChart>
              </ResponsiveContainer>
            </div>
            {history.some(h=>h.baseline_score) && <div className="p-3 border-b border-gray-800/50 hover:bg-gray-800/20 hover:-translate-y-0.5 transition-all duration-200 rounded-lg">
              <div className="text-[10px] text-gray-400 font-bold uppercase tracking-wider mb-2">Accuracy (0-10) based on LLM judge</div>
              <ResponsiveContainer width="100%" height={90}>
                <LineChart data={history.filter(h=>h.baseline_score).slice(-20).map((h,i)=>({...h,runNum:i+1}))}><CartesianGrid strokeDasharray="3 3" stroke="#1f2937"/><XAxis dataKey="runNum" tick={{fontSize:9,fill:'#6b7280'}} label={{value:"Runs",position:"bottom",fontSize:9,fill:"#6b7280",offset:-5}}/><YAxis domain={[0,10]} tick={{fontSize:9,fill:'#6b7280'}} width={25}/><Tooltip cursor={false} contentStyle={tooltipStyle}/><Line type="monotone" dataKey="baseline_score" stroke="#3b82f6" name="Baseline" strokeWidth={2} dot={{r:3}} animationDuration={1000} animationBegin={100}/><Line type="monotone" dataKey="router_score" stroke="#f97316" name="Router" strokeWidth={2} dot={{r:3}} animationDuration={1000} animationBegin={300}/></LineChart>
              </ResponsiveContainer>
            </div>}
          </>}

          {/* ─── Expanded view (full page grid) ─── */}
          {expanded && <>
            {/* Row 2: Scatter charts */}
            <div className="grid grid-cols-2 gap-4 mb-4">
              {history.some(h=>h.baseline_score) && <div className="rounded-lg p-3 border border-transparent hover:border-gray-700 hover:bg-gray-800/20 hover:-translate-y-1 hover:shadow-lg transition-all duration-200">
                <div className="text-[10px] text-gray-400 font-bold uppercase tracking-wider mb-2">Cost vs Accuracy (Pareto)</div>
                <ResponsiveContainer width="100%" height={220}>
                  <ScatterChart margin={{top:5,right:5,bottom:5,left:5}}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1f2937"/>
                    <XAxis type="number" dataKey="cost" name="Cost" tick={{fontSize:9,fill:'#6b7280'}} tickFormatter={v=>`$${v.toFixed(4)}`} label={{value:"Cost ($)",position:"bottom",fontSize:9,fill:"#6b7280",offset:-5}}/>
                    <YAxis type="number" dataKey="score" name="Accuracy" domain={[0,10]} tick={{fontSize:9,fill:'#6b7280'}} width={35} label={{value:'Accuracy Score',angle:-90,position:'insideLeft',fontSize:9,fill:'#6b7280'}}/>
                    <Legend wrapperStyle={{fontSize:10}} align="right" verticalAlign="top" />
                    <Tooltip cursor={false} contentStyle={tooltipStyle} content={({payload})=>{if(!payload||!payload.length)return null;const d=payload[0].payload;return<div style={{...tooltipStyle,padding:8}}><div style={{fontSize:10,color:'#9ca3af'}}>Run #{d.runNum} • {d.model}</div><div style={{fontSize:11}}><span style={{color:'#6b7280'}}>Cost:</span> ${d.cost?.toFixed(6)} <span style={{color:'#6b7280'}}>Score:</span> {d.score}/10</div></div>}}/>
                    <Scatter name="Baseline" data={history.filter(h=>h.baseline_score).map((h,i)=>({cost:h.baseline_cost,score:h.baseline_score,runNum:i+1,model:h.baseline_metrics?.model_used||'Baseline'}))} fill="#3b82f6" r={5} animationDuration={800}/>
                    <Scatter name="Smart Router" data={history.filter(h=>h.router_score).map((h,i)=>({cost:h.router_cost,score:h.router_score,runNum:i+1,model:h.router_model||'Router'}))} fill="#f97316" r={5} animationDuration={800} animationBegin={200}/>
                  </ScatterChart>
                </ResponsiveContainer>
              </div>}
              {history.some(h=>h.baseline_score) && <div className="rounded-lg p-3 border border-transparent hover:border-gray-700 hover:bg-gray-800/20 hover:-translate-y-1 hover:shadow-lg transition-all duration-200">
                <div className="text-[10px] text-gray-400 font-bold uppercase tracking-wider mb-2">Latency vs Accuracy (Pareto)</div>
                <ResponsiveContainer width="100%" height={220}>
                  <ScatterChart margin={{top:5,right:5,bottom:5,left:5}}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1f2937"/>
                    <XAxis type="number" dataKey="latency" name="Latency" tick={{fontSize:9,fill:'#6b7280'}} tickFormatter={v=>`${v}ms`} label={{value:"Latency (ms)",position:"bottom",fontSize:9,fill:"#6b7280",offset:-5}}/>
                    <YAxis type="number" dataKey="score" name="Accuracy" domain={[0,10]} tick={{fontSize:9,fill:'#6b7280'}} width={35} label={{value:'Accuracy Score',angle:-90,position:'insideLeft',fontSize:9,fill:'#6b7280'}}/>
                    <Legend wrapperStyle={{fontSize:10}} align="right" verticalAlign="top" />
                    <Tooltip cursor={false} contentStyle={tooltipStyle} content={({payload})=>{if(!payload||!payload.length)return null;const d=payload[0].payload;return<div style={{...tooltipStyle,padding:8}}><div style={{fontSize:10,color:'#9ca3af'}}>Run #{d.runNum} • {d.model}</div><div style={{fontSize:11}}><span style={{color:'#6b7280'}}>Latency:</span> {d.latency?.toFixed(0)}ms <span style={{color:'#6b7280'}}>Score:</span> {d.score}/10</div></div>}}/>
                    <Scatter name="Baseline" data={history.filter(h=>h.baseline_score).map((h,i)=>({latency:h.baseline_latency,score:h.baseline_score,runNum:i+1,model:h.baseline_metrics?.model_used||'Baseline'}))} fill="#3b82f6" r={5} animationDuration={800}/>
                    <Scatter name="Smart Router" data={history.filter(h=>h.router_score).map((h,i)=>({latency:h.router_latency,score:h.router_score,runNum:i+1,model:h.router_model||'Router'}))} fill="#f97316" r={5} animationDuration={800} animationBegin={200}/>
                  </ScatterChart>
                </ResponsiveContainer>
              </div>}
            </div>

            {/* Row 3: Latency, TTFT, Accuracy */}
            <div className="grid grid-cols-3 gap-4 mb-4">
              <div className="rounded-lg p-3 border border-transparent hover:border-gray-700 hover:bg-gray-800/20 hover:-translate-y-1 hover:shadow-lg transition-all duration-200">
                <div className="text-[10px] text-gray-400 font-bold uppercase tracking-wider mb-2">Latency (ms)</div>
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={chartData} barGap={1}><XAxis dataKey="runNum" tick={{fontSize:9,fill:'#6b7280'}} label={{value:"Runs",position:"bottom",fontSize:9,fill:"#6b7280",offset:-5}}/><YAxis tick={{fontSize:9,fill:'#6b7280'}} width={35}/><Tooltip cursor={false} contentStyle={tooltipStyle}/><Bar dataKey="baseline_latency" fill="#3b82f6" name="Baseline" radius={[2,2,0,0]} animationDuration={800} animationBegin={100}/><Bar dataKey="router_latency" fill="#f97316" name="Router" radius={[2,2,0,0]} animationDuration={800} animationBegin={300}/></BarChart>
                </ResponsiveContainer>
              </div>
              <div className="rounded-lg p-3 border border-transparent hover:border-gray-700 hover:bg-gray-800/20 hover:-translate-y-1 hover:shadow-lg transition-all duration-200">
                <div className="text-[10px] text-gray-400 font-bold uppercase tracking-wider mb-2">TTFT (ms)</div>
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={chartData} barGap={1}><XAxis dataKey="runNum" tick={{fontSize:9,fill:'#6b7280'}} label={{value:"Runs",position:"bottom",fontSize:9,fill:"#6b7280",offset:-5}}/><YAxis tick={{fontSize:9,fill:'#6b7280'}} width={35}/><Tooltip cursor={false} contentStyle={tooltipStyle}/><Bar dataKey="baseline_ttft" fill="#3b82f6" name="Baseline" radius={[2,2,0,0]} animationDuration={800} animationBegin={100}/><Bar dataKey="router_ttft" fill="#f97316" name="Router" radius={[2,2,0,0]} animationDuration={800} animationBegin={300}/></BarChart>
                </ResponsiveContainer>
              </div>
              {history.some(h=>h.baseline_score) && <div className="rounded-lg p-3 border border-transparent hover:border-gray-700 hover:bg-gray-800/20 hover:-translate-y-1 hover:shadow-lg transition-all duration-200">
                <div className="text-[10px] text-gray-400 font-bold uppercase tracking-wider mb-2">Accuracy (0-10) based on LLM judge</div>
                <ResponsiveContainer width="100%" height={200}>
                  <LineChart data={history.filter(h=>h.baseline_score).slice(-20).map((h,i)=>({...h,runNum:i+1}))}><CartesianGrid strokeDasharray="3 3" stroke="#1f2937"/><XAxis dataKey="runNum" tick={{fontSize:9,fill:'#6b7280'}} label={{value:"Runs",position:"bottom",fontSize:9,fill:"#6b7280",offset:-5}}/><YAxis domain={[0,10]} tick={{fontSize:9,fill:'#6b7280'}} width={25}/><Tooltip cursor={false} contentStyle={tooltipStyle}/><Line type="monotone" dataKey="baseline_score" stroke="#3b82f6" name="Baseline" strokeWidth={2} dot={{r:3}} animationDuration={1000} animationBegin={100}/><Line type="monotone" dataKey="router_score" stroke="#f97316" name="Router" strokeWidth={2} dot={{r:3}} animationDuration={1000} animationBegin={300}/></LineChart>
                </ResponsiveContainer>
              </div>}
            </div>

            {/* Row 4: Cost, Savings %, Model Distribution */}
            <div className="grid grid-cols-3 gap-4">
              <div className="rounded-lg p-3 border border-transparent hover:border-gray-700 hover:bg-gray-800/20 hover:-translate-y-1 hover:shadow-lg transition-all duration-200">
                <div className="text-[10px] text-gray-400 font-bold uppercase tracking-wider mb-2">Cost ($)</div>
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={chartData} barGap={1}><XAxis dataKey="runNum" tick={{fontSize:9,fill:'#6b7280'}} label={{value:"Runs",position:"bottom",fontSize:9,fill:"#6b7280",offset:-5}}/><YAxis tick={{fontSize:9,fill:'#6b7280'}} width={40} tickFormatter={v=>`$${Number(v).toFixed(4)}`}/><Tooltip cursor={false} contentStyle={tooltipStyle} formatter={v=>`$${Number(v).toFixed(6)}`}/><Bar dataKey="baseline_cost" fill="#3b82f6" name="Baseline" radius={[2,2,0,0]} animationDuration={800} animationBegin={100}/><Bar dataKey="router_cost" fill="#f97316" name="Router" radius={[2,2,0,0]} animationDuration={800} animationBegin={300}/></BarChart>
                </ResponsiveContainer>
              </div>
              <div className="rounded-lg p-3 border border-transparent hover:border-gray-700 hover:bg-gray-800/20 hover:-translate-y-1 hover:shadow-lg transition-all duration-200">
                <div className="text-[10px] text-gray-400 font-bold uppercase tracking-wider mb-2">Savings % per Run</div>
                <ResponsiveContainer width="100%" height={200}>
                  <LineChart data={history.map((h,i)=>({...h,runNum:i+1}))}><CartesianGrid strokeDasharray="3 3" stroke="#1f2937"/><XAxis dataKey="runNum" tick={{fontSize:9,fill:'#6b7280'}} label={{value:"Runs",position:"bottom",fontSize:9,fill:"#6b7280",offset:-5}}/><YAxis tick={{fontSize:9,fill:'#6b7280'}} width={35} tickFormatter={v=>`${v}%`}/><Tooltip cursor={false} contentStyle={tooltipStyle} formatter={v=>`${Number(v).toFixed(1)}%`}/><Line type="monotone" dataKey="savings_pct" stroke="#22c55e" name="Savings" strokeWidth={2} dot={{r:3}} animationDuration={1000} animationBegin={100}/></LineChart>
                </ResponsiveContainer>
              </div>
              <div className="rounded-lg p-3 border border-transparent hover:border-gray-700 hover:bg-gray-800/20 hover:-translate-y-1 hover:shadow-lg transition-all duration-200">
                <div className="text-[10px] text-gray-400 font-bold uppercase tracking-wider mb-2">Smart Router Model Distribution</div>
                <ResponsiveContainer width="100%" height={250}>
                  <PieChart>
                    <Pie data={Object.entries(history.reduce((acc,h)=>{acc[h.router_model]=(acc[h.router_model]||0)+1;return acc},{})).map(([name,value])=>({name,value}))} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label={({name,percent})=>`${name?.split(' ').slice(0,2).join(' ')} ${(percent*100).toFixed(0)}%`} labelLine={true} animationDuration={1000} animationBegin={200} fontSize={10}>
                      {Object.keys(history.reduce((acc,h)=>{acc[h.router_model]=1;return acc},{})).map((_,i)=>(
                        <Cell key={i} fill={['#f97316','#22c55e','#3b82f6','#a855f7','#eab308','#ec4899'][i%6]}/>
                      ))}
                    </Pie>
                    <Tooltip contentStyle={tooltipStyle}/>
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </div>
          </>}
        </div>
      )}
    </div>
    </>
  )
}
