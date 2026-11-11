import { useMemo, useState } from 'react'
import {
  Bar, BarChart, CartesianGrid, Cell, LabelList, Line, LineChart, Scatter, ScatterChart, XAxis, YAxis, ZAxis,
} from 'recharts'
import { ChartContainer, ChartLegend, ChartLegendContent, ChartTooltip, ChartTooltipContent, type ChartConfig } from '@/components/ui/chart'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import { HARDNESS, MODEL_COLOR, fmtInt, fmtMs, pct, results } from '@/lib/results'

const tall = 'aspect-auto h-[360px] w-full'

/* ---------- accuracy vs latency ---------- */
export function AccuracyLatencyChart() {
  const labelled = new Set(['teacher', 'base', 'merged'])  // the LoRA variants sit close together: name them in the tooltip only
  const data = results.models.map((m) => ({ id: m.id, name: m.short, tag: labelled.has(m.id) ? m.short : '', latency: m.p50_ms, ex: +(m.ex * 100).toFixed(1), fill: MODEL_COLOR[m.id] }))
  const config = { ex: { label: 'Execution accuracy' } } satisfies ChartConfig
  return (
    <ChartContainer config={config} className="aspect-auto h-[380px] w-full">
      <ScatterChart margin={{ top: 24, right: 40, bottom: 28, left: 8 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis type="number" dataKey="latency" scale="log" domain={[150, 1500]} ticks={[200, 300, 500, 1000]} tickFormatter={(v) => fmtMs(v)} name="Latency" label={{ value: 'median latency per question (log scale, lower is better) →', position: 'insideBottom', offset: -16, className: 'fill-muted-foreground text-xs' }} />
        <YAxis type="number" dataKey="ex" domain={[65, 80]} ticks={[65, 70, 75, 80]} tickFormatter={(v) => `${v}%`} name="EX" width={48} />
        <ZAxis range={[220, 220]} />
        <ChartTooltip
          cursor={{ strokeDasharray: '3 3' }}
          content={({ payload }) => {
            const p = payload?.[0]?.payload as (typeof data)[number] | undefined
            if (!p) return null
            return (
              <div className="rounded-lg border bg-background px-2.5 py-1.5 text-xs shadow-xl">
                <div className="font-medium">{p.name}</div>
                <div className="text-muted-foreground">accuracy {p.ex}% · latency {fmtMs(p.latency)}</div>
              </div>
            )
          }}
        />
        <Scatter isAnimationActive={false} data={data}>
          {data.map((d) => (
            <Cell key={d.id} fill={d.fill} />
          ))}
          <LabelList dataKey="tag" position="top" offset={10} className="fill-foreground text-xs font-medium" />
        </Scatter>
      </ScatterChart>
    </ChartContainer>
  )
}

/* ---------- accuracy by hardness ---------- */
export function HardnessChart() {
  const [selected, setSelected] = useState(['teacher', 'base', 'merged'])
  const shown = results.models.filter((m) => selected.includes(m.id))
  const config = Object.fromEntries(shown.map((m) => [m.id, { label: m.short, color: MODEL_COLOR[m.id] }])) satisfies ChartConfig
  const data = HARDNESS.map((h) => ({ hardness: h, ...Object.fromEntries(shown.map((m) => [m.id, +(m.hardness[h] * 100).toFixed(1)])) }))
  return (
    <div className="space-y-4">
      <ToggleGroup multiple value={selected} onValueChange={(v) => v.length && setSelected(v)} className="flex-wrap justify-start" variant="outline" size="sm">
        {results.models.map((m) => (
          <ToggleGroupItem key={m.id} value={m.id} className="gap-1.5">
            <span className="size-2 rounded-full" style={{ background: MODEL_COLOR[m.id] }} />
            {m.short}
          </ToggleGroupItem>
        ))}
      </ToggleGroup>
      <ChartContainer config={config} className={tall}>
        <BarChart data={data} margin={{ top: 20 }}>
          <CartesianGrid vertical={false} />
          <XAxis dataKey="hardness" tickFormatter={(v: string) => v[0].toUpperCase() + v.slice(1)} />
          <YAxis domain={[0, 100]} tickFormatter={(v) => `${v}%`} width={44} />
          <ChartTooltip content={<ChartTooltipContent />} />
          {shown.map((m) => (
            <Bar isAnimationActive={false} key={m.id} dataKey={m.id} fill={`var(--color-${m.id})`} radius={[4, 4, 0, 0]}>
              {shown.length <= 3 && <LabelList position="top" className="fill-foreground text-[11px]" />}
            </Bar>
          ))}
        </BarChart>
      </ChartContainer>
    </div>
  )
}

/* ---------- ablation 1: data source ---------- */
export function SourceAblationChart() {
  const order = ['base', 'human', 'teacher-data', 'combined']
  const rows = order.map((id) => results.models.find((m) => m.id === id)!)
  const base = rows[0].ex
  const data = rows.map((m) => ({ id: m.id, name: m.short, ex: +(m.ex * 100).toFixed(1), gain: m.id === 'base' ? '' : `+${((m.ex - base) * 100).toFixed(1)}`, fill: MODEL_COLOR[m.id] }))
  const config = { ex: { label: 'Execution accuracy' } } satisfies ChartConfig
  return (
    <ChartContainer config={config} className="aspect-auto h-[300px] w-full">
      <BarChart data={data} margin={{ top: 28 }}>
        <CartesianGrid vertical={false} />
        <XAxis dataKey="name" />
        <YAxis domain={[60, 80]} tickFormatter={(v) => `${v}%`} width={44} />
        <ChartTooltip content={<ChartTooltipContent hideLabel />} />
        <Bar isAnimationActive={false} dataKey="ex" radius={[6, 6, 0, 0]}>
          {data.map((d) => (
            <Cell key={d.id} fill={d.fill} />
          ))}
          <LabelList dataKey="ex" position="top" offset={16} formatter={(v: unknown) => `${v}%`} className="fill-foreground text-xs font-medium" />
          <LabelList dataKey="gain" position="top" offset={2} className="fill-muted-foreground text-[11px]" />
        </Bar>
      </BarChart>
    </ChartContainer>
  )
}

/* ---------- ablation 2: amount of teacher data ---------- */
export function SizeAblationChart() {
  const pcts = [25, 50, 100]
  const data = pcts.map((p) => {
    const t = results.ablation_size.find((a) => a.variant === 'teacher-only' && a.percent === p)!
    const c = results.ablation_size.find((a) => a.variant === 'human + teacher' && a.percent === p)!
    return { percent: `${p}%`, teacherOnly: +(t.ex * 100).toFixed(1), combined: +(c.ex * 100).toFixed(1) }
  })
  const config = {
    teacherOnly: { label: 'Teacher data only', color: 'var(--m-teacher-data)' },
    combined: { label: 'Human + teacher data', color: 'var(--m-combined)' },
  } satisfies ChartConfig
  return (
    <ChartContainer config={config} className={tall}>
      <LineChart data={data} margin={{ top: 16, right: 24, left: 8, bottom: 8 }}>
        <CartesianGrid vertical={false} />
        <XAxis dataKey="percent" label={{ value: 'share of the 23,703 teacher pairs used for training', position: 'insideBottom', offset: -2, className: 'fill-muted-foreground text-xs' }} height={52} />
        <YAxis domain={[72, 80]} tickFormatter={(v) => `${v}%`} width={44} />
        <ChartTooltip content={<ChartTooltipContent />} />
        <ChartLegend content={<ChartLegendContent />} verticalAlign="top" />
        <Line isAnimationActive={false} dataKey="teacherOnly" stroke="var(--color-teacherOnly)" strokeWidth={2.5} dot={{ r: 4 }}>
          <LabelList dataKey="teacherOnly" position="bottom" offset={10} className="fill-foreground text-[11px]" />
        </Line>
        <Line isAnimationActive={false} dataKey="combined" stroke="var(--color-combined)" strokeWidth={2.5} dot={{ r: 4 }}>
          <LabelList dataKey="combined" position="top" offset={10} className="fill-foreground text-[11px]" />
        </Line>
      </LineChart>
    </ChartContainer>
  )
}

/* ---------- CPU ---------- */
export function CpuChart() {
  const data = results.cpu.map((c) => ({
    name: c.label.replace(' (llama.cpp)', '').replace(' (transformers)', ''),
    seconds: +(c.p50_ms / 1000).toFixed(2),
    fill: c.model === 'teacher' ? 'var(--m-teacher)' : c.format === 'fp32' ? 'var(--m-base)' : 'var(--m-final)',
  }))
  const config = { seconds: { label: 'Median latency (s)' } } satisfies ChartConfig
  return (
    <ChartContainer config={config} className="aspect-auto h-[320px] w-full">
      <BarChart data={data} layout="vertical" margin={{ left: 8, right: 56, top: 8 }}>
        <CartesianGrid horizontal={false} />
        <YAxis dataKey="name" type="category" width={140} tickLine={false} axisLine={false} />
        <XAxis type="number" tickFormatter={(v) => `${v}s`} />
        <ChartTooltip content={<ChartTooltipContent hideLabel formatter={(v) => `${v} s per question`} />} />
        <Bar isAnimationActive={false} dataKey="seconds" radius={[0, 6, 6, 0]}>
          {data.map((d) => (
            <Cell key={d.name} fill={d.fill} />
          ))}
          <LabelList dataKey="seconds" position="right" formatter={(v: unknown) => `${v} s`} className="fill-foreground text-xs font-medium" />
        </Bar>
      </BarChart>
    </ChartContainer>
  )
}

/* ---------- loss curves ---------- */
export function LossChart() {
  const [run, setRun] = useState('combined')
  const r = results.training.find((t) => t.run === run)!
  const data = useMemo(() => {
    const byStep = new Map<number, { step: number; train?: number; val?: number }>()
    for (const p of r.train) byStep.set(p.step, { step: p.step, train: p.loss })
    for (const p of r.val) byStep.set(p.step, { ...(byStep.get(p.step) ?? { step: p.step }), val: p.loss })
    return [...byStep.values()].sort((a, b) => a.step - b.step)
  }, [r])
  const config = { train: { label: 'Training loss', color: 'var(--chart-2)' }, val: { label: 'Validation loss', color: 'var(--chart-1)' } } satisfies ChartConfig
  return (
    <div className="space-y-3">
      <Select value={run} onValueChange={(v) => v && setRun(v)}>
        <SelectTrigger className="w-64">
          <SelectValue>{r.label}</SelectValue>
        </SelectTrigger>
        <SelectContent>
          {results.training.map((t) => (
            <SelectItem key={t.run} value={t.run}>
              {t.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <ChartContainer config={config} className="aspect-auto h-[300px] w-full">
        <LineChart data={data} margin={{ top: 8, right: 16 }}>
          <CartesianGrid vertical={false} />
          <XAxis dataKey="step" type="number" domain={[0, 'dataMax']} tickFormatter={fmtInt} />
          <YAxis domain={[0, 'auto']} width={44} />
          <ChartTooltip content={<ChartTooltipContent />} />
          <Line isAnimationActive={false} dataKey="train" stroke="var(--color-train)" dot={false} strokeWidth={1.5} connectNulls />
          <Line isAnimationActive={false} dataKey="val" stroke="var(--color-val)" strokeWidth={2.5} dot={{ r: 3 }} connectNulls />
        </LineChart>
      </ChartContainer>
      <p className="text-xs text-muted-foreground">
        {fmtInt(r.train_examples)} examples · {fmtInt(r.steps)} steps · {r.minutes} min · best validation loss {r.best_val_loss}. Losses are measured on each run&apos;s own held-out set, so they are not comparable across variants.
      </p>
    </div>
  )
}

/* ---------- teacher data: filter funnel ---------- */
const STAGE_LABEL: Record<string, string> = {
  parse: 'parses as one SELECT',
  execute: 'runs without error',
  empty: 'non-empty result',
  self_consistency: 'self-consistent (≥3 of 5)',
  gold_check: 'matches Spider gold SQL',
  duplicate: 'deduplicated',
}

export function FunnelChart() {
  const [source, setSource] = useState<'synth' | 'relabel'>('synth')
  const stages = results.teacher_data.stages[source]
  const total = stages[0].in
  const data = [{ name: 'candidates', value: total }, ...stages.map((s) => ({ name: STAGE_LABEL[s.stage] ?? s.stage, value: s.passed }))].map((d) => ({ ...d, share: `${((d.value / total) * 100).toFixed(0)}%` }))
  const config = { value: { label: 'pairs', color: source === 'synth' ? 'var(--m-teacher-data)' : 'var(--m-human)' } } satisfies ChartConfig
  return (
    <div className="space-y-3">
      <ToggleGroup value={[source]} onValueChange={(v) => v[0] && setSource(v[0] as 'synth' | 'relabel')} variant="outline" size="sm">
        <ToggleGroupItem value="synth">Synthesised (new questions)</ToggleGroupItem>
        <ToggleGroupItem value="relabel">Relabelled (Spider questions)</ToggleGroupItem>
      </ToggleGroup>
      <ChartContainer config={config} className="aspect-auto h-[320px] w-full">
        <BarChart data={data} layout="vertical" margin={{ left: 8, right: 96 }}>
          <CartesianGrid horizontal={false} />
          <YAxis dataKey="name" type="category" width={170} tickLine={false} axisLine={false} />
          <XAxis type="number" hide />
          <ChartTooltip content={<ChartTooltipContent hideLabel />} />
          <Bar isAnimationActive={false} dataKey="value" fill="var(--color-value)" radius={[0, 6, 6, 0]}>
            <LabelList dataKey="value" position="right" formatter={(v: unknown) => fmtInt(Number(v))} className="fill-foreground text-xs" />
            <LabelList dataKey="share" position="right" offset={50} className="fill-muted-foreground text-[11px]" />
          </Bar>
        </BarChart>
      </ChartContainer>
    </div>
  )
}

/* ---------- teacher data: composition ---------- */
export function CompositionChart() {
  const pattern = Object.entries(results.teacher_data.kept_by_pattern).sort((a, b) => b[1] - a[1]).map(([name, value]) => ({ name: name === 'relabel' ? 'relabelled (Spider)' : name.replace('_', ' '), value }))
  const config = { value: { label: 'pairs', color: 'var(--m-combined)' } } satisfies ChartConfig
  return (
    <ChartContainer config={config} className="aspect-auto h-[300px] w-full">
      <BarChart data={pattern} layout="vertical" margin={{ left: 8, right: 48 }}>
        <CartesianGrid horizontal={false} />
        <YAxis dataKey="name" type="category" width={100} tickLine={false} axisLine={false} />
        <XAxis type="number" hide />
        <ChartTooltip content={<ChartTooltipContent hideLabel />} />
        <Bar isAnimationActive={false} dataKey="value" fill="var(--color-value)" radius={[0, 6, 6, 0]}>
          <LabelList dataKey="value" position="right" formatter={(v: unknown) => fmtInt(Number(v))} className="fill-foreground text-xs" />
        </Bar>
      </BarChart>
    </ChartContainer>
  )
}

/* ---------- error analysis ---------- */
export function ErrorChart() {
  const data = [...results.errors].sort((a, b) => b.count - a.count)
  const config = { count: { label: 'failures', color: 'var(--chart-5)' } } satisfies ChartConfig
  return (
    <ChartContainer config={config} className="aspect-auto h-[320px] w-full">
      <BarChart data={data} layout="vertical" margin={{ left: 8, right: 40 }}>
        <CartesianGrid horizontal={false} />
        <YAxis dataKey="category" type="category" width={170} tickLine={false} axisLine={false} />
        <XAxis type="number" hide />
        <ChartTooltip content={<ChartTooltipContent hideLabel />} />
        <Bar isAnimationActive={false} dataKey="count" fill="var(--color-count)" radius={[0, 6, 6, 0]}>
          <LabelList dataKey="count" position="right" formatter={(v: unknown) => `${v}%`} className="fill-foreground text-xs font-medium" />
        </Bar>
      </BarChart>
    </ChartContainer>
  )
}

/* ---------- student vs teacher, question by question ---------- */
export function PairBar() {
  const p = results.student_vs_teacher
  const total = p.both_correct + p.student_only + p.teacher_only + p.both_wrong
  const parts = [
    { key: 'both_correct', label: 'Both correct', n: p.both_correct, color: 'var(--m-final)' },
    { key: 'student_only', label: 'Only the student', n: p.student_only, color: 'var(--m-combined)' },
    { key: 'teacher_only', label: 'Only the teacher', n: p.teacher_only, color: 'var(--m-teacher)' },
    { key: 'both_wrong', label: 'Both wrong', n: p.both_wrong, color: 'var(--m-base)' },
  ]
  return (
    <div className="space-y-3">
      <div className="flex h-9 w-full overflow-hidden rounded-lg">
        {parts.map((s) => (
          <div key={s.key} className="grid place-items-center text-xs font-medium text-white" style={{ width: `${(s.n / total) * 100}%`, background: s.color }} title={`${s.label}: ${s.n}`}>
            {s.n / total > 0.06 ? s.n : ''}
          </div>
        ))}
      </div>
      <div className="grid grid-cols-2 gap-x-6 gap-y-2 text-sm sm:grid-cols-4">
        {parts.map((s) => (
          <div key={s.key} className="flex items-center gap-2">
            <span className="size-2.5 rounded-full" style={{ background: s.color }} />
            <span>
              {s.label} <span className="text-muted-foreground tabular-nums">· {s.n} ({pct(s.n / total, 0)}%)</span>
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}
