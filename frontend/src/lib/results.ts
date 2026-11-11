import raw from '@/content/results.json'

export type Hardness = 'easy' | 'medium' | 'hard' | 'extra'
export const HARDNESS: Hardness[] = ['easy', 'medium', 'hard', 'extra']

export interface RunSummary {
  run: string
  n: number
  ex: number
  hardness: Record<Hardness, number>
  p50_ms: number
  p95_ms: number
  mean_ms: number
  qps: number | null
  vram_gib: number | null
  size_gib: number
}
export interface ModelRow extends RunSummary {
  id: string
  label: string
  short: string
  kind: 'teacher' | 'base' | 'student' | 'final'
}
export interface CpuRow extends RunSummary {
  label: string
  model: 'student' | 'teacher'
  format: string
  runtime: string
}
export interface Results {
  benchmark: { name: string; questions: number; databases: number; metric: string }
  headline: Record<string, number>
  models: ModelRow[]
  cpu: CpuRow[]
  ablation_size: { variant: string; percent: number; ex: number; hardness: Record<Hardness, number>; train_examples: number }[]
  training: {
    run: string
    label: string
    train_examples: number
    steps: number
    minutes: number
    best_val_loss: number
    peak_gpu_gib: number
    by_source: Record<string, number>
    train: { step: number; loss: number }[]
    val: { step: number; loss: number }[]
  }[]
  teacher_data: {
    kept: number
    kept_by_source: Record<string, number>
    kept_by_pattern: Record<string, number>
    kept_by_hardness: Record<string, number>
    databases: number
    stages: Record<'synth' | 'relabel', { stage: string; in: number; passed: number }[]>
  }
  errors: { category: string; count: number }[]
  student_vs_teacher: { both_correct: number; student_only: number; teacher_only: number; both_wrong: number }
  export_check: Record<string, unknown>
}

export const results = raw as unknown as Results

/** CSS colour for each model on every chart. */
export const MODEL_COLOR: Record<string, string> = {
  teacher: 'var(--m-teacher)',
  'teacher-3shot': 'var(--m-teacher)',
  base: 'var(--m-base)',
  human: 'var(--m-human)',
  'teacher-data': 'var(--m-teacher-data)',
  combined: 'var(--m-combined)',
  merged: 'var(--m-final)',
}

export const pct = (x: number, d = 1) => (x * 100).toFixed(d)
export const fmtMs = (ms: number) => (ms >= 1000 ? `${(ms / 1000).toFixed(ms >= 10000 ? 1 : 2)} s` : `${Math.round(ms)} ms`)
export const fmtInt = (n: number) => n.toLocaleString('en-US')
export const fmtGib = (g: number | null) => (g === null ? '–' : `${g.toFixed(g < 10 ? 2 : 1)} GiB`)
