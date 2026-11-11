import type { ReactNode } from 'react'
import { AlertTriangle, Lightbulb } from 'lucide-react'
import { useSearchParams } from 'react-router-dom'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import {
  AccuracyLatencyChart, CompositionChart, CpuChart, ErrorChart, FunnelChart, HardnessChart, LossChart, PairBar, SizeAblationChart, SourceAblationChart,
} from '@/components/results/Charts'
import { ModelTable } from '@/components/results/ModelTable'
import { StatCard } from '@/components/results/StatCard'
import { fmtGib, fmtInt, fmtMs, pct, results } from '@/lib/results'

function Section({ title, description, children, className }: { title: string; description?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
        {description && <CardDescription>{description}</CardDescription>}
      </CardHeader>
      <CardContent className="space-y-4">{children}</CardContent>
    </Card>
  )
}

function Insight({ children }: { children: ReactNode }) {
  return (
    <div className="flex gap-2.5 rounded-lg bg-muted/60 px-3 py-2.5 text-sm">
      <Lightbulb className="mt-0.5 size-4 shrink-0 text-[var(--m-teacher)]" />
      <div className="space-y-1 leading-relaxed">{children}</div>
    </div>
  )
}

const TABS = ['overview', 'ablations', 'cpu', 'data', 'errors']

export default function Results() {
  const h = results.headline
  const cpuStudent = results.cpu.find((c) => c.model === 'student' && c.format === 'Q4_K_M')!
  const cpuTeacher = results.cpu.find((c) => c.model === 'teacher' && c.format === 'Q4_K_M')!
  const se = (h.std_error * 100).toFixed(1)
  const [params, setParams] = useSearchParams()
  const tab = TABS.includes(params.get('tab') ?? '') ? params.get('tab')! : 'overview'

  return (
    <div className="space-y-8">
      <section className="space-y-4">
        <div className="space-y-2">
          <Badge variant="secondary">
            {results.benchmark.name} · {fmtInt(results.benchmark.questions)} questions · {results.benchmark.databases} unseen databases
          </Badge>
          <h1 className="max-w-3xl text-3xl font-semibold tracking-tight sm:text-4xl">A 1.5B model keeps {h.retained_pct}% of a 9B teacher&apos;s SQL accuracy</h1>
          <p className="max-w-3xl text-muted-foreground">
            Qwen3.5-9B wrote and verified {fmtInt(results.teacher_data.kept)} question/SQL pairs; Qwen2.5-Coder-1.5B was fine-tuned on them with QLoRA. Everything below is measured with
            the official test-suite execution-accuracy scorer, zero-shot, on one RTX 4000 Ada.
          </p>
        </div>
        <div className="grid grid-cols-2 gap-3 md:grid-cols-3 lg:grid-cols-5">
          <StatCard value={`${h.retained_pct}%`} label="of teacher accuracy" sub={`${pct(h.student_ex)}% vs ${pct(h.teacher_ex)}% · base student ${pct(h.base_ex)}%`} accent="var(--m-final)" />
          <StatCard value={`${h.latency_speedup}×`} label="faster per question" sub="median latency, batch size 1" />
          <StatCard value={`${h.throughput_speedup}×`} label="more throughput" sub="queries per second, batch 32" />
          <StatCard value={`${h.vram_reduction}×`} label="less GPU memory" sub="6.5 vs 15.5 GiB peak" />
          <StatCard value={`${h.size_reduction}×`} label="smaller on disk" sub="2.9 vs 13.1 GiB" />
        </div>
      </section>

      <Tabs value={tab} onValueChange={(v) => setParams(v === 'overview' ? {} : { tab: String(v) }, { replace: true, preventScrollReset: true })} className="space-y-4">
        <TabsList className="h-auto flex-wrap justify-start">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="ablations">Ablations</TabsTrigger>
          <TabsTrigger value="cpu">CPU</TabsTrigger>
          <TabsTrigger value="data">Training &amp; data</TabsTrigger>
          <TabsTrigger value="errors">Errors</TabsTrigger>
        </TabsList>

        {/* ---------------- overview ---------------- */}
        <TabsContent value="overview" className="space-y-6">
          <Section title="All models" description="Execution accuracy per hardness level, latency at batch size 1, throughput at batch 32, peak GPU memory and size on disk.">
            <ModelTable />
            <p className="text-xs text-muted-foreground">
              The three LoRA rows run the adapter through vLLM, which adds latency and memory; the final row has the adapter merged into the weights (same accuracy, {pct(results.models.at(-1)!.ex)}% vs{' '}
              {pct(results.models.find((m) => m.id === 'combined')!.ex)}%).
            </p>
          </Section>

          <div className="grid gap-6 lg:grid-cols-2">
            <Section title="Accuracy vs latency" description="Up and to the left is better.">
              <AccuracyLatencyChart />
            </Section>
            <Section title="Accuracy by question hardness" description="Spider hardness levels. Pick the models to compare.">
              <HardnessChart />
            </Section>
          </div>

          <Insight>
            <p>
              <strong>Distillation closed {h.gap_closed_pct}% of the gap</strong> between the untuned student ({pct(h.base_ex)}%) and the teacher ({pct(h.teacher_ex)}%). The gain is concentrated on harder
              questions: hard {pct(results.models.find((m) => m.id === 'base')!.hardness.hard, 0)}% → {pct(results.models.at(-1)!.hardness.hard, 0)}%, extra-hard{' '}
              {pct(results.models.find((m) => m.id === 'base')!.hardness.extra, 0)}% → {pct(results.models.at(-1)!.hardness.extra, 0)}%.
            </p>
            <p>The teacher&apos;s 3-shot prompt did not help (77.5% vs 78.2%) and was 1.3× slower, so every comparison uses zero-shot.</p>
          </Insight>
        </TabsContent>

        {/* ---------------- ablations ---------------- */}
        <TabsContent value="ablations" className="space-y-6">
          <div className="grid gap-6 lg:grid-cols-2">
            <Section title="Which training data?" description="Same QLoRA recipe, only the data changes. Gain over the untuned student in points.">
              <SourceAblationChart />
              <Insight>
                <p>Every variant helps by 8–10 points. Teacher data alone is at least as good as Spider&apos;s human labels (76.9 vs 75.5, inside the noise), and combining both is best on the hard end:</p>
                <p>teacher data helps most on <em>hard</em> questions (71% vs 63%), human data on <em>extra-hard</em> (51% vs 44%).</p>
              </Insight>
            </Section>
            <Section title="How much teacher data?" description={`One training run per setting; differences under about ${se} points (one standard error) are noise.`}>
              <SizeAblationChart />
              <Insight>
                <p>Teacher-only keeps improving by about a point per doubling with no sign of saturation. With the human data already in the mix, 25% of the teacher pairs (5.8k) reaches 76.7%, within noise of using all of them.</p>
                <p>The dip at 50% is about one standard error; do not read it as a real effect.</p>
              </Insight>
            </Section>
          </div>
          <Section title="Hardness breakdown by data source">
            <div className="overflow-x-auto rounded-lg border">
              <Table>
                <TableHeader className="bg-muted/50">
                  <TableRow>
                    <TableHead>Training data</TableHead>
                    <TableHead className="text-right">Overall</TableHead>
                    <TableHead className="text-right">Easy</TableHead>
                    <TableHead className="text-right">Medium</TableHead>
                    <TableHead className="text-right">Hard</TableHead>
                    <TableHead className="text-right">Extra</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {['base', 'human', 'teacher-data', 'combined'].map((id) => {
                    const m = results.models.find((x) => x.id === id)!
                    return (
                      <TableRow key={id}>
                        <TableCell>{m.short}</TableCell>
                        <TableCell className="text-right font-medium tabular-nums">{pct(m.ex)}</TableCell>
                        {(['easy', 'medium', 'hard', 'extra'] as const).map((k) => (
                          <TableCell key={k} className="text-right tabular-nums">{pct(m.hardness[k])}</TableCell>
                        ))}
                      </TableRow>
                    )
                  })}
                </TableBody>
              </Table>
            </div>
          </Section>
        </TabsContent>

        {/* ---------------- CPU ---------------- */}
        <TabsContent value="cpu" className="space-y-6">
          <div className="grid gap-3 sm:grid-cols-3">
            <StatCard value={`${h.cpu_speedup_q4}×`} label="faster than the teacher on CPU" sub={`${fmtMs(cpuStudent.p50_ms)} vs ${fmtMs(cpuTeacher.p50_ms)} median, both Q4_K_M`} accent="var(--m-final)" />
            <StatCard value={fmtGib(cpuStudent.size_gib)} label="student Q4_K_M model" sub={`${h.cpu_size_reduction_q4}× smaller than the teacher's ${fmtGib(cpuTeacher.size_gib)}`} />
            <StatCard value={`${h.cpu_speedup_q8}×`} label="faster at Q8_0 too" sub="1.24 s vs 13.7 s median" />
          </div>
          <Section title="Median latency per question on a CPU" description="Intel i9-14900K, 32 threads, no GPU, one question at a time, first 100 dev questions.">
            <CpuChart />
          </Section>
          <Section title="CPU runs in detail">
            <div className="overflow-x-auto rounded-lg border">
              <Table>
                <TableHeader className="bg-muted/50">
                  <TableRow>
                    <TableHead>Model</TableHead>
                    <TableHead>Runtime</TableHead>
                    <TableHead className="text-right">Size</TableHead>
                    <TableHead className="text-right">p50</TableHead>
                    <TableHead className="text-right">p95</TableHead>
                    <TableHead className="text-right">EX (n=100)</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {results.cpu.map((c) => (
                    <TableRow key={c.run}>
                      <TableCell>{c.label.replace(/ \(.*\)/, '')}</TableCell>
                      <TableCell className="text-muted-foreground">{c.runtime}</TableCell>
                      <TableCell className="text-right tabular-nums">{fmtGib(c.size_gib)}</TableCell>
                      <TableCell className="text-right tabular-nums">{fmtMs(c.p50_ms)}</TableCell>
                      <TableCell className="text-right tabular-nums">{fmtMs(c.p95_ms)}</TableCell>
                      <TableCell className="text-right tabular-nums">{pct(c.ex, 0)}%</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          </Section>
          <Insight>
            <p>On the same CPU, runtime and 4-bit quantisation the student answers in under a second; the teacher needs about ten. The student&apos;s accuracy gap is larger here (77% vs 84%) than on the full GPU evaluation (1 point).</p>
            <p>With 100 questions the standard error is about ±4 points, so that gap is only 1.5 to 2 standard errors and may be partly noise, partly the cost of 4-bit quantisation for the small model. The teacher GGUF was re-quantised from its FP8 checkpoint.</p>
          </Insight>
        </TabsContent>

        {/* ---------------- training & data ---------------- */}
        <TabsContent value="data" className="space-y-6">
          <div className="grid gap-6 lg:grid-cols-2">
            <Section title="Teacher data: the filter chain" description="Every pair must execute and be verified. Only 57% of synthesised and 79% of relabelled pairs survive the correctness check.">
              <FunnelChart />
            </Section>
            <Section title="What the kept pairs look like" description={`${fmtInt(results.teacher_data.kept)} pairs over ${results.teacher_data.databases} training databases, by SQL pattern.`}>
              <CompositionChart />
            </Section>
          </div>
          <Section title="Training curves" description="QLoRA, 2 epochs, learning rate 2e-4 (cosine).">
            <LossChart />
          </Section>
          <Section title="Training runs">
            <div className="overflow-x-auto rounded-lg border">
              <Table>
                <TableHeader className="bg-muted/50">
                  <TableRow>
                    <TableHead>Run</TableHead>
                    <TableHead className="text-right">Examples</TableHead>
                    <TableHead className="text-right">Steps</TableHead>
                    <TableHead className="text-right">Minutes</TableHead>
                    <TableHead className="text-right">Best val loss</TableHead>
                    <TableHead className="text-right">Peak GPU</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {results.training.map((t) => (
                    <TableRow key={t.run}>
                      <TableCell>{t.label}</TableCell>
                      <TableCell className="text-right tabular-nums">{fmtInt(t.train_examples)}</TableCell>
                      <TableCell className="text-right tabular-nums">{fmtInt(t.steps)}</TableCell>
                      <TableCell className="text-right tabular-nums">{t.minutes.toFixed(0)}</TableCell>
                      <TableCell className="text-right tabular-nums">{t.best_val_loss}</TableCell>
                      <TableCell className="text-right tabular-nums">{t.peak_gpu_gib.toFixed(1)} GiB</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          </Section>
        </TabsContent>

        {/* ---------------- errors ---------------- */}
        <TabsContent value="errors" className="space-y-6">
          <div className="grid gap-6 lg:grid-cols-2">
            <Section title="What the student gets wrong" description="100 random failures of the best adapter model, categorised by which parts of the SQL differ from the gold query.">
              <ErrorChart />
              <Insight>
                <p>Only 1% of failures are SQL that does not run: the student writes valid SQL that answers a slightly different question. Schema linking (wrong table or column) is the main weakness.</p>
              </Insight>
            </Section>
            <Section title="Student vs teacher, question by question" description={`All ${fmtInt(results.benchmark.questions)} dev questions, best adapter model vs the teacher.`}>
              <PairBar />
              <Insight>
                <p>The two models fail on largely different questions: the student is right on {results.student_vs_teacher.student_only} that the teacher gets wrong, and the other way round on {results.student_vs_teacher.teacher_only}, which is why their accuracies are so close.</p>
              </Insight>
            </Section>
          </div>
        </TabsContent>
      </Tabs>

      <Card className="border-[var(--m-teacher)]/40">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <AlertTriangle className="size-4 text-[var(--m-teacher)]" /> How to read these numbers
          </CardTitle>
        </CardHeader>
        <CardContent>
          <ul className="list-disc space-y-1.5 pl-5 text-sm text-muted-foreground">
            <li>With {fmtInt(results.benchmark.questions)} questions the standard error of an accuracy near 77% is about ±{se} points; gaps under ~2.5 points are not established.</li>
            <li>Each variant was trained once (seed 0), so no variance across training runs was measured.</li>
            <li>One benchmark (Spider dev, SQLite). Results may not carry over to other dialects, much larger schemas or messier real-world questions.</li>
            <li>Execution accuracy credits any query returning the right result and penalises correct answers to ambiguous questions; some &quot;failures&quot; are benchmark artefacts.</li>
            <li>CPU numbers use 100 questions (easier than average) on one desktop CPU; compare them with each other, not with the full-dev accuracies.</li>
          </ul>
        </CardContent>
      </Card>
    </div>
  )
}
