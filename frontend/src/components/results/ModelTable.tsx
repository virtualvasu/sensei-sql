import { Badge } from '@/components/ui/badge'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { cn } from '@/lib/utils'
import { MODEL_COLOR, fmtGib, fmtMs, pct, results } from '@/lib/results'

export function ModelTable() {
  const teacher = results.models.find((m) => m.id === 'teacher')!
  return (
    <div className="overflow-x-auto rounded-lg border">
      <Table>
        <TableHeader className="bg-muted/50">
          <TableRow>
            <TableHead>Model</TableHead>
            <TableHead className="min-w-40">Execution accuracy</TableHead>
            <TableHead className="text-right">Easy</TableHead>
            <TableHead className="text-right">Medium</TableHead>
            <TableHead className="text-right">Hard</TableHead>
            <TableHead className="text-right">Extra</TableHead>
            <TableHead className="text-right">p50</TableHead>
            <TableHead className="text-right">p95</TableHead>
            <TableHead className="text-right">q/s</TableHead>
            <TableHead className="text-right">VRAM</TableHead>
            <TableHead className="text-right">Size</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {results.models.map((m) => (
            <TableRow key={m.id} className={cn(m.kind === 'final' && 'bg-[var(--m-final)]/8 font-medium')}>
              <TableCell className="whitespace-nowrap">
                <span className="mr-2 inline-block size-2.5 rounded-full align-middle" style={{ background: MODEL_COLOR[m.id] }} />
                {m.label}
                {m.kind === 'final' && <Badge className="ml-2 bg-[var(--m-final)] text-white">final</Badge>}
              </TableCell>
              <TableCell>
                <div className="flex items-center gap-2">
                  <div className="h-1.5 w-24 overflow-hidden rounded-full bg-muted">
                    <div className="h-full rounded-full" style={{ width: `${m.ex * 100}%`, background: MODEL_COLOR[m.id] }} />
                  </div>
                  <span className="tabular-nums">{pct(m.ex)}%</span>
                  {m.id !== 'teacher' && <span className="text-xs text-muted-foreground tabular-nums">{pct(m.ex / teacher.ex, 0)}% of teacher</span>}
                </div>
              </TableCell>
              {(['easy', 'medium', 'hard', 'extra'] as const).map((h) => (
                <TableCell key={h} className="text-right tabular-nums">
                  {pct(m.hardness[h])}
                </TableCell>
              ))}
              <TableCell className="text-right tabular-nums">{fmtMs(m.p50_ms)}</TableCell>
              <TableCell className="text-right tabular-nums">{fmtMs(m.p95_ms)}</TableCell>
              <TableCell className="text-right tabular-nums">{m.qps?.toFixed(1) ?? '–'}</TableCell>
              <TableCell className="text-right tabular-nums">{fmtGib(m.vram_gib)}</TableCell>
              <TableCell className="text-right tabular-nums">{fmtGib(m.size_gib)}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  )
}
