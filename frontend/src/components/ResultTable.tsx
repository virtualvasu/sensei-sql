import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import type { QueryResult } from '@/lib/api'

const isNumber = (v: unknown) => typeof v === 'number'

export function ResultTable({ result }: { result: QueryResult }) {
  if (!result.columns.length) return null
  return (
    <div className="space-y-2">
      <div className="max-h-96 overflow-auto rounded-lg border">
        <Table>
          <TableHeader className="sticky top-0 bg-muted/90 backdrop-blur">
            <TableRow>
              <TableHead className="w-10 text-right text-muted-foreground">#</TableHead>
              {result.columns.map((c, i) => (
                <TableHead key={i} className={result.rows.some((r) => isNumber(r[i])) ? 'text-right' : undefined}>
                  {c}
                </TableHead>
              ))}
            </TableRow>
          </TableHeader>
          <TableBody>
            {result.rows.map((row, r) => (
              <TableRow key={r}>
                <TableCell className="text-right text-xs text-muted-foreground tabular-nums">{r + 1}</TableCell>
                {row.map((v, c) => (
                  <TableCell key={c} className={isNumber(v) ? 'text-right tabular-nums' : 'max-w-xs truncate'}>
                    {v === null ? <span className="text-muted-foreground italic">NULL</span> : String(v)}
                  </TableCell>
                ))}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
      {result.truncated && (
        <p className="text-xs text-muted-foreground">Showing the first {result.rows.length.toLocaleString()} rows; the query returned more.</p>
      )}
    </div>
  )
}
