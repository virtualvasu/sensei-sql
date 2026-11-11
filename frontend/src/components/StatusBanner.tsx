import { CheckCircle2, Clock, Info, OctagonX, RefreshCw, ShieldAlert, XCircle } from 'lucide-react'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import type { QueryResult } from '@/lib/api'

export function StatusBanner({ result }: { result: QueryResult }) {
  const { status, row_count, truncated, error, attempts } = result
  const retried = attempts.length > 0
  let node
  switch (status) {
    case 'ok':
      node = (
        <Alert>
          <CheckCircle2 className="text-[var(--m-final)]" />
          <AlertTitle>
            {row_count.toLocaleString()} row{row_count === 1 ? '' : 's'}
            {truncated ? ' (first rows shown)' : ''}
          </AlertTitle>
        </Alert>
      )
      break
    case 'empty':
      node = (
        <Alert>
          <Info />
          <AlertTitle>The query ran but returned no rows</AlertTitle>
          <AlertDescription>If you expected data, try rephrasing the question or check the schema.</AlertDescription>
        </Alert>
      )
      break
    case 'rejected':
      node = (
        <Alert variant="destructive">
          <ShieldAlert />
          <AlertTitle>Only read-only SELECT queries are allowed</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )
      break
    case 'timeout':
      node = (
        <Alert variant="destructive">
          <Clock />
          <AlertTitle>The query took too long and was stopped</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )
      break
    case 'no_sql':
      node = (
        <Alert variant="destructive">
          <OctagonX />
          <AlertTitle>The model did not produce a SQL query</AlertTitle>
          <AlertDescription>Try rephrasing the question.</AlertDescription>
        </Alert>
      )
      break
    default:
      node = (
        <Alert variant="destructive">
          <XCircle />
          <AlertTitle>The query failed</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )
  }
  return (
    <div className="space-y-2">
      {node}
      {retried && (
        <Alert>
          <RefreshCw />
          <AlertTitle>Self-corrected</AlertTitle>
          <AlertDescription>
            The first attempt failed (<code className="font-mono text-xs">{attempts[0].error}</code>), so the model rewrote its query.
          </AlertDescription>
        </Alert>
      )}
    </div>
  )
}
