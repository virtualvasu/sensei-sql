import { cn } from '@/lib/utils'

const KEYWORDS = new Set(
  `select from where group by having order limit offset join inner left right outer full cross on as and or not in is null like between exists
   union intersect except all distinct asc desc case when then else end with count sum avg min max`.split(/\s+/),
)

const TOKEN = /('(?:[^']|'')*'|"(?:[^"]|"")*")|(--[^\n]*)|(\b\d+(?:\.\d+)?\b)|([A-Za-z_][A-Za-z0-9_]*)|(\s+)|([\s\S])/g

/** Read-only, syntax-highlighted SQL (no dependency; colours come from the theme tokens). */
export function SqlView({ sql, className }: { sql: string; className?: string }) {
  const out: React.ReactNode[] = []
  let m: RegExpExecArray | null
  let i = 0
  TOKEN.lastIndex = 0
  while ((m = TOKEN.exec(sql)) !== null) {
    const [tok, str, comment, num, word] = m
    const key = i++
    if (str) out.push(<span key={key} className="text-[var(--m-final)]">{tok}</span>)
    else if (comment) out.push(<span key={key} className="text-muted-foreground italic">{tok}</span>)
    else if (num) out.push(<span key={key} className="text-[var(--m-teacher)]">{tok}</span>)
    else if (word) {
      const isFn = /^\s*\(/.test(sql.slice(m.index + tok.length)) && !KEYWORDS.has(tok.toLowerCase())
      if (KEYWORDS.has(tok.toLowerCase())) out.push(<span key={key} className="font-semibold text-[var(--m-teacher-data)]">{tok}</span>)
      else if (isFn) out.push(<span key={key} className="text-[var(--m-combined)]">{tok}</span>)
      else out.push(<span key={key}>{tok}</span>)
    } else out.push(tok)
  }
  return (
    <pre className={cn('overflow-x-auto whitespace-pre-wrap break-words rounded-lg border bg-muted/40 p-4 font-mono text-sm leading-relaxed', className)}>
      <code>{out}</code>
    </pre>
  )
}
