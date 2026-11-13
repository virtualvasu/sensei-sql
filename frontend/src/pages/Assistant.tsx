import { useCallback, useEffect, useRef, useState } from 'react'
import { ChevronDown, Clipboard, ClipboardCheck, Database, Download, History, Loader2, MessageSquareText, Pencil, Play, Sparkles, Table2, Trash2, Upload } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import { Skeleton } from '@/components/ui/skeleton'
import { Textarea } from '@/components/ui/textarea'
import { DatabasePicker } from '@/components/DatabasePicker'
import { ResultTable } from '@/components/ResultTable'
import { SqlView } from '@/components/SqlView'
import { StatusBanner } from '@/components/StatusBanner'
import { api, type DatabaseList, type Health, type QueryResult, type Status } from '@/lib/api'
import { cn } from '@/lib/utils'

interface HistoryItem {
  id: number
  time: string
  db: string
  question: string
  sql: string
  status: Status
  rows: number
}

const statusTone: Record<Status, string> = {
  ok: 'bg-[var(--m-final)]/15 text-[var(--m-final)]',
  empty: 'bg-muted text-muted-foreground',
  rejected: 'bg-destructive/10 text-destructive',
  timeout: 'bg-destructive/10 text-destructive',
  error: 'bg-destructive/10 text-destructive',
  no_sql: 'bg-destructive/10 text-destructive',
}

const tableNames = (schema: string) =>
  [...schema.matchAll(/CREATE TABLE\s+[`"[]?([A-Za-z0-9_ ]+?)[`"\]]?\s*\(/gi)].map((m) => m[1])

function downloadCsv(r: QueryResult) {
  const esc = (v: unknown) => {
    const t = v === null || v === undefined ? '' : String(v)
    return /[",\n]/.test(t) ? `"${t.replace(/"/g, '""')}"` : t
  }
  const csv = [r.columns.map(esc).join(','), ...r.rows.map((row) => row.map(esc).join(','))].join('\n')
  const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8' }))
  const a = document.createElement('a')
  a.href = url
  a.download = 'sensei-sql-result.csv'
  a.click()
  URL.revokeObjectURL(url)
}

const Kbd = ({ children }: { children: React.ReactNode }) => (
  <kbd className="rounded border bg-muted px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground">{children}</kbd>
)

export default function Assistant() {
  const [health, setHealth] = useState<Health | null>(null)
  const [dbs, setDbs] = useState<DatabaseList['databases']>([])
  const [db, setDb] = useState<string | null>(null)
  const [schema, setSchema] = useState('')
  const [examples, setExamples] = useState<string[]>([])
  const [question, setQuestion] = useState('')
  const [sql, setSql] = useState('')
  const [result, setResult] = useState<QueryResult | null>(null)
  const [busy, setBusy] = useState<'ask' | 'run' | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [uploadMsg, setUploadMsg] = useState<{ ok: boolean; text: string } | null>(null)
  const [history, setHistory] = useState<HistoryItem[]>([])
  const [copied, setCopied] = useState(false)
  const [editing, setEditing] = useState(false)
  const fileInput = useRef<HTMLInputElement>(null)
  const nextId = useRef(1)

  useEffect(() => {
    api.health().then(setHealth).catch(() => setError('Cannot reach the sensei-sql server. Is `python -m sensei_sql.app` running?'))
    api
      .databases()
      .then((d) => {
        setDbs(d.databases)
        setDb(d.default)
      })
      .catch(() => {})
  }, [])

  useEffect(() => {
    if (!db) return
    let live = true
    api.schema(db).then((s) => live && setSchema(s.schema)).catch(() => live && setSchema(''))
    setExamples([])
    api.examples(db).then((r) => live && setExamples(r.examples)).catch(() => live && setExamples([]))
    return () => {
      live = false
    }
  }, [db])

  const record = useCallback((entry: Omit<HistoryItem, 'id' | 'time'>) => {
    const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
    setHistory((h) => [{ id: nextId.current++, time, ...entry }, ...h].slice(0, 50))
  }, [])

  const ask = async (q = question) => {
    if (!db || !q.trim() || busy) return
    setBusy('ask')
    setError(null)
    try {
      const r = await api.ask(db, q)
      setResult(r)
      setSql(r.sql)
      setEditing(false)
      record({ db, question: q, sql: r.sql, status: r.status, rows: r.row_count })
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(null)
    }
  }

  const run = async () => {
    if (!db || !sql.trim() || busy) return
    setBusy('run')
    setError(null)
    try {
      const r = await api.run(db, sql)
      setResult(r)
      record({ db, question: '(edited SQL)', sql, status: r.status, rows: r.row_count })
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(null)
    }
  }

  const onUpload = async (file: File | undefined) => {
    if (!file) return
    try {
      const { name } = await api.upload(file)
      const d = await api.databases()
      setDbs(d.databases)
      setDb(name)
      setUploadMsg({ ok: true, text: `Added database ${name}` })
    } catch (e) {
      setUploadMsg({ ok: false, text: e instanceof Error ? e.message : String(e) })
    }
    if (fileInput.current) fileInput.current.value = ''
  }

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(sql)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch {
      /* clipboard unavailable on plain http: ignore */
    }
  }

  const onKey = (action: () => void) => (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
      e.preventDefault()
      action()
    }
  }

  const names = tableNames(schema)
  const tables = names.length || (schema.match(/CREATE TABLE/gi) ?? []).length

  return (
    <div className="space-y-8">
      <div className="space-y-3">
        <h1 className="text-3xl font-semibold tracking-tight">Ask your database</h1>
        <p className="max-w-2xl text-sm text-muted-foreground">
          Type a question in plain English and get SQL back, plus the result. Runs entirely on your machine; only read-only SELECT queries are executed.
        </p>
        <div className="flex flex-wrap items-center gap-2">
          <Badge variant="secondary">1.5B model</Badge>
          {health && <Badge variant="secondary">{health.backend}</Badge>}
          <Badge variant="secondary">fully offline</Badge>
          {health?.self_correct && <Badge variant="secondary">auto-retry on SQL errors</Badge>}
        </div>
      </div>

      {error && (
        <div className="rounded-lg border border-destructive/30 bg-destructive/10 px-3 py-2 text-sm text-destructive">{error}</div>
      )}

      <div className="grid gap-6 lg:grid-cols-[320px_1fr]">
        <aside className="space-y-4 lg:sticky lg:top-20 lg:self-start">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Database className="size-4 text-muted-foreground" /> Database
              </CardTitle>
              <CardDescription>The 166 Spider databases, or your own SQLite file.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <DatabasePicker databases={dbs} value={db} onChange={setDb} />
              <input ref={fileInput} type="file" accept=".sqlite,.sqlite3,.db" className="hidden" onChange={(e) => onUpload(e.target.files?.[0])} />
              <Button variant="outline" className="w-full" onClick={() => fileInput.current?.click()}>
                <Upload /> Upload a SQLite file
              </Button>
              {uploadMsg && <p className={cn('text-xs', uploadMsg.ok ? 'text-[var(--m-final)]' : 'text-destructive')}>{uploadMsg.text}</p>}
            </CardContent>
          </Card>

          <Card>
            <Collapsible defaultOpen>
              <CardHeader>
                <CollapsibleTrigger render={<button type="button" className="group flex w-full items-center justify-between text-left" />}>
                  <span>
                    <CardTitle className="flex items-center gap-2">
                      <Table2 className="size-4 text-muted-foreground" /> Schema
                    </CardTitle>
                    <CardDescription>{tables ? `${tables} table${tables === 1 ? '' : 's'}, as the model sees it` : 'Pick a database'}</CardDescription>
                  </span>
                  <ChevronDown className="size-4 text-muted-foreground transition-transform group-aria-expanded:rotate-180" />
                </CollapsibleTrigger>
              </CardHeader>
              <CollapsibleContent>
                <CardContent className="space-y-3">
                  {names.length > 0 && (
                    <div className="flex flex-wrap gap-1.5">
                      {names.map((n) => (
                        <Badge key={n} variant="outline" className="font-mono">
                          {n}
                        </Badge>
                      ))}
                    </div>
                  )}
                  <div className="h-64 overflow-auto rounded-lg border bg-muted/40">
                    <pre className="w-max min-w-full p-3 font-mono text-xs leading-relaxed">{schema}</pre>
                  </div>
                </CardContent>
              </CollapsibleContent>
            </Collapsible>
          </Card>
        </aside>

        <div className="min-w-0 space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <MessageSquareText className="size-4 text-muted-foreground" /> Your question
              </CardTitle>
              <CardDescription>{db ? <>Asking <span className="font-mono text-foreground">{db}</span></> : 'Pick a database first'}</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <Textarea
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                onKeyDown={onKey(() => ask())}
                placeholder="e.g. How many singers are older than 30?"
                rows={3}
                className="resize-none text-base md:text-base"
              />
              <div className="flex flex-wrap items-center justify-between gap-3">
                <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
                  <Kbd>Ctrl / ⌘</Kbd> + <Kbd>Enter</Kbd> to ask
                </span>
                <Button size="lg" onClick={() => ask()} disabled={!db || !question.trim() || !!busy}>
                  {busy === 'ask' ? <Loader2 className="animate-spin" /> : <Sparkles />} Generate SQL
                </Button>
              </div>
              {examples.length > 0 && (
                <div className="space-y-2 border-t pt-4">
                  <p className="text-xs font-medium text-muted-foreground">Try one of these</p>
                  <div className="flex flex-wrap gap-2">
                    {examples.map((ex) => (
                      <button
                        key={ex}
                        type="button"
                        disabled={!!busy}
                        onClick={() => {
                          setQuestion(ex)
                          ask(ex)
                        }}
                        className="rounded-full border bg-muted/30 px-3 py-1.5 text-left text-xs text-foreground/80 transition-colors hover:border-[var(--m-final)]/60 hover:bg-muted hover:text-foreground disabled:opacity-50"
                      >
                        {ex}
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          {busy === 'ask' && (
            <Card>
              <CardContent className="space-y-3 pt-4">
                <Skeleton className="h-4 w-1/3" />
                <Skeleton className="h-20 w-full" />
                <Skeleton className="h-32 w-full" />
              </CardContent>
            </Card>
          )}

          {!result && busy !== 'ask' && (
            <Card className="border-dashed bg-transparent">
              <CardContent className="grid gap-4 py-2 sm:grid-cols-3">
                {[
                  ['1', 'Pick a database', 'Any Spider database, or upload your own SQLite file.'],
                  ['2', 'Ask in plain English', 'Click a suggestion or type your own question.'],
                  ['3', 'Review, edit, run', 'Inspect the SQL, tweak it, and export the result.'],
                ].map(([n, title, text]) => (
                  <div key={n} className="flex gap-3">
                    <span className="grid size-7 shrink-0 place-items-center rounded-full bg-muted text-sm font-semibold">{n}</span>
                    <div>
                      <p className="text-sm font-medium">{title}</p>
                      <p className="text-xs text-muted-foreground">{text}</p>
                    </div>
                  </div>
                ))}
              </CardContent>
            </Card>
          )}

          {result && busy !== 'ask' && (
            <>
              <Card>
                <CardHeader>
                  <CardTitle className="flex flex-wrap items-center gap-2">
                    Generated SQL
                    <span className="flex flex-wrap items-center gap-1.5 text-xs font-normal">
                      {result.gen_ms !== null && <Badge variant="secondary">generated in {Math.round(result.gen_ms)} ms</Badge>}
                      <Badge variant="secondary">ran in {Math.round(result.exec_ms)} ms</Badge>
                      {result.attempts.length > 0 && <Badge variant="outline">auto-corrected</Badge>}
                    </span>
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  {editing ? (
                    <Textarea
                      value={sql}
                      onChange={(e) => setSql(e.target.value)}
                      onKeyDown={onKey(run)}
                      spellCheck={false}
                      autoFocus
                      rows={Math.min(12, Math.max(4, sql.split('\n').length + 1))}
                      className="font-mono text-sm"
                    />
                  ) : (
                    <SqlView sql={sql} />
                  )}
                  <div className="flex flex-wrap gap-2">
                    {editing ? (
                      <>
                        <Button onClick={run} disabled={!sql.trim() || !!busy}>
                          {busy === 'run' ? <Loader2 className="animate-spin" /> : <Play />} Run edited SQL
                        </Button>
                        <Button variant="ghost" onClick={() => setEditing(false)}>
                          Cancel
                        </Button>
                      </>
                    ) : (
                      <Button variant="secondary" onClick={() => setEditing(true)}>
                        <Pencil /> Edit SQL
                      </Button>
                    )}
                    <Button variant="ghost" onClick={copy}>
                      {copied ? <ClipboardCheck /> : <Clipboard />} {copied ? 'Copied' : 'Copy'}
                    </Button>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center justify-between gap-2">
                    Result
                    {result.columns.length > 0 && result.rows.length > 0 && (
                      <Button variant="outline" size="sm" onClick={() => downloadCsv(result)}>
                        <Download /> Export CSV
                      </Button>
                    )}
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <StatusBanner result={result} />
                  <ResultTable result={result} />
                </CardContent>
              </Card>
            </>
          )}

          {history.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center justify-between gap-2">
                  <span className="flex items-center gap-2">
                    <History className="size-4" /> This session
                  </span>
                  <Button variant="ghost" size="sm" onClick={() => setHistory([])}>
                    <Trash2 /> Clear
                  </Button>
                </CardTitle>
                <CardDescription>Click an entry to bring back its question and SQL.</CardDescription>
              </CardHeader>
              <CardContent>
                <ul className="divide-y">
                  {history.map((h) => (
                    <li key={h.id}>
                      <button
                        type="button"
                        className="flex w-full items-start gap-3 rounded-md px-1 py-2 text-left text-sm hover:bg-muted/50"
                        onClick={() => {
                          setDb(h.db)
                          if (h.question !== '(edited SQL)') setQuestion(h.question)
                          setSql(h.sql)
                        }}
                      >
                        <span className="w-16 shrink-0 pt-0.5 text-xs text-muted-foreground tabular-nums">{h.time}</span>
                        <span className="min-w-0 flex-1">
                          <span className="block truncate">{h.question}</span>
                          <span className="block truncate font-mono text-xs text-muted-foreground">{h.sql}</span>
                        </span>
                        <span className="flex shrink-0 flex-col items-end gap-1">
                          <span className="text-xs text-muted-foreground">{h.db}</span>
                          <span className={cn('rounded-full px-2 py-0.5 text-xs', statusTone[h.status])}>{h.status === 'ok' ? `${h.rows} rows` : h.status}</span>
                        </span>
                      </button>
                    </li>
                  ))}
                </ul>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  )
}
