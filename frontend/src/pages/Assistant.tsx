import { useCallback, useEffect, useRef, useState } from 'react'
import { ChevronDown, Clipboard, ClipboardCheck, History, Loader2, Play, Sparkles, Upload } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Skeleton } from '@/components/ui/skeleton'
import { Textarea } from '@/components/ui/textarea'
import { DatabasePicker } from '@/components/DatabasePicker'
import { ResultTable } from '@/components/ResultTable'
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

  const tables = (schema.match(/CREATE TABLE/gi) ?? []).length

  return (
    <div className="space-y-6">
      <div className="space-y-1">
        <h1 className="text-2xl font-semibold tracking-tight">Ask your database</h1>
        <p className="text-sm text-muted-foreground">
          Plain-English questions in, SQL out. A 1.5B model running {health ? <Badge variant="secondary">{health.backend}</Badge> : 'locally'}, fully offline; only read-only
          SELECT queries are executed.
        </p>
      </div>

      {error && (
        <div className="rounded-lg border border-destructive/30 bg-destructive/10 px-3 py-2 text-sm text-destructive">{error}</div>
      )}

      <div className="grid gap-6 lg:grid-cols-[320px_1fr]">
        <aside className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Database</CardTitle>
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
            <Collapsible>
              <CardHeader>
                <CollapsibleTrigger render={<button type="button" className="group flex w-full items-center justify-between text-left" />}>
                  <span>
                    <CardTitle>Schema</CardTitle>
                    <CardDescription>{tables ? `${tables} table${tables === 1 ? '' : 's'}, as the model sees it` : 'Pick a database'}</CardDescription>
                  </span>
                  <ChevronDown className="size-4 text-muted-foreground transition-transform group-aria-expanded:rotate-180" />
                </CollapsibleTrigger>
              </CardHeader>
              <CollapsibleContent>
                <CardContent>
                  <ScrollArea className="h-72 rounded-lg border bg-muted/40">
                    <pre className="p-3 font-mono text-xs leading-relaxed">{schema}</pre>
                  </ScrollArea>
                </CardContent>
              </CollapsibleContent>
            </Collapsible>
          </Card>
        </aside>

        <div className="min-w-0 space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Question</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <Textarea
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                onKeyDown={onKey(() => ask())}
                placeholder="e.g. How many singers are older than 30?"
                rows={2}
                className="resize-none text-base md:text-sm"
              />
              <div className="flex flex-wrap items-center gap-2">
                <Button onClick={() => ask()} disabled={!db || !question.trim() || !!busy}>
                  {busy === 'ask' ? <Loader2 className="animate-spin" /> : <Sparkles />} Ask
                </Button>
                <span className="text-xs text-muted-foreground">Ctrl/⌘ + Enter</span>
              </div>
              {examples.length > 0 && (
                <div className="flex flex-wrap gap-2 pt-1">
                  {examples.map((ex) => (
                    <button
                      key={ex}
                      type="button"
                      onClick={() => {
                        setQuestion(ex)
                        ask(ex)
                      }}
                      className="rounded-full border bg-background px-3 py-1 text-xs text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
                    >
                      {ex}
                    </button>
                  ))}
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

          {result && busy !== 'ask' && (
            <Card>
              <CardHeader>
                <CardTitle className="flex flex-wrap items-center gap-2">
                  Generated SQL
                  <span className="flex flex-wrap items-center gap-1.5 text-xs font-normal">
                    {result.gen_ms !== null && <Badge variant="secondary">generation {Math.round(result.gen_ms)} ms</Badge>}
                    <Badge variant="secondary">execution {Math.round(result.exec_ms)} ms</Badge>
                  </span>
                </CardTitle>
                <CardDescription>Editable: change the query and run it again.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <Textarea
                  value={sql}
                  onChange={(e) => setSql(e.target.value)}
                  onKeyDown={onKey(run)}
                  spellCheck={false}
                  rows={Math.min(10, Math.max(3, sql.split('\n').length + 1))}
                  className="font-mono text-sm"
                />
                <div className="flex flex-wrap gap-2">
                  <Button variant="secondary" onClick={run} disabled={!sql.trim() || !!busy}>
                    {busy === 'run' ? <Loader2 className="animate-spin" /> : <Play />} Run edited SQL
                  </Button>
                  <Button variant="ghost" onClick={copy}>
                    {copied ? <ClipboardCheck /> : <Clipboard />} {copied ? 'Copied' : 'Copy'}
                  </Button>
                </div>
                <StatusBanner result={result} />
                <ResultTable result={result} />
              </CardContent>
            </Card>
          )}

          {history.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <History className="size-4" /> This session
                </CardTitle>
                <CardDescription>Click an entry to bring back its question and SQL.</CardDescription>
              </CardHeader>
              <CardContent>
                <ul className="divide-y">
                  {history.map((h) => (
                    <li key={h.id}>
                      <button
                        type="button"
                        className="flex w-full items-start gap-3 py-2 text-left text-sm hover:bg-muted/50"
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
