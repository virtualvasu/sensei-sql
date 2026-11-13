export type Status = 'ok' | 'empty' | 'rejected' | 'timeout' | 'error' | 'no_sql'

export interface QueryResult {
  sql: string
  status: Status
  gen_ms: number | null
  exec_ms: number
  attempts: { sql: string; error: string }[]
  columns: string[]
  rows: (string | number | boolean | null)[][]
  row_count: number
  truncated: boolean
  error: string | null
}

export interface Health {
  title: string
  backend: string
  self_correct: boolean
}

export interface DatabaseList {
  databases: { name: string; upload: boolean }[]
  default: string | null
}

async function call<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init)
  if (!res.ok) {
    let detail = res.statusText
    try {
      detail = (await res.json()).detail ?? detail
    } catch {
      /* not JSON */
    }
    throw new Error(String(detail))
  }
  return res.json() as Promise<T>
}

const post = (url: string, body: unknown) =>
  call<QueryResult>(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })

export const api = {
  health: () => call<Health>('/api/health'),
  databases: () => call<DatabaseList>('/api/databases'),
  schema: (db: string) => call<{ schema: string }>(`/api/schema?db=${encodeURIComponent(db)}`),
  examples: (db: string) => call<{ examples: string[] }>(`/api/examples?db=${encodeURIComponent(db)}`),
  ask: (db: string, question: string) => post('/api/ask', { db, question }),
  run: (db: string, sql: string) => post('/api/run', { db, sql }),
  upload: (file: File) => {
    const form = new FormData()
    form.append('file', file)
    return call<{ name: string }>('/api/upload', { method: 'POST', body: form })
  },
}
