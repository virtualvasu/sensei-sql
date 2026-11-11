import type { ReactNode } from 'react'
import { Card, CardContent } from '@/components/ui/card'

export function StatCard({ value, label, sub, accent }: { value: ReactNode; label: string; sub?: string; accent?: string }) {
  return (
    <Card size="sm" className="gap-1">
      <CardContent className="space-y-1">
        <div className="text-3xl font-semibold tracking-tight tabular-nums" style={accent ? { color: accent } : undefined}>
          {value}
        </div>
        <div className="text-sm font-medium">{label}</div>
        {sub && <div className="text-xs text-muted-foreground">{sub}</div>}
      </CardContent>
    </Card>
  )
}
