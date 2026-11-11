import { Suspense, lazy, useEffect } from 'react'
import { NavLink, Route, Routes, useLocation } from 'react-router-dom'
import { BarChart3, Database, Monitor, Moon, Sun } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'
import { useTheme } from '@/lib/theme'
import Assistant from '@/pages/Assistant'
const Results = lazy(() => import('@/pages/Results'))  // pulls in the chart library: load on demand

const links = [
  { to: '/', label: 'Assistant', icon: Database },
  { to: '/results', label: 'Results', icon: BarChart3 },
]

function ThemeToggle() {
  const { theme, setTheme } = useTheme()
  const next = theme === 'light' ? 'dark' : theme === 'dark' ? 'system' : 'light'
  const Icon = theme === 'light' ? Sun : theme === 'dark' ? Moon : Monitor
  return (
    <Button variant="ghost" size="icon" onClick={() => setTheme(next)} aria-label={`Theme: ${theme} (click for ${next})`} title={`Theme: ${theme}`}>
      <Icon />
    </Button>
  )
}

function ScrollToTop() {
  const { pathname } = useLocation()
  useEffect(() => {
    window.scrollTo(0, 0)
  }, [pathname])
  return null
}

export default function App() {
  return (
    <div className="min-h-svh bg-background">
      <header className="sticky top-0 z-30 border-b bg-background/80 backdrop-blur">
        <div className="mx-auto flex h-14 max-w-7xl items-center gap-6 px-4 sm:px-6">
          <NavLink to="/" className="flex items-center gap-2 font-semibold tracking-tight">
            <span className="grid size-7 place-items-center rounded-lg bg-[var(--m-final)] text-white">
              <Database className="size-4" />
            </span>
            sensei-sql
          </NavLink>
          <nav className="flex items-center gap-1">
            {links.map(({ to, label, icon: Icon }) => (
              <NavLink
                key={to}
                to={to}
                end
                className={({ isActive }) =>
                  cn(
                    'flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm font-medium transition-colors',
                    isActive ? 'bg-muted text-foreground' : 'text-muted-foreground hover:text-foreground',
                  )
                }
              >
                <Icon className="size-4" />
                {label}
              </NavLink>
            ))}
          </nav>
          <div className="ml-auto">
            <ThemeToggle />
          </div>
        </div>
      </header>
      <ScrollToTop />
      <main className="mx-auto max-w-7xl px-4 py-6 sm:px-6 sm:py-8">
        <Suspense fallback={<div className="py-24 text-center text-sm text-muted-foreground">Loading…</div>}>
          <Routes>
            <Route path="/" element={<Assistant />} />
            <Route path="/results" element={<Results />} />
            <Route path="*" element={<Assistant />} />
          </Routes>
        </Suspense>
      </main>
    </div>
  )
}
