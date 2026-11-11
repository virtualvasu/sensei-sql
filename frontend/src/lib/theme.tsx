import { createContext, useContext, useEffect, useState, type ReactNode } from 'react'

type Theme = 'light' | 'dark' | 'system'
const KEY = 'sensei-theme'

const Ctx = createContext<{ theme: Theme; setTheme: (t: Theme) => void }>({ theme: 'system', setTheme: () => {} })

function stored(): Theme {
  try {
    const v = localStorage.getItem(KEY)
    if (v === 'light' || v === 'dark' || v === 'system') return v
  } catch {
    /* storage blocked: fall back to the system theme */
  }
  return 'system'
}

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setThemeState] = useState<Theme>(stored)

  useEffect(() => {
    const media = window.matchMedia('(prefers-color-scheme: dark)')
    const apply = () => document.documentElement.classList.toggle('dark', theme === 'dark' || (theme === 'system' && media.matches))
    apply()
    media.addEventListener('change', apply)
    return () => media.removeEventListener('change', apply)
  }, [theme])

  const setTheme = (t: Theme) => {
    setThemeState(t)
    try {
      localStorage.setItem(KEY, t)
    } catch {
      /* ignore */
    }
  }
  return <Ctx.Provider value={{ theme, setTheme }}>{children}</Ctx.Provider>
}

export const useTheme = () => useContext(Ctx)
