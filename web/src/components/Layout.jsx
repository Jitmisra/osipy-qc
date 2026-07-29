import { useEffect, useState } from 'react'
import { Link, NavLink, useParams } from 'react-router-dom'
import { Button, VerdictDot } from '../lib/ui.jsx'

function Logo() {
  return (
    <svg width="28" height="28" viewBox="0 0 40 40" fill="none" aria-hidden="true" className="flex-none">
      <defs>
        <linearGradient id="osipi" x1="4" y1="8" x2="36" y2="32" gradientUnits="userSpaceOnUse">
          <stop stopColor="#F0479B" />
          <stop offset="1" stopColor="#7C3AED" />
        </linearGradient>
      </defs>
      <path
        d="M20 20c-3.4-4.6-6.4-7-9.6-7C6.3 13 4 16.1 4 20s2.3 7 6.4 7c3.2 0 6.2-2.4 9.6-7 3.4-4.6 6.4-7 9.6-7C33.7 13 36 16.1 36 20s-2.3 7-6.4 7c-3.2 0-6.2-2.4-9.6-7z"
        stroke="url(#osipi)" strokeWidth="3.4" strokeLinecap="round" fill="none"
      />
    </svg>
  )
}

const Icon = {
  grid: 'M3 3h7v7H3zM14 3h7v7h-7zM3 14h7v7H3zM14 14h7v7h-7z',
  book: 'M4 5a2 2 0 0 1 2-2h13v18H6a2 2 0 0 1-2-2zM19 3v18',
  plus: 'M12 5v14M5 12h14',
  gear: 'M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6z',
}

function NavItem({ to, d, children, end }) {
  return (
    <NavLink
      to={to}
      end={end}
      className={({ isActive }) =>
        `relative flex h-9 items-center gap-2.5 rounded-[var(--radius-md)] px-2.5 text-[13.5px] font-semibold transition-colors ${
          isActive
            ? 'bg-[var(--color-accent-050)] text-[var(--color-accent-600)] before:absolute before:left-0 before:top-1.5 before:bottom-1.5 before:w-[3px] before:rounded-full before:bg-[var(--color-accent)]'
            : 'text-[var(--color-ink)] hover:bg-[var(--color-well)]'
        }`
      }
    >
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor"
           strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
        <path d={d} />
      </svg>
      {children}
    </NavLink>
  )
}

function ThemeToggle() {
  const [theme, setTheme] = useState(
    () => localStorage.getItem('osipi-theme') ||
      (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'),
  )
  useEffect(() => {
    document.documentElement.dataset.theme = theme
    localStorage.setItem('osipi-theme', theme)
  }, [theme])
  const dark = theme === 'dark'
  return (
    <button
      type="button"
      onClick={() => setTheme(dark ? 'light' : 'dark')}
      aria-label={dark ? 'Switch to light theme' : 'Switch to dark theme'}
      title={dark ? 'Light theme' : 'Dark theme'}
      className="h-8 w-8 rounded-[var(--radius-sm)] border border-[var(--color-line)] bg-[var(--color-surface)] p-0 text-[var(--color-muted)] transition-colors hover:text-[var(--color-ink)]"
    >
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor"
           strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
        {dark ? (
          <>
            <circle cx="12" cy="12" r="4" />
            <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
          </>
        ) : (
          <path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" />
        )}
      </svg>
    </button>
  )
}

export default function Layout({ cohort, onOpenThresholds, children }) {
  const { sid } = useParams()
  const subjects = cohort?.subjects || []

  return (
    <div className="flex min-h-screen">
      <aside className="no-print glass-chrome sticky top-3 ml-2 hidden sm:ml-3 h-[calc(100vh-24px)] w-[248px] flex-none flex-col overflow-y-auto rounded-[var(--radius-pane)] md:flex">
        <div className="flex items-center gap-2.5 px-4 py-4">
          <Logo />
          <span className="flex flex-col leading-tight">
            <b className="text-[15px] font-bold tracking-wide">OSIPI</b>
            <small className="font-mono text-[10px] uppercase tracking-[0.09em] text-[var(--color-faint)]">
              QC-ToolBox v1.0
            </small>
          </span>
        </div>

        <div className="rule-h mx-3" />

        <nav className="flex flex-col gap-0.5 p-2.5">
          <NavItem to="/" d={Icon.grid} end>Overview</NavItem>
        </nav>

        {subjects.length > 0 && (
          <div className="px-2.5 pb-6">
            <div className="flex items-baseline justify-between px-2.5 pb-1.5 pt-3">
              <span className="font-mono text-[10px] uppercase tracking-[0.1em] text-[var(--color-faint)]">
                Participants
              </span>
              <span className="num font-mono text-[10px] text-[var(--color-faint)]">{subjects.length}</span>
            </div>
            {subjects.map((s) => (
              <NavLink
                key={s.sid}
                to={`/subject/${encodeURIComponent(s.sid)}`}
                title={`${s.sid} — ${s.verdict}${s.qei != null ? ` · QEI ${s.qei.toFixed(2)}` : ''}`}
                className={({ isActive }) =>
                  `flex h-8 items-center gap-2 rounded-[var(--radius-sm)] px-2.5 font-mono text-[12.5px] transition-colors ${
                    isActive
                      ? 'bg-[var(--color-accent-050)] font-semibold text-[var(--color-accent-600)]'
                      : 'text-[var(--color-muted)] hover:bg-[var(--color-well)] hover:text-[var(--color-ink)]'
                  }`
                }
              >
                <VerdictDot verdict={s.verdict} title={`${s.sid} — ${s.verdict}`} />
                <span className="min-w-0 flex-1 truncate">{s.sid}</span>
                {/* the anchor metric, so the list is scannable without leaving the page */}
                <span className="num text-[11px] tabular-nums text-[var(--color-faint)]">
                  {s.qei == null ? '—' : s.qei.toFixed(2)}
                </span>
              </NavLink>
            ))}
          </div>
        )}
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="no-print glass-chrome sticky top-3 z-30 mx-2 mt-2 flex h-14 min-w-0 sm:mx-3 sm:mt-3 items-center gap-2 rounded-[var(--radius-pane)] px-3 sm:gap-3 sm:px-4">
          <span className="hidden flex-none font-bold tracking-[-.01em] text-[var(--color-accent)] sm:inline">QC-ToolBox</span>
          {cohort && (
            <>
              <Chip>🧠 {cap(cohort.organ)}</Chip>
              <Chip title={cohort.strict
                ? 'Uncalibrated cut-offs may raise a failure'
                : 'Uncalibrated cut-offs are demoted to a warning'}>
                {cohort.strict ? 'strict grading' : 'lenient grading'}
              </Chip>
              {cohort.customThresholds && (
                <Chip accent>custom thresholds</Chip>
              )}
            </>
          )}
          <div className="min-w-0 flex-1" />
          <div className="flex flex-none items-center gap-2">
          <ThemeToggle />
          {cohort && (
            <button
              type="button"
              onClick={onOpenThresholds}
              className="inline-flex h-8 items-center gap-2 rounded-[var(--radius-sm)] border border-[var(--color-line)] bg-[var(--color-surface)] px-3 text-[12.5px] font-semibold transition-colors hover:border-[var(--color-line-strong)]"
            >
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                   strokeWidth="2" strokeLinecap="round" aria-hidden="true">
                <circle cx="12" cy="12" r="3" />
                <path d="M12 2v3M12 19v3M4.2 4.2l2.1 2.1M17.7 17.7l2.1 2.1M2 12h3M19 12h3M4.2 19.8l2.1-2.1M17.7 6.3l2.1-2.1" />
              </svg>
              Thresholds
            </button>
          )}
          {/* the one thing that creates something: primary, and last, so it is
              where the eye lands after reading the bar left to right */}
          <Button as={Link} to="/upload" variant="primary">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                 strokeWidth="2.2" strokeLinecap="round" aria-hidden="true">
              <path d={Icon.plus} />
            </svg>
            Grade a new scan
          </Button>
          </div>
        </header>
        <main className="mx-auto w-full max-w-[1560px] flex-1 px-6 pb-20 pt-6">{children}</main>
      </div>
    </div>
  )
}

const cap = (s) => (s ? s[0].toUpperCase() + s.slice(1) : '')

function Chip({ children, accent, title }) {
  return (
    <span
      title={title}
      className={`hidden h-7 flex-none items-center whitespace-nowrap rounded-[var(--radius-pill)] px-2.5 font-mono text-[11px] lg:inline-flex ${
        accent
          ? 'bg-[var(--color-accent-050)] text-[var(--color-accent-600)]'
          : 'raised text-[var(--color-muted)]'
      }`}
    >
      {children}
    </span>
  )
}
