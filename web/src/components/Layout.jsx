import { useEffect, useState } from 'react'
import { NavLink, useParams } from 'react-router-dom'
import { VerdictDot } from '../lib/ui.jsx'

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
        `flex items-center gap-2.5 rounded-[9px] px-2.5 py-2 text-sm font-medium transition-colors ${
          isActive
            ? 'bg-[var(--color-accent-050)] text-[var(--color-accent-600)]'
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
      className="rounded-[9px] border border-[var(--color-line)] p-2 text-[var(--color-muted)] transition-colors hover:text-[var(--color-ink)]"
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
      <aside className="no-print sticky top-0 hidden h-screen w-[236px] flex-none flex-col overflow-y-auto border-r border-[var(--color-line)] bg-[var(--color-surface)] md:flex">
        <div className="flex items-center gap-2.5 border-b border-[var(--color-line)] px-4 py-4">
          <Logo />
          <span className="flex flex-col leading-tight">
            <b className="text-[15px] font-bold tracking-wide">OSIPI</b>
            <small className="font-mono text-[10px] uppercase tracking-[0.09em] text-[var(--color-faint)]">
              QC-ToolBox v1.0
            </small>
          </span>
        </div>

        <nav className="flex flex-col gap-0.5 p-2.5">
          <NavItem to="/" d={Icon.grid} end>Overview</NavItem>
          <NavItem to="/methodology" d={Icon.book}>Methodology</NavItem>
          <NavItem to="/upload" d={Icon.plus}>Grade a new scan</NavItem>
        </nav>

        {subjects.length > 0 && (
          <div className="px-2.5 pb-6">
            <div className="px-2.5 pb-1.5 pt-3 font-mono text-[10px] uppercase tracking-[0.1em] text-[var(--color-faint)]">
              Participants · {subjects.length}
            </div>
            {subjects.map((s) => (
              <NavLink
                key={s.sid}
                to={`/subject/${encodeURIComponent(s.sid)}`}
                title={`${s.sid} — ${s.verdict}`}
                className={({ isActive }) =>
                  `flex items-center gap-2 rounded-lg px-2.5 py-1.5 font-mono text-[13px] transition-colors ${
                    isActive
                      ? 'bg-[var(--color-accent-050)] font-semibold text-[var(--color-accent-600)]'
                      : 'text-[var(--color-muted)] hover:bg-[var(--color-well)] hover:text-[var(--color-ink)]'
                  }`
                }
              >
                <VerdictDot verdict={s.verdict} title={`${s.sid} — ${s.verdict}`} />
                <span className="truncate">{s.sid}</span>
              </NavLink>
            ))}
          </div>
        )}
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="no-print sticky top-0 z-20 flex items-center gap-3 border-b border-[var(--color-line)] bg-[color-mix(in_srgb,var(--color-paper)_92%,transparent)] px-5 py-3 backdrop-blur">
          <span className="font-bold tracking-tight text-[var(--color-accent)]">QC-ToolBox</span>
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
          <div className="flex-1" />
          <ThemeToggle />
          {cohort && (
            <button
              type="button"
              onClick={onOpenThresholds}
              className="inline-flex items-center gap-2 rounded-[9px] border border-[var(--color-line)] bg-[var(--color-surface)] px-3 py-2 text-sm font-semibold transition-colors hover:border-[var(--color-faint)]"
            >
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                   strokeWidth="2" strokeLinecap="round" aria-hidden="true">
                <circle cx="12" cy="12" r="3" />
                <path d="M12 2v3M12 19v3M4.2 4.2l2.1 2.1M17.7 17.7l2.1 2.1M2 12h3M19 12h3M4.2 19.8l2.1-2.1M17.7 6.3l2.1-2.1" />
              </svg>
              Thresholds
            </button>
          )}
        </header>
        <main className="mx-auto w-full max-w-[1500px] flex-1 px-5 pb-16 pt-6">{children}</main>
      </div>
    </div>
  )
}

const cap = (s) => (s ? s[0].toUpperCase() + s.slice(1) : '')

function Chip({ children, accent, title }) {
  return (
    <span
      title={title}
      className={`hidden whitespace-nowrap rounded-full border px-2.5 py-1 font-mono text-[11px] sm:inline ${
        accent
          ? 'border-[var(--color-accent)]/40 bg-[var(--color-accent-050)] text-[var(--color-accent-600)]'
          : 'border-[var(--color-line)] bg-[var(--color-surface)] text-[var(--color-muted)]'
      }`}
    >
      {children}
    </span>
  )
}
