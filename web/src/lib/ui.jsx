// Shared primitives. Kept deliberately small: a handful of components used
// everywhere, so the whole app reads as one system.

import { useEffect, useRef, useState } from 'react'

/*
  Verdict is a *status* palette. Each verdict carries a glyph and a word as well
  as a colour, so it never depends on colour alone — which also makes it legible
  in greyscale print and to colour-blind readers.
*/
export const VERDICT = {
  PASS: { glyph: '✓', label: 'Pass', fg: 'var(--color-pass)', tint: 'var(--color-pass-tint)' },
  WARN: { glyph: '!', label: 'Warn', fg: 'var(--color-warn)', tint: 'var(--color-warn-tint)' },
  FAIL: { glyph: '✕', label: 'Fail', fg: 'var(--color-fail)', tint: 'var(--color-fail-tint)' },
  INFO: { glyph: 'i', label: 'Info', fg: 'var(--color-info)', tint: 'var(--color-info-tint)' },
  UNKNOWN: { glyph: '?', label: 'Not run', fg: 'var(--color-none)', tint: 'var(--color-none-tint)' },
  'N/A': { glyph: '–', label: 'N/A', fg: 'var(--color-none)', tint: 'var(--color-none-tint)' },
}
export const verdictOf = (v) => VERDICT[v] || VERDICT.UNKNOWN

export function VerdictPill({ verdict, size = 'sm', className = '' }) {
  const v = verdictOf(verdict)
  const pad = size === 'lg' ? 'px-3 py-1 text-sm' : 'px-2 py-0.5 text-[11px]'
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full font-semibold tracking-wide ${pad} ${className}`}
      style={{ color: v.fg, background: v.tint }}
    >
      <span aria-hidden="true" className="font-bold leading-none">{v.glyph}</span>
      {verdict}
    </span>
  )
}

export function VerdictDot({ verdict, title }) {
  const v = verdictOf(verdict)
  return (
    <span
      role="img"
      aria-label={title || v.label}
      title={title || v.label}
      className="inline-flex h-4 w-4 flex-none items-center justify-center rounded-full text-[9px] font-bold text-white"
      style={{ background: v.fg }}
    >
      <span aria-hidden="true">{v.glyph}</span>
    </span>
  )
}

export function Card({ className = '', children, ...rest }) {
  return (
    <div
      className={`rounded-[14px] border border-[var(--color-line)] bg-[var(--color-surface)] ${className}`}
      {...rest}
    >
      {children}
    </div>
  )
}

export function SectionTitle({ children, right }) {
  return (
    <div className="mb-3 flex items-baseline justify-between gap-4">
      <h2 className="text-[13px] font-semibold uppercase tracking-[0.09em] text-[var(--color-faint)]">
        {children}
      </h2>
      {right}
    </div>
  )
}

export function Button({ variant = 'ghost', className = '', as: As = 'button', ...rest }) {
  const base =
    'inline-flex items-center justify-center gap-2 rounded-[10px] px-3.5 py-2 text-sm font-semibold transition-colors disabled:opacity-50 disabled:cursor-not-allowed'
  const styles = {
    primary:
      'bg-[var(--color-accent)] text-white hover:bg-[var(--color-accent-600)]',
    ghost:
      'border border-[var(--color-line)] bg-[var(--color-surface)] text-[var(--color-ink)] hover:border-[var(--color-faint)]',
    quiet: 'text-[var(--color-muted)] hover:text-[var(--color-ink)]',
  }
  return <As className={`${base} ${styles[variant]} ${className}`} {...rest} />
}

/** A hero number with a caption; used where a chart would be overkill. */
export function StatTile({ label, value, unit, caption, tone, bar }) {
  return (
    <Card className="p-4">
      <div className="text-[13px] text-[var(--color-muted)]">{label}</div>
      <div className="mt-1 flex items-baseline gap-1">
        <span
          className="text-[30px] font-bold leading-none tracking-tight tabular-nums"
          style={tone ? { color: tone } : undefined}
        >
          {value}
        </span>
        {unit && <span className="text-sm font-medium text-[var(--color-muted)]">{unit}</span>}
      </div>
      {caption && <div className="mt-1.5 text-xs text-[var(--color-faint)]">{caption}</div>}
      {bar != null && (
        <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-[var(--color-well)]">
          <div
            className="h-full rounded-full"
            style={{ width: `${Math.max(0, Math.min(1, bar.value)) * 100}%`, background: bar.color }}
          />
        </div>
      )}
    </Card>
  )
}

/** Hover tooltip anchored to the pointer — used by every chart mark. */
export function useTooltip() {
  const [tip, setTip] = useState(null)
  const show = (e, content) =>
    setTip({ x: e.clientX, y: e.clientY, content })
  const hide = () => setTip(null)
  const node = tip ? (
    <div
      role="tooltip"
      className="pointer-events-none fixed z-50 max-w-[260px] rounded-lg border border-[var(--color-line)] bg-[var(--color-surface)] px-2.5 py-1.5 text-xs shadow-lg"
      style={{ left: Math.min(tip.x + 12, window.innerWidth - 280), top: tip.y + 14 }}
    >
      {tip.content}
    </div>
  ) : null
  return { show, hide, node }
}

/** Click any figure to inspect it full-screen. */
export function Lightbox({ src, alt, onClose }) {
  const ref = useRef(null)
  useEffect(() => {
    const onKey = (e) => e.key === 'Escape' && onClose()
    document.addEventListener('keydown', onKey)
    ref.current?.focus()
    return () => document.removeEventListener('keydown', onKey)
  }, [onClose])
  if (!src) return null
  return (
    <div
      ref={ref}
      tabIndex={-1}
      role="dialog"
      aria-label={alt}
      onClick={onClose}
      className="fixed inset-0 z-50 grid cursor-zoom-out place-items-center bg-black/85 p-6 backdrop-blur-sm"
    >
      <img
        src={src}
        alt={alt}
        className="max-h-[92vh] w-auto max-w-[95vw] rounded-xl bg-[#0b0a09] p-3 shadow-2xl"
      />
    </div>
  )
}

export function Spinner({ label = 'Loading' }) {
  return (
    <div className="flex items-center gap-3 p-8 text-sm text-[var(--color-muted)]">
      <span
        aria-hidden="true"
        className="h-4 w-4 animate-spin rounded-full border-2 border-[var(--color-line)] border-t-[var(--color-accent)]"
      />
      {label}…
    </div>
  )
}

export function ErrorNote({ error, hint }) {
  return (
    <Card className="border-[var(--color-fail)]/30 bg-[var(--color-fail-tint)] p-5">
      <div className="text-sm font-semibold" style={{ color: 'var(--color-fail)' }}>
        Could not load this view
      </div>
      <p className="mt-1 text-sm text-[var(--color-ink)]">{String(error?.message || error)}</p>
      {hint && <p className="mt-2 text-xs text-[var(--color-muted)]">{hint}</p>}
    </Card>
  )
}

export function EmptyState({ title, children }) {
  return (
    <Card className="p-10 text-center">
      <div className="text-base font-semibold">{title}</div>
      <div className="mx-auto mt-2 max-w-md text-sm text-[var(--color-muted)]">{children}</div>
    </Card>
  )
}
