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

/**
 * Where a value sits relative to the range it is graded against.
 *
 * A bare number ("3472.8") tells a reader nothing about whether that is fine or
 * catastrophic. The meter draws the expected band on a fixed axis and pins the
 * marker at the edge when the value runs past it, so "far out of range" stays
 * visible instead of the scale quietly stretching to accommodate it.
 */
export function BandMeter({ value, band, axis, color, className = '' }) {
  if (!band || !axis || typeof value !== 'number') return null
  const [amin, amax] = axis
  const span = amax - amin || 1
  const at = (v) => Math.max(0, Math.min(1, (v - amin) / span)) * 100
  const over = value > amax
  const under = value < amin
  return (
    <div className={`relative mt-2.5 h-1.5 rounded-full bg-[var(--color-well)] ${className}`}>
      <div
        className="absolute inset-y-0 rounded-full"
        style={{
          left: `${at(band[0])}%`,
          right: `${100 - at(band[1])}%`,
          background: 'var(--color-pass)',
          opacity: 0.22,
        }}
      />
      <div
        className="absolute top-1/2 h-[11px] w-[3px] -translate-y-1/2 rounded-full ring-2 ring-[var(--color-surface)]"
        style={{ left: `calc(${at(value)}% - 1.5px)`, background: color }}
        title={`${value} (expected ${band[0]}–${band[1]})`}
      />
      {(over || under) && (
        <span
          aria-hidden="true"
          className="absolute top-1/2 -translate-y-1/2 text-[10px] font-bold leading-none"
          style={{ color, [over ? 'right' : 'left']: '-9px' }}
        >
          {over ? '▸' : '◂'}
        </span>
      )}
    </div>
  )
}

/**
 * The cohort's verdict split as one bar. Replaces three separate percentage
 * tiles: the proportions are the message, and a single bar shows them at a
 * glance while still naming every count in text.
 */
export function VerdictBar({ counts, total, onSelect, active }) {
  const order = ['FAIL', 'WARN', 'PASS']
  const present = order.filter((k) => (counts[k] || 0) > 0)
  return (
    <div>
      <div className="flex h-2.5 gap-[3px] overflow-hidden rounded-full">
        {present.map((k) => {
          const v = verdictOf(k)
          const share = ((counts[k] || 0) / Math.max(total, 1)) * 100
          return (
            <button
              key={k}
              type="button"
              onClick={() => onSelect?.(active === k ? 'all' : k)}
              aria-label={`${counts[k]} ${k} of ${total}`}
              title={`${counts[k]} ${k} — click to filter`}
              className="h-full rounded-full transition-opacity hover:opacity-80"
              style={{ width: `${share}%`, background: v.fg, opacity: active && active !== k ? 0.3 : 1 }}
            />
          )
        })}
      </div>
      <div className="mt-2.5 flex flex-wrap gap-x-5 gap-y-1">
        {order.map((k) => {
          const v = verdictOf(k)
          const n = counts[k] || 0
          return (
            <span key={k} className="flex items-baseline gap-1.5 text-[13px]">
              <span aria-hidden="true" className="inline-block h-2 w-2 translate-y-[-1px] rounded-full" style={{ background: v.fg }} />
              <b className="num font-semibold" style={{ color: n ? v.fg : 'var(--color-faint)' }}>{n}</b>
              <span className="text-[var(--color-muted)]">{v.label.toLowerCase()}</span>
            </span>
          )
        })}
      </div>
    </div>
  )
}

/** Placeholder that matches the shape of the real content, not a spinner. */
export function Skeleton({ className = '', rounded = 'rounded-lg' }) {
  return <div className={`skeleton ${rounded} ${className}`} />
}

export function OverviewSkeleton() {
  return (
    <div className="animate-none">
      <Skeleton className="mb-6 h-9 w-64" />
      <div className="mb-4 grid gap-3 md:grid-cols-3">
        {[0, 1, 2].map((i) => <Skeleton key={i} className="h-[104px]" rounded="rounded-[14px]" />)}
      </div>
      <div className="grid gap-4 xl:grid-cols-[1fr_330px]">
        <Skeleton className="h-[340px]" rounded="rounded-[14px]" />
        <Skeleton className="h-[340px]" rounded="rounded-[14px]" />
      </div>
    </div>
  )
}

export function SubjectSkeleton() {
  return (
    <div>
      <Skeleton className="mb-5 h-9 w-72" />
      <Skeleton className="mb-4 h-[168px]" rounded="rounded-[14px]" />
      <div className="mb-5 grid gap-3 md:grid-cols-3 xl:grid-cols-5">
        {[0, 1, 2, 3, 4].map((i) => <Skeleton key={i} className="h-[112px]" rounded="rounded-[14px]" />)}
      </div>
      <div className="flex flex-col gap-2.5">
        {[0, 1, 2].map((i) => <Skeleton key={i} className="h-[86px]" rounded="rounded-[14px]" />)}
      </div>
    </div>
  )
}
