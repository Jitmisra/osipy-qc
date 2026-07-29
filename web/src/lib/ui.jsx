// Shared primitives. Kept deliberately small: a handful of components used
// everywhere, so the whole app reads as one system.

import { useEffect, useRef, useState } from 'react'

/*
  Verdict is a *status* palette. Each verdict carries a glyph and a word as well
  as a colour, so it never depends on colour alone — which also makes it legible
  in greyscale print and to colour-blind readers.
*/
export const VERDICT = {
  PASS: { glyph: '✓', label: 'Pass', fg: 'var(--color-pass)', mark: 'var(--color-pass-mark)', tint: 'var(--color-pass-tint)', shape: 'circle' },
  WARN: { glyph: '!', label: 'Warn', fg: 'var(--color-warn)', mark: 'var(--color-warn-mark)', tint: 'var(--color-warn-tint)', shape: 'diamond' },
  FAIL: { glyph: '✕', label: 'Fail', fg: 'var(--color-fail)', mark: 'var(--color-fail-mark)', tint: 'var(--color-fail-tint)', shape: 'square' },
  INFO: { glyph: 'i', label: 'Info', fg: 'var(--color-info)', mark: 'var(--color-info-mark)', tint: 'var(--color-info-tint)', shape: 'open' },
  UNKNOWN: { glyph: '?', label: 'Not run', fg: 'var(--color-none)', mark: 'var(--color-none-mark)', tint: 'var(--color-none-tint)', shape: 'open' },
  'N/A': { glyph: '–', label: 'N/A', fg: 'var(--color-none)', mark: 'var(--color-none-mark)', tint: 'var(--color-none-tint)', shape: 'open' },
}
export const verdictOf = (v) => VERDICT[v] || VERDICT.UNKNOWN

export function VerdictPill({ verdict, size = 'sm', className = '' }) {
  const v = verdictOf(verdict)
  const neutral = verdict === 'N/A' || verdict === 'UNKNOWN'
  const pad = size === 'lg' ? 'px-3 h-7 text-[13px]' : 'px-2 h-[22px] text-[11px] tracking-[.01em]'
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-[var(--radius-pill)] font-semibold ${pad} ${className}`}
      style={{
        // the inset ring is what keeps the pill readable when a browser drops
        // background colours in print
        color: neutral ? 'var(--color-muted)' : v.fg,
        background: v.tint,
        boxShadow: 'inset 0 0 0 1px color-mix(in srgb, currentColor 18%, transparent)',
      }}
    >
      <VerdictDot verdict={verdict} size={10} />
      {verdict}
    </span>
  )
}

/**
 * The signature mark: a SHAPE as well as a colour.
 *
 * Pass is a circle, warn a diamond, fail a rounded square, and anything not run
 * an open dashed ring. It reads at 8px, survives a greyscale photocopy, and
 * scales unchanged up to the 72px hero medallion — so status never depends on
 * hue alone.
 */
const MARK_PATH = {
  PASS: 'M4.5 8.6 7 11.1l4.6-5',
  WARN: 'M8 4.4v4.4M8 11.2v.1',
  FAIL: 'M5 5l6 6M11 5l-6 6',
  INFO: 'M8 7.2v4.2M8 4.6v.1',
  UNKNOWN: 'M6 6.2a2 2 0 1 1 2 2.4v.9M8 11.4v.1',
  'N/A': 'M4.8 8h6.4',
}

export function VerdictDot({ verdict, title, size = 16 }) {
  const v = verdictOf(verdict)
  const open = v.shape === 'open'
  const d = MARK_PATH[verdict] || MARK_PATH.UNKNOWN
  // stroke scales with the mark so a 56px medallion is not a hairline
  const w = size >= 48 ? 1.6 : size >= 24 ? 1.9 : 2.1
  return (
    <span
      role="img"
      aria-label={title || v.label}
      title={title || v.label}
      className={`vshape vshape-${v.shape} inline-flex flex-none items-center justify-center`}
      style={{
        width: size, height: size,
        background: open ? 'transparent' : v.mark,
        color: open ? v.fg : 'var(--color-on-status)',
        borderWidth: open ? (size >= 48 ? 2 : 1.5) : 0,
      }}
    >
      <svg viewBox="0 0 16 16" width="100%" height="100%" fill="none" stroke="currentColor"
           strokeWidth={w} strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
        <path d={d} />
      </svg>
    </span>
  )
}

export function Card({ className = '', interactive, children, ...rest }) {
  return (
    <div className={`panel ${interactive ? 'panel--link' : ''} ${className}`} {...rest}>
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

export function Button({ variant = 'ghost', size, className = '', as: As = 'button', ...rest }) {
  const base =
    'inline-flex items-center justify-center gap-2 rounded-[var(--radius-md)] px-3.5 h-9 text-[13.5px] font-semibold ' +
    'transition-[background-color,border-color,color,box-shadow,transform] duration-[120ms] ease-[var(--ease-out)] ' +
    'disabled:opacity-45 disabled:cursor-not-allowed active:scale-[.98]'
  const styles = {
    // the inset top highlight does the lifting; no gradient anywhere
    primary: 'text-white bg-[var(--color-accent-fill,var(--color-accent))] hover:brightness-110 ' +
             'shadow-[inset_0_1px_0_rgba(255,255,255,.28),0_1px_2px_rgba(20,26,48,.20)]',
    ghost: 'border border-[var(--color-line)] bg-[var(--color-surface)] text-[var(--color-ink)] ' +
           'shadow-[var(--shadow-e1)] hover:border-[var(--color-line-strong)] hover:shadow-[var(--shadow-e2)]',
    // no border ring to fight the panel it sits inside
    subtle: 'raised text-[var(--color-ink)] hover:brightness-[.97]',
    quiet: 'text-[var(--color-muted)] hover:text-[var(--color-ink)] hover:bg-[var(--color-well)]',
  }
  const sizes = { sm: 'h-8 px-3 text-[12.5px] rounded-[var(--radius-sm)]', lg: 'h-11 px-5 text-[15px] rounded-[14px]' }
  return <As className={`${base} ${styles[variant]} ${sizes[size] || ''} ${className}`} {...rest} />
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
      className="scrim scrim--lightbox fixed inset-0 z-50 grid cursor-zoom-out place-items-center p-6"
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
