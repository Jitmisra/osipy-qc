import { useState } from 'react'
import {
  Card, CoverageMeter, NotGradedStrip, VerdictDot, VerdictPill, verdictGloss, verdictOf,
} from '../lib/ui.jsx'

/*
  One scan's report body.

  Shared deliberately: the cohort page and the upload page render the same
  payload, so they must render it with the same components. They used to
  diverge — an upload returned a server-built HTML document in a foreign
  stylesheet — which is how the product ended up with two designs at once.
*/

/** The headline number a check produced, if it produced one worth showing. */
function headline(check) {
  const m = check.metric || {}
  const first = ['qei', 'sCoV_pct', 'spatial_snr', 'gm_wm_ratio', 'mean_gm_cbf',
                 'gm_coverage', 'negGM_fraction', 'dice', 'mean_fwd_mm', 'skewness']
    .find((k) => typeof m[k] === 'number')
  if (!first) return null
  const v = m[first]
  const unit = first.endsWith('_pct') ? '%' : first === 'gm_coverage' ? '%'
    : first === 'mean_gm_cbf' ? 'mL/100g/min' : first === 'mean_fwd_mm' ? 'mm' : ''
  const shown = first === 'gm_coverage' ? v * 100 : v
  const text = Math.abs(shown) >= 100 ? Math.round(shown).toLocaleString()
    : Math.abs(shown) >= 1 ? shown.toFixed(2)
    : Math.abs(shown) >= 0.001 ? shown.toFixed(3)
    : shown === 0 ? '0' : shown.toExponential(1)
  return { text, unit }
}

/**
 * One check, as a tile.
 *
 * The previous card was a full-bleed bar: one line of text across 1100px, nine
 * of them stacked, most of the width empty. A tile puts the number where the eye
 * goes — top right, at size — and lets three sit side by side, so the whole
 * stream is visible at once instead of scrolled through.
 */
function CheckCard({ check }) {
  const [open, setOpen] = useState(false)
  const v = verdictOf(check.verdict)
  const rows = Object.entries(check.metric || {})
  const big = headline(check)

  return (
    <Card
      id={`chk-${check.id.replace(/\./g, '-')}`}
      className="panel--card relative flex flex-col self-start overflow-hidden"
    >
      {/* the spine states the verdict before any text is read; a provisional one
          is striped, so an uncalibrated cut-off never looks evidence-backed */}
      <span
        aria-hidden="true"
        className="absolute inset-y-0 left-0 w-1"
        style={{
          ...(check.provisional
            ? { backgroundImage: `repeating-linear-gradient(45deg, ${v.fg} 0 5px, transparent 5px 10px)` }
            : { background: v.fg }),
          printColorAdjust: 'exact',
        }}
      />

      <div className="flex flex-1 flex-col p-5 pl-6">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <VerdictDot verdict={check.verdict} size={18} />
              <h3 className="truncate text-[16px] font-semibold tracking-[-.01em]">{check.label}</h3>
            </div>
            <div className="mt-1 flex items-center gap-2 font-mono text-[11px] text-[var(--color-faint)]">
              {check.id}
              {check.provisional && (
                <span title="Decided by an uncalibrated cut-off, so it is provisional rather than evidence-backed."
                      style={{ color: 'var(--color-warn)' }}>
                  provisional
                </span>
              )}
            </div>
          </div>

          {big && (
            <div className="flex flex-none items-baseline gap-1 text-right">
              <span className="num text-[26px] font-semibold leading-none tracking-[-.02em]"
                    style={{ color: v.fg }}>
                {big.text}
              </span>
              {big.unit && (
                <span className="text-[11px] text-[var(--color-muted)]">{big.unit}</span>
              )}
            </div>
          )}
        </div>

        <p className="mt-3 text-[14px] leading-relaxed text-[var(--color-ink)]">{check.reason}</p>

        {rows.length > 0 && (
          <div className="mt-3">
            <button
              onClick={() => setOpen((o) => !o)}
              aria-expanded={open}
              className="inline-flex items-center gap-1.5 rounded-[var(--radius-sm)] px-2 py-1 -ml-2 text-[12.5px] font-medium text-[var(--color-muted)] transition-colors hover:bg-[var(--color-well)] hover:text-[var(--color-ink)]"
            >
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                   strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round"
                   className={`transition-transform ${open ? 'rotate-90' : ''}`} aria-hidden="true">
                <path d="M9 6l6 6-6 6" />
              </svg>
              {rows.length} measurement{rows.length === 1 ? '' : 's'}
            </button>
            {open && (
              <dl className="mt-2 rounded-[var(--radius-sm)] bg-[var(--color-well)] px-3 py-1.5">
                {rows.map(([k, val]) => {
                  const numeric = typeof val === 'number'
                  return (
                    <div
                      key={k}
                      className={`border-b border-[var(--color-line)] py-1.5 last:border-0 ${
                        numeric ? 'flex items-baseline justify-between gap-4' : ''
                      }`}
                    >
                      <dt className="font-mono text-[12px] text-[var(--color-muted)]">{k}</dt>
                      {/* a measured number is right-aligned so decimals stack; a
                          sentence is not a number and reads left, like prose */}
                      <dd
                        className={
                          numeric
                            ? 'num flex-none text-right font-mono text-[12px] font-medium'
                            : 'mt-0.5 text-[12.5px] leading-relaxed text-[var(--color-ink)]'
                        }
                      >
                        {numeric ? Number(val.toFixed(4)) : String(val)}
                      </dd>
                    </div>
                  )
                })}
              </dl>
            )}
          </div>
        )}
      </div>
    </Card>
  )
}

/** The verdict, the drivers, and every check as a tile. */
export default function ScanReport({ data }) {
  const v = verdictOf(data.verdict)
  const byStream = data.checks.reduce((acc, c) => {
    ;(acc[c.streamTitle] ||= []).push(c)
    return acc
  }, {})

  return (
    <>
      <Card className="panel--hero relative mb-5 overflow-hidden">
        <span aria-hidden="true" className="absolute inset-y-0 left-0 w-1"
              style={{ background: v.fg, printColorAdjust: 'exact' }} />

        <div className="flex flex-wrap items-center justify-between gap-6 px-7 py-6">
          <div className="flex min-w-0 items-center gap-4">
            <VerdictDot verdict={data.verdict} size={52} />
            <div className="min-w-0">
              <div className="text-[32px] font-semibold leading-none tracking-[-.025em]"
                   style={{ color: v.fg }}>
                {data.verdict}
              </div>
              <p className="mt-1.5 text-[14.5px] text-[var(--color-muted)]">
                {verdictGloss(data.verdict, data.coverage)}
              </p>
              <CoverageMeter coverage={data.coverage} className="mt-2.5" />
            </div>
          </div>

          <dl className="flex flex-none flex-wrap items-center gap-x-6 gap-y-2">
            {Object.entries(data.summary || {}).map(([k, n]) => (
              <div key={k} className="flex items-center gap-2">
                <VerdictDot verdict={k} size={14} />
                <dd className="num text-[17px] font-semibold leading-none">{n}</dd>
                <dt className="text-[12px] text-[var(--color-muted)]">{k.toLowerCase()}</dt>
              </div>
            ))}
          </dl>
        </div>

        {data.drivers?.length > 0 && (
          <div className="flex flex-wrap items-center gap-x-2 gap-y-1.5 border-t border-[var(--color-line)] bg-[var(--color-well)] px-7 py-3">
            <span className="text-[13px] text-[var(--color-muted)]">Needs attention</span>
            {data.drivers.map((d) => (
              <a key={d.id} href={`#chk-${d.id.replace(/\./g, '-')}`}
                 className="inline-flex items-center gap-1.5 rounded-[var(--radius-sm)] bg-[var(--color-surface)] px-2.5 py-1 text-[13px] font-medium transition-colors hover:bg-[var(--color-accent-050)]">
                <VerdictDot verdict={d.verdict} size={12} />
                {d.label}
              </a>
            ))}
          </div>
        )}

        <NotGradedStrip coverage={data.coverage} checks={data.checks} />
      </Card>

      {Object.entries(byStream).map(([stream, checks]) => (
        <section key={stream} className="mb-6">
          <h2 className="mb-3 text-[13px] font-semibold uppercase tracking-[0.09em] text-[var(--color-muted)]">
            Checks — {stream}
          </h2>
          <div className="grid gap-3 lg:grid-cols-2 2xl:grid-cols-3">
            {checks.map((c) => <CheckCard key={c.id} check={c} />)}
          </div>
        </section>
      ))}
    </>
  )
}
