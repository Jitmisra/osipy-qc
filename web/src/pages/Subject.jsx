import { useEffect, useMemo, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { fetchSubject, figureUrl } from '../lib/api.js'
import {
  Card, SectionTitle, Button, VerdictPill, VerdictDot, SubjectSkeleton, ErrorNote, Lightbox,
  BandMeter, verdictOf,
} from '../lib/ui.jsx'

const GLOSS = {
  PASS: 'This scan looks usable. Every applicable check passed.',
  WARN: 'Usable with caveats — one or more checks flagged something worth reviewing.',
  FAIL: 'At least one check found a disqualifying problem. Inspect before using this scan.',
  UNKNOWN: 'Not enough was provided to reach a verdict.',
}

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

export default function Subject({ cohort }) {
  const { sid } = useParams()
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const [zoom, setZoom] = useState(null)

  useEffect(() => {
    setData(null)
    setError(null)
    fetchSubject(sid).then(setData).catch(setError)
  }, [sid])

  const { prev, next } = useMemo(() => {
    const ids = (cohort?.subjects || []).map((s) => s.sid)
    const i = ids.indexOf(sid)
    return { prev: i > 0 ? ids[i - 1] : null, next: i >= 0 && i < ids.length - 1 ? ids[i + 1] : null }
  }, [cohort, sid])

  if (error) return <ErrorNote error={error} hint="Check that this participant exists in the loaded cohort." />
  if (!data) return <SubjectSkeleton />

  const v = verdictOf(data.verdict)
  const byStream = data.checks.reduce((acc, c) => {
    ;(acc[c.streamTitle] ||= []).push(c)
    return acc
  }, {})

  return (
    <>
      <nav className="mb-3 font-mono text-[12px] text-[var(--color-muted)]">
        <Link to="/" className="hover:underline">Overview</Link>
        <span className="px-1.5">/</span>
        <span className="font-semibold text-[var(--color-accent-600)]">{data.sid}</span>
      </nav>

      <div className="mb-5 flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-[28px] font-bold tracking-tight">Participant quality report</h1>
          <p className="mt-1 text-sm text-[var(--color-muted)]">
            <b className="font-mono text-[var(--color-ink)]">{data.sid}</b> · {data.population} profile ·{' '}
            <b style={{ color: v.fg }}>{data.verdict}</b>
          </p>
        </div>
        <div className="no-print flex gap-2">
          {prev
            ? <Button as={Link} to={`/subject/${encodeURIComponent(prev)}`}>← Prev</Button>
            : <Button disabled>← Prev</Button>}
          {next
            ? <Button as={Link} to={`/subject/${encodeURIComponent(next)}`}>Next →</Button>
            : <Button disabled>Next →</Button>}
          <Button variant="primary" onClick={() => window.print()}>Print / Save PDF</Button>
        </div>
      </div>

      {/* verdict hero */}
      {/*
        The verdict, stated once and calmly.

        It used to be a saturated pink slab the width of the page. A whole panel
        flooded with status colour competes with the terracotta the rest of the
        product is built from, and it shouts the one thing the reader already
        knows from the sidebar. The status now lives in the medallion, the word
        and a spine; the surface stays neutral.
      */}
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
                {GLOSS[data.verdict] || ''}
              </p>
            </div>
          </div>

          {/* the split, as a compact readout rather than five pills */}
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

        {data.drivers.length > 0 && (
          <div className="flex flex-wrap items-center gap-x-2 gap-y-1.5 border-t border-[var(--color-line)] bg-[var(--color-well)] px-7 py-3">
            <span className="text-[13px] text-[var(--color-muted)]">Needs attention</span>
            {data.drivers.map((d) => (
              <a
                key={d.id}
                href={`#chk-${d.id.replace(/\./g, '-')}`}
                className="inline-flex items-center gap-1.5 rounded-[var(--radius-sm)] bg-[var(--color-surface)] px-2.5 py-1 text-[13px] font-medium transition-colors hover:bg-[var(--color-accent-050)]"
              >
                <VerdictDot verdict={d.verdict} size={12} />
                {d.label}
              </a>
            ))}
          </div>
        )}
      </Card>

      {/* findings before images: the checks are the diagnosis, the pictures are evidence */}
      {Object.entries(byStream).map(([stream, checks]) => (
        <section key={stream} className="mb-6">
          <SectionTitle>Checks · {stream}</SectionTitle>
          <div className="grid gap-3 lg:grid-cols-2 2xl:grid-cols-3">
            {checks.map((c) => <CheckCard key={c.id} check={c} />)}
          </div>
        </section>
      ))}

      {data.figures.length > 0 && (
        <section className="mb-6">
          <SectionTitle right={<span className="text-[11px] text-[var(--color-faint)]">click to enlarge</span>}>
            Images
          </SectionTitle>
          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            {data.figures.map((f) => {
              const src = figureUrl(data.sid, `${f.name}.${f.kind === 'png' ? 'png' : 'svg'}`)
              return (
                <Card key={f.name} className="overflow-hidden shadow-[var(--shadow-e1)]">
                  <button
                    onClick={() => setZoom({ src, alt: f.title })}
                    className="block w-full cursor-zoom-in"
                    aria-label={`Enlarge: ${f.title}`}
                  >
                    <img src={src} alt={f.title} loading="lazy"
                         className={`block h-auto w-full ${f.kind === 'png' ? 'bg-[#0b0a09]' : 'bg-[var(--color-surface)] p-1'}`} />
                  </button>
                  {/* The window is per-scan, so without a labelled scale the
                      colours have no units and are quietly incomparable. */}
                  {f.window && (
                    <div className="px-3 pt-2.5">
                      <div
                        className="h-2 w-full rounded-full"
                        style={{
                          background: `linear-gradient(90deg, ${f.window.ramp
                            .map((r) => `${r.color} ${r.at * 100}%`)
                            .join(', ')})`,
                        }}
                      />
                      <div className="num mt-1 flex justify-between font-mono text-[10px] text-[var(--color-faint)]">
                        <span>{Math.round(f.window.vmin).toLocaleString()}</span>
                        <span>{f.window.unit}</span>
                        <span>{Math.round(f.window.vmax).toLocaleString()}</span>
                      </div>
                    </div>
                  )}
                  <div className="border-t border-[var(--color-line)] p-3">
                    <div className="text-[13px] font-semibold">{f.title}</div>
                    <p className="mt-0.5 text-[12px] leading-snug text-[var(--color-muted)]">{f.caption}</p>
                    {f.window && (
                      <p className="mt-1 text-[11px] leading-snug text-[var(--color-faint)]">
                        {f.window.note}
                      </p>
                    )}
                  </div>
                </Card>
              )
            })}
          </div>
        </section>
      )}

      <Card className="border-l-[3px] border-l-[var(--color-accent)] bg-[var(--color-accent-050)] p-4 text-[13px] leading-relaxed">
        A verdict marked <b>provisional</b> was decided by an <b>uncalibrated</b> cut-off — an
        engineering default with no published derivation. Under strict grading it can raise a
        failure; turn strict off and it becomes a warning instead. Checks marked <b>N/A</b> or{' '}
        <b>INFO</b> are excluded from the overall verdict. Every threshold, with the source it
        came from, is listed under <b>Thresholds</b>.
      </Card>

      <Lightbox src={zoom?.src} alt={zoom?.alt} onClose={() => setZoom(null)} />
    </>
  )
}
