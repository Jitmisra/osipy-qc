// Hand-built SVG charts. No chart library: the shapes here are simple, and
// keeping them local means the visual language matches the rest of the app
// exactly (same status palette, same type scale, same hover behaviour).

import { Link } from 'react-router-dom'
import { Card, SectionTitle, verdictOf, useTooltip } from '../lib/ui.jsx'

/**
 * Every subject's QEI against the published cut-off.
 *
 * A distribution strip, deliberately not a time series: subjects carry no
 * acquisition dates, so an x-axis of "time" would be invented. One dot per
 * scan, coloured by its verdict, with the cut-off drawn as a reference line.
 */
export function QeiStrip({ subjects, cutoff = 0.5 }) {
  const tip = useTooltip()
  const W = 640
  const H = 132
  const padX = 34
  const baseY = 62
  const withQei = subjects.filter((s) => s.qei != null)
  const missing = subjects.length - withQei.length
  const x = (q) => padX + (W - 2 * padX) * Math.max(0, Math.min(1, q))

  return (
    <Card className="p-5">
      <SectionTitle
        right={
          <span className="font-mono text-[11px] text-[var(--color-faint)]">
            {withQei.length} of {subjects.length} scored
          </span>
        }
      >
        QEI across the cohort
      </SectionTitle>
      <p className="-mt-1 mb-2 text-xs text-[var(--color-muted)]">
        One dot per scan. Anything left of the dashed line is below the published
        cut-off of {cutoff} and is a likely reject.
      </p>

      {withQei.length === 0 ? (
        <p className="py-6 text-center text-sm text-[var(--color-faint)]">
          No scan in this cohort has a QEI — tissue maps are required to compute it.
        </p>
      ) : (
        <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" role="img"
             aria-label={`QEI distribution for ${withQei.length} scans against a cut-off of ${cutoff}`}>
          {/* recessive axis */}
          <line x1={padX} y1={baseY} x2={W - padX} y2={baseY}
                stroke="var(--color-line)" strokeWidth="2" strokeLinecap="round" />
          {[0, 0.5, 1].map((t) => (
            <text key={t} x={x(t)} y={baseY + 20} textAnchor="middle"
                  fontSize="10" fill="var(--color-faint)" fontFamily="var(--font-mono)">
              {t}
            </text>
          ))}
          {/* the one published reference on this chart */}
          <line x1={x(cutoff)} y1={baseY - 34} x2={x(cutoff)} y2={baseY + 7}
                stroke="var(--color-fail)" strokeWidth="1.5" strokeDasharray="4 3" />
          <text x={x(cutoff)} y={baseY - 40} textAnchor="middle" fontSize="10"
                fill="var(--color-fail)" fontFamily="var(--font-mono)">
            cut-off {cutoff}
          </text>
          {withQei.map((s, i) => {
            const v = verdictOf(s.verdict)
            const dy = (i % 2 ? -1 : 1) * (4 + (i % 3) * 5)
            return (
              <Link key={s.sid} to={`/subject/${encodeURIComponent(s.sid)}`}>
                <circle
                  cx={x(s.qei)} cy={baseY + dy} r="6"
                  fill={v.fg} fillOpacity="0.9"
                  stroke="var(--color-surface)" strokeWidth="2"
                  className="cursor-pointer"
                  onMouseEnter={(e) =>
                    tip.show(e, (
                      <span>
                        <b className="font-mono">{s.sid}</b> · QEI {s.qei.toFixed(3)} ·{' '}
                        <span style={{ color: v.fg }}>{s.verdict}</span>
                      </span>
                    ))
                  }
                  onMouseLeave={tip.hide}
                />
              </Link>
            )
          })}
        </svg>
      )}
      {missing > 0 && (
        <p className="mt-1 text-xs text-[var(--color-faint)]">
          {missing} scan{missing > 1 ? 's' : ''} could not be scored (no tissue maps supplied).
        </p>
      )}
      {tip.node}
    </Card>
  )
}

/**
 * Checks × subjects. A whole row lit up means a systemic problem with the
 * protocol rather than with one scan — which the per-scan views cannot show.
 */
export function CheckMatrix({ matrix, subjects }) {
  const tip = useTooltip()
  if (!matrix?.length) return null
  // Small cohorts get bigger cells so the grid fills its card and reads as a
  // deliberate figure; large cohorts shrink so everything stays on screen.
  const n = subjects.length
  const cell = n <= 6 ? 38 : n <= 12 ? 30 : n <= 24 ? 24 : 18
  return (
    <Card className="p-5">
      <SectionTitle
        right={
          <div className="flex flex-wrap items-center gap-3">
            {['PASS', 'WARN', 'FAIL', 'N/A'].map((k) => {
              const v = verdictOf(k)
              return (
                <span key={k} className="flex items-center gap-1.5 font-mono text-[10px] text-[var(--color-muted)]">
                  <span className="inline-block h-2.5 w-2.5 rounded-[3px]" style={{ background: v.fg }} />
                  {k}
                </span>
              )
            })}
          </div>
        }
      >
        Check matrix
      </SectionTitle>
      <p className="-mt-1 mb-3 text-xs text-[var(--color-muted)]">
        Every check against every scan. Rows are ordered by how often the check fired.
      </p>
      <div className="overflow-x-auto">
        <table className="mx-auto border-separate border-spacing-[4px]">
          <thead>
            <tr>
              <th className="sticky left-0 bg-[var(--color-surface)]" />
              {subjects.map((s) => (
                <th key={s.sid} className="h-[74px] align-bottom">
                  <Link
                    to={`/subject/${encodeURIComponent(s.sid)}`}
                    className="block whitespace-nowrap font-mono text-[10px] text-[var(--color-faint)] hover:text-[var(--color-ink)]"
                    style={{ writingMode: 'vertical-rl', transform: 'rotate(180deg)' }}
                  >
                    {s.sid}
                  </Link>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.map((row) => (
              <tr key={row.id}>
                <th className="sticky left-0 whitespace-nowrap bg-[var(--color-surface)] pr-3.5 text-right text-[13px] font-semibold">
                  {row.label}
                </th>
                {row.cells.map((c) => {
                  const v = verdictOf(c.verdict)
                  return (
                    <td key={c.sid}>
                      <Link
                        to={`/subject/${encodeURIComponent(c.sid)}`}
                        aria-label={`${c.sid}: ${row.label} ${c.verdict || 'not run'}`}
                        className="flex items-center justify-center rounded-[6px] font-bold text-white transition-transform hover:z-10 hover:scale-110"
                        style={{
                          background: c.verdict ? v.fg : 'var(--color-line)',
                          width: cell, height: cell, fontSize: Math.max(10, cell * 0.4),
                        }}
                        onMouseEnter={(e) =>
                          tip.show(e, (
                            <span>
                              <b className="font-mono">{c.sid}</b> · {row.label} ·{' '}
                              <span style={{ color: v.fg }}>{c.verdict || 'not run'}</span>
                            </span>
                          ))
                        }
                        onMouseLeave={tip.hide}
                      >
                        <span aria-hidden="true">{c.verdict ? v.glyph : ''}</span>
                      </Link>
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {tip.node}
    </Card>
  )
}

/** How often each check flagged a scan, as a share of the cohort. */
export function FlagBars({ breakdown, total }) {
  if (!breakdown?.length) {
    return (
      <p className="py-3 text-sm text-[var(--color-faint)]">
        No check flagged any scan — a clean cohort.
      </p>
    )
  }
  return (
    <div className="flex flex-col gap-3">
      {breakdown.slice(0, 8).map((f) => (
        <div key={f.label}>
          <div className="mb-1 flex items-baseline justify-between text-[13px]">
            <span>{f.label}</span>
            <span className="font-mono text-xs text-[var(--color-muted)]">
              {f.n}/{total}
            </span>
          </div>
          <div className="h-[7px] overflow-hidden rounded-full bg-[var(--color-well)]">
            <div
              className="h-full rounded-full bg-[var(--color-accent)]"
              style={{ width: `${(f.n / Math.max(total, 1)) * 100}%` }}
            />
          </div>
        </div>
      ))}
    </div>
  )
}

/** A compact 0..1 gauge for a single scan's QEI. */
export function QeiGauge({ value, cutoff, verdict }) {
  const v = verdictOf(verdict)
  const r = 46
  const cx = 60
  const cy = 56
  const pt = (t) => {
    const a = Math.PI * (1 - Math.max(0, Math.min(1, t)))
    return [cx + r * Math.cos(a), cy - r * Math.sin(a)]
  }
  const [sx, sy] = pt(0)
  const [ex, ey] = pt(1)
  const [vx, vy] = pt(value)
  const [tx, ty] = pt(cutoff)
  const ac = Math.PI * (1 - cutoff)
  return (
    <svg viewBox="0 0 120 74" className="w-[112px]" role="img"
         aria-label={`QEI ${value.toFixed(3)}, cut-off ${cutoff}`}>
      <path d={`M${sx} ${sy} A${r} ${r} 0 0 1 ${ex} ${ey}`} fill="none"
            stroke="var(--color-well)" strokeWidth="10" strokeLinecap="round" />
      <path d={`M${sx} ${sy} A${r} ${r} 0 0 1 ${vx} ${vy}`} fill="none"
            stroke={v.fg} strokeWidth="10" strokeLinecap="round" />
      <line x1={tx} y1={ty} x2={cx + (r - 9) * Math.cos(ac)} y2={cy - (r - 9) * Math.sin(ac)}
            stroke="var(--color-ink)" strokeWidth="1.6" />
    </svg>
  )
}
