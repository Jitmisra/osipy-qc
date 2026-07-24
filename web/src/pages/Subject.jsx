import { useEffect, useMemo, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { fetchSubject, figureUrl } from '../lib/api.js'
import {
  Card, SectionTitle, Button, VerdictPill, SubjectSkeleton, ErrorNote, Lightbox,
  BandMeter, verdictOf,
} from '../lib/ui.jsx'

const GLOSS = {
  PASS: 'This scan looks usable. Every applicable check passed.',
  WARN: 'Usable with caveats — one or more checks flagged something worth reviewing.',
  FAIL: 'At least one check found a disqualifying problem. Inspect before using this scan.',
  UNKNOWN: 'Not enough was provided to reach a verdict.',
}

function CheckCard({ check }) {
  const [open, setOpen] = useState(false)
  const v = verdictOf(check.verdict)
  const rows = Object.entries(check.metric || {})
  return (
    <Card className="overflow-hidden" id={`chk-${check.id.replace(/\./g, '-')}`}>
      <div className="flex">
        {/* a provisional verdict gets a striped rail, so an uncalibrated cut-off
            never reads as a hard, evidence-backed failure */}
        <div
          className="w-[5px] flex-none"
          style={
            check.provisional
              ? { backgroundImage: `repeating-linear-gradient(45deg, ${v.fg} 0 4px, transparent 4px 8px)` }
              : { background: v.fg }
          }
        />
        <div className="min-w-0 flex-1 p-4">
          <div className="flex flex-wrap items-center gap-2">
            <VerdictPill verdict={check.verdict} />
            <span className="font-semibold">{check.label}</span>
            <span className="font-mono text-[11px] text-[var(--color-faint)]">{check.id}</span>
            {check.provisional && (
              <span
                title="Decided by an uncalibrated cut-off, so it is provisional rather than evidence-backed."
                className="rounded border border-dashed px-1.5 py-px font-mono text-[10px] uppercase tracking-wide"
                style={{ color: 'var(--color-warn)', borderColor: 'var(--color-warn)' }}
              >
                provisional
              </span>
            )}
          </div>
          <p className="mt-1.5 text-[13.5px] text-[var(--color-muted)]">{check.reason}</p>
          {rows.length > 0 && (
            <>
              <button
                onClick={() => setOpen((o) => !o)}
                aria-expanded={open}
                className="mt-2 font-mono text-[11px] text-[var(--color-accent-600)] hover:underline"
              >
                {open ? '▾' : '▸'} measurements
              </button>
              {open && (
                <div className="mt-2 overflow-x-auto">
                  <table className="w-full border-collapse text-[12.5px]">
                    <tbody>
                      {rows.map(([k, val]) => (
                        <tr key={k} className="border-b border-[var(--color-line)] last:border-0">
                          <td className="w-2/5 py-1.5 pr-3 font-mono align-top text-[var(--color-muted)]">{k}</td>
                          <td className="py-1.5 tabular-nums">
                            {typeof val === 'number' ? Number(val.toFixed(4)) : String(val)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </>
          )}
        </div>
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
      <Card className="mb-4 overflow-hidden p-6"
            style={{ background: `linear-gradient(180deg, ${v.tint}, var(--color-surface))` }}>
        <div className="font-mono text-[11px] font-semibold uppercase tracking-[0.12em]" style={{ color: v.fg }}>
          Overall verdict
        </div>
        <h2 className="mt-1.5 text-[40px] font-bold leading-none tracking-tight" style={{ color: v.fg }}>
          {data.verdict}
        </h2>
        <p className="mt-2 max-w-2xl text-[15px]">{GLOSS[data.verdict] || ''}</p>
        <div className="mt-4 flex flex-wrap gap-2">
          {Object.entries(data.summary || {}).map(([k, n]) => (
            <span key={k} className="inline-flex items-center gap-1.5 rounded-full border border-[var(--color-line)] bg-[var(--color-surface)] px-2.5 py-1 font-mono text-[11px]">
              <span className="inline-block h-2 w-2 rounded-full" style={{ background: verdictOf(k).fg }} />
              {n} {k}
            </span>
          ))}
        </div>
        {data.drivers.length > 0 && (
          <div className="mt-3 flex flex-wrap items-center gap-2 text-[13px]">
            <span className="text-[var(--color-muted)]">Needs attention:</span>
            {data.drivers.map((d) => (
              <a
                key={d.id}
                href={`#chk-${d.id.replace(/\./g, '-')}`}
                className="inline-flex items-center gap-1.5 rounded-full border border-[var(--color-line)] bg-[var(--color-surface)] px-2.5 py-1 font-mono text-[12px] hover:border-[var(--color-faint)]"
              >
                <span className="inline-block h-2 w-2 rounded-full" style={{ background: verdictOf(d.verdict).fg }} />
                {d.label}
              </a>
            ))}
          </div>
        )}
      </Card>

      {/* One ruled panel, not five cards. A fixed five-column grid left blank
          cells whenever a check did not produce its metric, and five competing
          numerals in five card frames shouted at each other. Rows also let each
          measurement sit beside the range it was graded on. */}
      {data.kpis.length > 0 && (
        <Card className="mb-6 overflow-hidden p-0 shadow-[var(--shadow-e1)]">
          <div className="flex items-center justify-between gap-3 border-b border-[var(--color-line)] px-5 py-3">
            <h2 className="text-[13px] font-semibold uppercase tracking-[0.09em] text-[var(--color-faint)]">
              Measured against expectation
            </h2>
            <Link
              to="/methodology"
              className="font-mono text-[11px] text-[var(--color-accent-600)] hover:underline"
            >
              where these ranges come from →
            </Link>
          </div>
          <ul>
            {data.kpis.map((k) => {
              const kv = verdictOf(k.verdict)
              const text = k.format === '0.000' ? k.value.toFixed(3)
                : k.format === '0.00' ? k.value.toFixed(2)
                : k.format === '0.0' ? k.value.toFixed(1)
                : Math.round(k.value).toLocaleString()
              return (
                <li
                  key={k.key}
                  className="grid items-center gap-4 border-b border-[var(--color-line)] px-5 py-2.5 last:border-0 md:grid-cols-[150px_112px_minmax(160px,1fr)_minmax(0,210px)]"
                >
                  <a href={`#chk-${k.check.replace(/\./g, '-')}`} className="group min-w-0">
                    <span className="block truncate text-[12.5px] font-semibold group-hover:text-[var(--color-accent-600)]">
                      {k.label}
                    </span>
                    <span className="block truncate font-mono text-[10px] text-[var(--color-faint)]">
                      {k.check}
                    </span>
                  </a>
                  <div className="flex items-baseline gap-1 md:justify-end">
                    <span className="num text-[19px] font-semibold leading-none" style={{ color: kv.fg }}>
                      {text}
                    </span>
                    {k.unit && <span className="text-[10.5px] text-[var(--color-muted)]">{k.unit}</span>}
                  </div>
                  <BandMeter value={k.value} band={k.band} axis={k.axis} color={kv.fg} className="!mt-0" />
                  <div className="flex items-center gap-2 text-[11.5px] text-[var(--color-faint)]">
                    <span
                      aria-label={k.verdict}
                      title={k.verdict}
                      className="inline-block h-2 w-2 flex-none rounded-full"
                      style={{ background: kv.fg }}
                    />
                    <span className="truncate">{k.foot}</span>
                  </div>
                </li>
              )
            })}
          </ul>
        </Card>
      )}

      {/* findings before images: the checks are the diagnosis, the pictures are evidence */}
      {Object.entries(byStream).map(([stream, checks]) => (
        <section key={stream} className="mb-6">
          <SectionTitle>Checks · {stream}</SectionTitle>
          <div className="flex flex-col gap-2.5">
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
        <b>INFO</b> are excluded from the overall verdict.{' '}
        <Link to="/methodology" className="font-semibold text-[var(--color-accent-600)] underline">
          See where every threshold comes from →
        </Link>
      </Card>

      <Lightbox src={zoom?.src} alt={zoom?.alt} onClose={() => setZoom(null)} />
    </>
  )
}
