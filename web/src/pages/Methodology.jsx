import { useEffect, useMemo, useState } from 'react'
import { fetchProvenance, fetchChecks } from '../lib/api.js'
import { Card, SectionTitle, Spinner, ErrorNote } from '../lib/ui.jsx'

const LEVEL = {
  published: {
    label: 'Published', color: 'var(--color-pass)', tint: 'var(--color-pass-tint)',
    blurb: 'A peer-reviewed paper states this number for this purpose.',
  },
  implementation: {
    label: 'Implementation', color: 'var(--color-info)', tint: 'var(--color-info-tint)',
    blurb: "A reference pipeline's code uses it. Code, not a paper — weaker, because a code choice is not always explained or reviewed.",
  },
  uncalibrated: {
    label: 'Uncalibrated', color: 'var(--color-warn)', tint: 'var(--color-warn-tint)',
    blurb: 'Our engineering default, with no published source. These never fail a scan on their own.',
  },
}

export default function Methodology() {
  const [prov, setProv] = useState(null)
  const [checks, setChecks] = useState(null)
  const [error, setError] = useState(null)
  const [level, setLevel] = useState('all')

  useEffect(() => {
    Promise.all([fetchProvenance(), fetchChecks()])
      .then(([p, c]) => { setProv(p); setChecks(c) })
      .catch(setError)
  }, [])

  const rows = useMemo(
    () => (prov?.rows || []).filter((r) => level === 'all' || r.level === level),
    [prov, level],
  )

  if (error) return <ErrorNote error={error} />
  if (!prov || !checks) return <Spinner label="Loading methodology" />

  const byStream = checks.reduce((acc, c) => {
    ;(acc[c.streamTitle] ||= []).push(c)
    return acc
  }, {})

  return (
    <>
      <div className="mb-5">
        <h1 className="text-[28px] font-bold tracking-tight">Methodology</h1>
        <p className="mt-1 max-w-3xl text-sm text-[var(--color-muted)]">
          Every threshold this toolbox grades against, and where the number came from. Most of
          this field has no published pass/fail cut-offs, so anything we could not cite is
          labelled plainly rather than dressed up.
        </p>
      </div>

      <div className="mb-4 grid grid-cols-1 gap-3 sm:grid-cols-3">
        {Object.entries(LEVEL).map(([k, l]) => (
          <Card key={k} className="p-4" style={{ background: l.tint }}>
            <div className="flex items-baseline justify-between">
              <span className="font-semibold" style={{ color: l.color }}>{l.label}</span>
              <span className="text-[26px] font-bold leading-none tabular-nums" style={{ color: l.color }}>
                {prov.counts[k] || 0}
              </span>
            </div>
            <p className="mt-1.5 text-[12.5px] leading-snug text-[var(--color-ink)]/80">{l.blurb}</p>
          </Card>
        ))}
      </div>

      <Card className="mb-6 p-5">
        <SectionTitle
          right={
            <div className="flex gap-1.5">
              {['all', 'published', 'implementation', 'uncalibrated'].map((k) => (
                <button
                  key={k}
                  onClick={() => setLevel(k)}
                  aria-pressed={level === k}
                  className={`rounded-full border px-2.5 py-1 font-mono text-[10px] font-semibold transition-colors ${
                    level === k
                      ? 'border-[var(--color-accent)] bg-[var(--color-accent-050)] text-[var(--color-accent-600)]'
                      : 'border-[var(--color-line)] text-[var(--color-muted)] hover:text-[var(--color-ink)]'
                  }`}
                >
                  {k === 'all' ? 'All' : LEVEL[k].label}
                </button>
              ))}
            </div>
          }
        >
          Threshold provenance · {prov.total} values
        </SectionTitle>
        <div className="overflow-x-auto">
          <table className="w-full border-collapse text-[13px]">
            <thead>
              <tr className="border-b border-[var(--color-line)] text-left font-mono text-[10px] uppercase tracking-wider text-[var(--color-faint)]">
                <th className="py-2 pr-3 font-semibold">Threshold</th>
                <th className="py-2 pr-3 font-semibold">Basis</th>
                <th className="py-2 pr-3 font-semibold">Source</th>
                <th className="py-2 font-semibold">What the source says</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => {
                const l = LEVEL[r.level] || LEVEL.uncalibrated
                return (
                  <tr key={r.field} className="border-b border-[var(--color-line)] align-top last:border-0">
                    <td className="py-2.5 pr-3 font-mono font-semibold">{r.field}</td>
                    <td className="py-2.5 pr-3">
                      <span className="whitespace-nowrap rounded-full px-2 py-0.5 font-mono text-[10px] font-semibold"
                            style={{ color: l.color, background: l.tint }}>
                        {l.label}
                      </span>
                    </td>
                    <td className="max-w-[280px] py-2.5 pr-3 text-[12px] text-[var(--color-muted)]">
                      {r.citation === 'NONE' || !r.citation
                        ? <span className="italic text-[var(--color-warn)]">no published source</span>
                        : r.citation}
                    </td>
                    <td className="max-w-[420px] py-2.5 text-[12px] leading-snug text-[var(--color-muted)]">
                      {r.note}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </Card>

      <SectionTitle>What the toolbox checks · {checks.length} checks</SectionTitle>
      <div className="grid gap-4 lg:grid-cols-2">
        {Object.entries(byStream).map(([stream, list]) => (
          <Card key={stream} className="p-5">
            <h3 className="mb-1 font-semibold">{stream}</h3>
            <p className="mb-3 text-xs text-[var(--color-muted)]">
              {stream === 'CBF map'
                ? 'Is the final perfusion map good?'
                : 'Was the data acquired correctly, following the protocol?'}
            </p>
            <ul className="flex flex-col gap-2.5">
              {list.map((c) => (
                <li key={c.id} className="border-b border-[var(--color-line)] pb-2.5 last:border-0 last:pb-0">
                  <div className="flex items-baseline gap-2">
                    <span className="font-mono text-[11px] text-[var(--color-faint)]">{c.id}</span>
                    <span className="text-[13px] font-semibold">{c.label}</span>
                    {!c.required && (
                      <span className="font-mono text-[10px] text-[var(--color-faint)]">optional</span>
                    )}
                  </div>
                  {c.description && (
                    <p className="mt-0.5 text-[12px] leading-snug text-[var(--color-muted)]">{c.description}</p>
                  )}
                </li>
              ))}
            </ul>
          </Card>
        ))}
      </div>
    </>
  )
}
