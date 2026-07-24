import { useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  Card, SectionTitle, StatTile, Button, VerdictPill, EmptyState, verdictOf,
} from '../lib/ui.jsx'
import { QeiStrip, CheckMatrix, FlagBars } from '../components/charts.jsx'

const pct = (x) => `${Math.round((x || 0) * 100)}`

export default function Overview({ cohort, onOpenThresholds }) {
  const [filter, setFilter] = useState('all')
  const [query, setQuery] = useState('')

  const rows = useMemo(() => {
    const q = query.trim().toLowerCase()
    return cohort.subjects.filter(
      (s) => (filter === 'all' || s.verdict === filter) && (!q || s.sid.toLowerCase().includes(q)),
    )
  }, [cohort.subjects, filter, query])

  if (!cohort.total) {
    return (
      <EmptyState title="No scans in this cohort">
        Point the toolbox at a folder of subject directories, each containing a CBF map, then
        reload. You can also <Link className="text-[var(--color-accent-600)] underline" to="/upload">grade a single scan</Link>.
      </EmptyState>
    )
  }

  const counts = cohort.counts || {}
  const tiles = [
    { label: 'Scans analysed', value: cohort.total, caption: `${cohort.population} profile` },
    {
      label: 'Median QEI',
      value: cohort.medianQei == null ? '—' : cohort.medianQei.toFixed(2),
      caption: cohort.medianQei == null
        ? 'no tissue maps supplied'
        : `${cohort.belowCutoff}/${cohort.nWithQei} below the ${cohort.qeiCutoff} cut-off`,
      tone: cohort.medianQei == null ? undefined
        : cohort.medianQei >= cohort.qeiCutoff ? 'var(--color-pass)' : 'var(--color-fail)',
      bar: cohort.medianQei == null ? null
        : { value: cohort.medianQei, color: cohort.medianQei >= cohort.qeiCutoff ? 'var(--color-pass)' : 'var(--color-fail)' },
    },
    {
      label: 'Pass rate', value: pct(cohort.rates?.PASS), unit: '%',
      caption: 'usable without caveats', tone: 'var(--color-pass)',
      bar: { value: cohort.rates?.PASS || 0, color: 'var(--color-pass)' },
    },
    {
      label: 'Warnings', value: pct(cohort.rates?.WARN), unit: '%',
      caption: 'review recommended', tone: 'var(--color-warn)',
      bar: { value: cohort.rates?.WARN || 0, color: 'var(--color-warn)' },
    },
    {
      label: 'Fail rate', value: pct(cohort.rates?.FAIL), unit: '%',
      caption: 'a disqualifying problem', tone: 'var(--color-fail)',
      bar: { value: cohort.rates?.FAIL || 0, color: 'var(--color-fail)' },
    },
  ]

  return (
    <>
      <div className="mb-5 flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-[28px] font-bold tracking-tight">Batch overview</h1>
          <p className="mt-1 text-sm text-[var(--color-muted)]">
            <span className="font-mono">{cohort.dataset}</span> · {cohort.total} participants ·{' '}
            {cohort.population} profile
          </p>
        </div>
        <div className="no-print flex gap-2">
          <Button onClick={onOpenThresholds}>Thresholds</Button>
          <Button variant="primary" onClick={() => window.print()}>Print / Save PDF</Button>
        </div>
      </div>

      <div className="mb-4 grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-5">
        {tiles.map((t) => <StatTile key={t.label} {...t} />)}
      </div>

      <div className="grid gap-4 xl:grid-cols-[1fr_340px]">
        <Card className="min-w-0 p-5">
          <SectionTitle
            right={<span className="font-mono text-[11px] text-[var(--color-faint)]">worst first</span>}
          >
            Participant ledger
          </SectionTitle>

          <div className="no-print mb-3 flex flex-wrap items-center gap-2">
            {['all', 'PASS', 'WARN', 'FAIL'].map((k) => {
              const n = k === 'all' ? cohort.total : counts[k] || 0
              const active = filter === k
              const v = k === 'all' ? null : verdictOf(k)
              return (
                <button
                  key={k}
                  onClick={() => setFilter(k)}
                  aria-pressed={active}
                  className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1 font-mono text-[11px] font-semibold transition-colors ${
                    active
                      ? 'border-[var(--color-accent)] bg-[var(--color-accent-050)] text-[var(--color-accent-600)]'
                      : 'border-[var(--color-line)] text-[var(--color-muted)] hover:text-[var(--color-ink)]'
                  }`}
                >
                  {v && <span className="inline-block h-2 w-2 rounded-full" style={{ background: v.fg }} />}
                  {k === 'all' ? 'All' : k}
                  <span className="opacity-60">{n}</span>
                </button>
              )
            })}
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Find a participant…"
              aria-label="Filter participants by name"
              className="ml-auto w-44 rounded-full border border-[var(--color-line)] bg-[var(--color-paper)] px-3 py-1 text-xs"
            />
          </div>

          <div className="overflow-x-auto">
            <table className="w-full border-collapse text-sm">
              <thead>
                <tr className="border-b border-[var(--color-line)] text-left font-mono text-[10px] uppercase tracking-wider text-[var(--color-faint)]">
                  <th className="py-2 pr-3 font-semibold">Participant</th>
                  <th className="py-2 pr-3 font-semibold">Verdict</th>
                  <th className="py-2 pr-3 font-semibold">QEI</th>
                  <th className="py-2 pr-3 font-semibold">Leading flag</th>
                  <th className="py-2 font-semibold" />
                </tr>
              </thead>
              <tbody>
                {rows.map((s) => (
                  <tr key={s.sid} className="border-b border-[var(--color-line)] last:border-0 hover:bg-[var(--color-paper)]">
                    <td className="py-2.5 pr-3 font-mono font-semibold">{s.sid}</td>
                    <td className="py-2.5 pr-3"><VerdictPill verdict={s.verdict} /></td>
                    <td className="py-2.5 pr-3">
                      {s.qei == null ? (
                        <span className="text-[var(--color-faint)]">—</span>
                      ) : (
                        <span className="flex items-center gap-2">
                          <span className="h-1.5 w-12 overflow-hidden rounded-full bg-[var(--color-well)]">
                            <span className="block h-full rounded-full"
                                  style={{ width: `${s.qei * 100}%`, background: verdictOf(s.verdict).fg }} />
                          </span>
                          <span className="tabular-nums">{s.qei.toFixed(2)}</span>
                        </span>
                      )}
                    </td>
                    <td className={`py-2.5 pr-3 ${s.incomplete ? 'text-[var(--color-faint)]' : ''}`}>
                      {s.flag}
                    </td>
                    <td className="py-2.5 text-right">
                      <Link
                        to={`/subject/${encodeURIComponent(s.sid)}`}
                        className="inline-block rounded-lg bg-[var(--color-accent)] px-3 py-1.5 text-xs font-semibold text-white transition-colors hover:bg-[var(--color-accent-600)]"
                      >
                        View report →
                      </Link>
                    </td>
                  </tr>
                ))}
                {rows.length === 0 && (
                  <tr>
                    <td colSpan="5" className="py-8 text-center text-sm text-[var(--color-faint)]">
                      No participant matches this filter.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
          {rows.length !== cohort.total && (
            <p className="mt-3 font-mono text-[11px] text-[var(--color-faint)]">
              Showing {rows.length} of {cohort.total}
            </p>
          )}
        </Card>

        <Card className="p-5">
          <SectionTitle>Flagged checks</SectionTitle>
          <p className="-mt-1 mb-3 text-xs text-[var(--color-muted)]">
            How many scans each check flagged.
          </p>
          <FlagBars breakdown={cohort.flagBreakdown} total={cohort.total} />
          <div className="mt-5 rounded-xl border border-[var(--color-accent)]/20 bg-[var(--color-accent-050)] p-3.5 text-[13px] leading-relaxed">
            Graded <b>{cohort.total}</b> scans against the <b>{cohort.population}</b> profile:{' '}
            <b>{pct(cohort.rates?.PASS)}%</b> pass, {pct(cohort.rates?.WARN)}% warn,{' '}
            {pct(cohort.rates?.FAIL)}% fail.
            {cohort.flagBreakdown?.[0] && (
              <> The most common flag is <b>{cohort.flagBreakdown[0].label}</b>.</>
            )}
          </div>
        </Card>
      </div>

      <div className="mt-4 grid items-start gap-4 xl:grid-cols-[minmax(0,420px)_1fr]">
        <QeiStrip subjects={cohort.subjects} cutoff={cohort.qeiCutoff} />
        <CheckMatrix matrix={cohort.matrix} subjects={cohort.subjects} />
      </div>
    </>
  )
}
