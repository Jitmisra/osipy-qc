import { useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  Card, SectionTitle, Button, VerdictPill, EmptyState, VerdictBar, BandMeter, verdictOf,
} from '../lib/ui.jsx'
import { QeiStrip, CheckMatrix, FlagBars } from '../components/charts.jsx'

const pct = (x) => Math.round((x || 0) * 100)

/** Sortable column header. Sorting is the reviewer's main way to re-triage. */
function Th({ id, sort, setSort, align = 'left', children, className = '' }) {
  const active = sort.key === id
  return (
    <th className={`py-2 pr-3 font-semibold ${align === 'right' ? 'text-right' : 'text-left'} ${className}`}>
      <button
        type="button"
        onClick={() => setSort({ key: id, dir: active && sort.dir === 'asc' ? 'desc' : 'asc' })}
        aria-sort={active ? (sort.dir === 'asc' ? 'ascending' : 'descending') : 'none'}
        className={`inline-flex items-center gap-1 transition-colors hover:text-[var(--color-ink)] ${
          active ? 'text-[var(--color-ink)]' : ''
        }`}
      >
        {children}
        <span aria-hidden="true" className={`text-[9px] ${active ? 'opacity-100' : 'opacity-0'}`}>
          {active && sort.dir === 'desc' ? '▼' : '▲'}
        </span>
      </button>
    </th>
  )
}

export default function Overview({ cohort, onOpenThresholds }) {
  const [filter, setFilter] = useState('all')
  const [query, setQuery] = useState('')
  const [sort, setSort] = useState({ key: 'severity', dir: 'asc' })

  const rows = useMemo(() => {
    const q = query.trim().toLowerCase()
    const out = cohort.subjects.filter(
      (s) => (filter === 'all' || s.verdict === filter) && (!q || s.sid.toLowerCase().includes(q)),
    )
    const dir = sort.dir === 'asc' ? 1 : -1
    return [...out].sort((a, b) => {
      if (sort.key === 'sid') return dir * a.sid.localeCompare(b.sid)
      if (sort.key === 'qei') {
        // scans with no QEI always sort last, whichever direction is chosen
        if (a.qei == null) return 1
        if (b.qei == null) return -1
        return dir * (a.qei - b.qei)
      }
      if (sort.key === 'flag') return dir * a.flag.localeCompare(b.flag)
      return dir * (a.severity - b.severity || (a.qei ?? 2) - (b.qei ?? 2))
    })
  }, [cohort.subjects, filter, query, sort])

  if (!cohort.total) {
    return (
      <EmptyState title="No scans in this cohort">
        Point the toolbox at a folder of subject directories, each holding a CBF map, then
        reload — or{' '}
        <Link className="font-semibold text-[var(--color-accent-600)] underline" to="/upload">
          grade a single scan
        </Link>
        .
      </EmptyState>
    )
  }

  const counts = cohort.counts || {}
  const needsReview = (counts.FAIL || 0) + (counts.WARN || 0)
  const worst = cohort.flagBreakdown?.[0]

  return (
    <>
      <header className="mb-5 flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-[30px] font-bold leading-none tracking-[-0.02em]">Batch overview</h1>
          <p className="mt-2 text-sm text-[var(--color-muted)]">
            <span className="font-mono">{cohort.dataset}</span>
            <Sep />
            {cohort.total} participants
            <Sep />
            {cohort.population} profile
            <Sep />
            {cohort.strict ? 'strict' : 'lenient'} grading
          </p>
        </div>
        <div className="no-print flex gap-2">
          <Button onClick={onOpenThresholds}>Thresholds</Button>
          <Button variant="primary" onClick={() => window.print()}>Print / Save PDF</Button>
        </div>
      </header>

      {/* One row that answers "how did this batch do, and what do I look at first?" */}
      <div className="mb-4 grid items-start gap-3 lg:grid-cols-[1.5fr_1fr_1fr]">
        <Card className="p-4 shadow-[var(--shadow-e1)]">
          <div className="mb-3 flex items-baseline justify-between">
            <span className="text-[13px] text-[var(--color-muted)]">Verdict distribution</span>
            <span className="num text-[13px] text-[var(--color-faint)]">
              {needsReview} of {cohort.total} need review
            </span>
          </div>
          <VerdictBar counts={counts} total={cohort.total} onSelect={setFilter} active={filter === 'all' ? null : filter} />
          <p className="mt-3 text-[11.5px] text-[var(--color-faint)]">
            Click a segment to filter the ledger.
          </p>
        </Card>

        <Card className="p-4 shadow-[var(--shadow-e1)]">
          <div className="text-[13px] text-[var(--color-muted)]">Median QEI</div>
          <div className="mt-1 flex items-baseline gap-2">
            <span
              className="num text-[32px] font-bold leading-none"
              style={{
                color: cohort.medianQei == null ? 'var(--color-faint)'
                  : cohort.medianQei >= cohort.qeiCutoff ? 'var(--color-pass)' : 'var(--color-fail)',
              }}
            >
              {cohort.medianQei == null ? '—' : cohort.medianQei.toFixed(2)}
            </span>
            <span className="text-xs text-[var(--color-faint)]">of 1.00</span>
          </div>
          {cohort.medianQei != null && (
            <BandMeter
              value={cohort.medianQei}
              band={[cohort.qeiCutoff, 1]}
              axis={[0, 1]}
              color={cohort.medianQei >= cohort.qeiCutoff ? 'var(--color-pass)' : 'var(--color-fail)'}
            />
          )}
          <div className="mt-2 text-[11.5px] text-[var(--color-faint)]">
            {cohort.medianQei == null
              ? 'no tissue maps supplied'
              : `${cohort.belowCutoff} of ${cohort.nWithQei} below the ${cohort.qeiCutoff} cut-off`}
          </div>
        </Card>

        <Card className="p-4 shadow-[var(--shadow-e1)]">
          <div className="text-[13px] text-[var(--color-muted)]">Leading problem</div>
          <div className="mt-1 truncate text-[22px] font-bold leading-tight tracking-tight">
            {worst ? worst.label : 'None'}
          </div>
          <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-[var(--color-well)]">
            <div
              className="h-full rounded-full bg-[var(--color-accent)]"
              style={{ width: `${((worst?.n || 0) / Math.max(cohort.total, 1)) * 100}%` }}
            />
          </div>
          <div className="mt-2 text-[11.5px] text-[var(--color-faint)]">
            {worst ? `flagged ${worst.n} of ${cohort.total} scans` : 'no check flagged a scan'}
          </div>
        </Card>
      </div>

      <div className="grid items-start gap-4 xl:grid-cols-[1fr_330px]">
        <Card className="min-w-0 p-5 shadow-[var(--shadow-e1)]">
          <SectionTitle
            right={
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Find a participant…"
                aria-label="Filter participants by name"
                className="no-print w-44 rounded-full border border-[var(--color-line)] bg-[var(--color-paper)] px-3 py-1 text-xs transition-colors focus:border-[var(--color-accent)]"
              />
            }
          >
            Participant ledger
          </SectionTitle>

          <div className="no-print mb-1 flex flex-wrap items-center gap-1.5">
            {['all', 'PASS', 'WARN', 'FAIL'].map((k) => {
              const n = k === 'all' ? cohort.total : counts[k] || 0
              const active = filter === k
              const v = k === 'all' ? null : verdictOf(k)
              return (
                <button
                  key={k}
                  onClick={() => setFilter(k)}
                  aria-pressed={active}
                  disabled={n === 0 && k !== 'all'}
                  className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 font-mono text-[11px] font-semibold transition-colors disabled:opacity-40 ${
                    active
                      ? 'border-[var(--color-accent)] bg-[var(--color-accent-050)] text-[var(--color-accent-600)]'
                      : 'border-[var(--color-line)] text-[var(--color-muted)] hover:border-[var(--color-faint)] hover:text-[var(--color-ink)]'
                  }`}
                >
                  {v && <span className="inline-block h-2 w-2 rounded-full" style={{ background: v.fg }} />}
                  {k === 'all' ? 'All' : k}
                  <span className="num opacity-60">{n}</span>
                </button>
              )
            })}
          </div>

          <div className="overflow-x-auto">
            <table className="w-full border-collapse text-sm">
              <thead>
                <tr className="border-b border-[var(--color-line)] font-mono text-[10px] uppercase tracking-wider text-[var(--color-faint)]">
                  <Th id="sid" sort={sort} setSort={setSort}>Participant</Th>
                  <Th id="severity" sort={sort} setSort={setSort}>Verdict</Th>
                  <Th id="qei" sort={sort} setSort={setSort} className="w-[190px]">QEI</Th>
                  <Th id="flag" sort={sort} setSort={setSort}>Leading flag</Th>
                  <th className="py-2" />
                </tr>
              </thead>
              <tbody>
                {rows.map((s) => (
                  <tr
                    key={s.sid}
                    className="group border-b border-[var(--color-line)] transition-colors last:border-0 hover:bg-[var(--color-paper)]"
                  >
                    <td className="py-2.5 pr-3 font-mono font-semibold">{s.sid}</td>
                    <td className="py-2.5 pr-3"><VerdictPill verdict={s.verdict} /></td>
                    <td className="py-2.5 pr-3">
                      {s.qei == null ? (
                        <span className="text-[var(--color-faint)]">—</span>
                      ) : (
                        <div className="flex items-center gap-2.5">
                          <span className="num w-9 text-right font-medium">{s.qei.toFixed(2)}</span>
                          <div className="relative h-1.5 flex-1 rounded-full bg-[var(--color-well)]">
                            {/* the published cut-off, drawn once so every row is comparable */}
                            <span
                              aria-hidden="true"
                              className="absolute top-1/2 h-2.5 w-px -translate-y-1/2 bg-[var(--color-faint)]"
                              style={{ left: `${cohort.qeiCutoff * 100}%` }}
                            />
                            <div
                              className="h-full rounded-full"
                              style={{ width: `${s.qei * 100}%`, background: verdictOf(s.verdict).fg }}
                            />
                          </div>
                        </div>
                      )}
                    </td>
                    <td className={`py-2.5 pr-3 ${s.incomplete ? 'text-[var(--color-faint)]' : ''}`}>
                      {s.flag}
                    </td>
                    <td className="py-2.5 text-right">
                      <Link
                        to={`/subject/${encodeURIComponent(s.sid)}`}
                        className="inline-flex items-center gap-1 rounded-lg px-2.5 py-1.5 text-xs font-semibold text-[var(--color-accent-600)] transition-colors hover:bg-[var(--color-accent-050)]"
                      >
                        Report<span aria-hidden="true" className="transition-transform group-hover:translate-x-0.5">→</span>
                      </Link>
                    </td>
                  </tr>
                ))}
                {rows.length === 0 && (
                  <tr>
                    <td colSpan="5" className="py-10 text-center text-sm text-[var(--color-faint)]">
                      No participant matches this filter.{' '}
                      <button
                        onClick={() => { setFilter('all'); setQuery('') }}
                        className="font-semibold text-[var(--color-accent-600)] underline"
                      >
                        Clear
                      </button>
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
          {rows.length !== cohort.total && rows.length > 0 && (
            <p className="num mt-3 text-[11px] text-[var(--color-faint)]">
              Showing {rows.length} of {cohort.total}
            </p>
          )}
        </Card>

        <Card className="p-5 shadow-[var(--shadow-e1)]">
          <SectionTitle>Flagged checks</SectionTitle>
          <p className="-mt-1 mb-3 text-xs text-[var(--color-muted)]">
            How many scans each check flagged.
          </p>
          <FlagBars breakdown={cohort.flagBreakdown} total={cohort.total} />
          <div className="mt-5 rounded-xl border border-[var(--color-accent)]/20 bg-[var(--color-accent-050)] p-3.5 text-[13px] leading-relaxed">
            Graded <b>{cohort.total}</b> scans against the <b>{cohort.population}</b> profile:{' '}
            <b>{pct(cohort.rates?.PASS)}%</b> pass, {pct(cohort.rates?.WARN)}% warn,{' '}
            {pct(cohort.rates?.FAIL)}% fail.
            {worst && <> The most common flag is <b>{worst.label}</b>.</>}
          </div>
        </Card>
      </div>

      <div className="mt-4 grid items-start gap-4 xl:grid-cols-[minmax(0,1fr)_auto]">
        <QeiStrip subjects={cohort.subjects} cutoff={cohort.qeiCutoff} />
        <CheckMatrix matrix={cohort.matrix} subjects={cohort.subjects} />
      </div>
    </>
  )
}

const Sep = () => <span aria-hidden="true" className="px-2 text-[var(--color-line)]">·</span>
