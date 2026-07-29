import { useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { Card, Button, VerdictPill, EmptyState, VerdictBar, BandMeter, verdictOf } from '../lib/ui.jsx'

/** Sortable column header. Re-sorting is the reviewer's main way to re-triage. */
function Th({ id, sort, setSort, children, className = '' }) {
  const active = sort.key === id
  return (
    <th className={`py-2 font-semibold ${className}`}>
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

/** One zone of the cohort band. Every zone shares the same vertical rhythm. */
function Zone({ label, meta, children }) {
  return (
    <div className="flex min-h-[116px] flex-col justify-center bg-[var(--color-surface)] px-5 py-4">
      <div className="flex items-baseline justify-between gap-3">
        <span className="font-mono text-[10px] uppercase tracking-[0.09em] text-[var(--color-faint)]">
          {label}
        </span>
        {meta && <span className="num text-[11px] text-[var(--color-faint)]">{meta}</span>}
      </div>
      {children}
    </div>
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
        if (a.qei == null) return 1        // unscored scans always sort last
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
        Point the toolbox at a folder of subject directories, each holding a CBF map, then reload — or{' '}
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
  const wide = cohort.subjects.length > 8

  return (
    <>
      <header className="mb-4 flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-[24px] font-bold leading-none tracking-[-0.02em]">Batch overview</h1>
          <p className="mt-2 font-mono text-[13px] text-[var(--color-muted)]">{cohort.dataset}</p>
        </div>
        <div className="no-print flex gap-2">
          <Button onClick={onOpenThresholds}>Thresholds</Button>
          <Button variant="primary" onClick={() => window.print()}>Print / Save PDF</Button>
        </div>
      </header>

      <div className="grid grid-cols-12 items-start gap-4">
        {/* One band answers "how did this batch do, and what do I look at first?".
            A hairline gap divides the zones, so there is one card frame, not four. */}
        <Card className="col-span-12 overflow-hidden p-0 shadow-[var(--shadow-e1)]">
          <div className="grid gap-px bg-[var(--color-line)] sm:grid-cols-2 xl:grid-cols-[0.85fr_1.7fr_1fr_1.15fr]">
            <Zone label="Cohort" meta={`${cohort.population} profile`}>
              <div className="num mt-1 text-[32px] font-bold leading-none">{cohort.total}</div>
              <div className="mt-2 text-[11.5px] text-[var(--color-faint)]">
                {cohort.nWithQei} of {cohort.total} scored · {cohort.strict ? 'strict' : 'lenient'} grading
              </div>
            </Zone>

            <Zone label="Verdict distribution" meta={`${needsReview} need review`}>
              <div className="mt-2.5">
                <VerdictBar
                  counts={counts}
                  total={cohort.total}
                  onSelect={setFilter}
                  active={filter === 'all' ? null : filter}
                />
              </div>
              <div className="mt-2 text-[11.5px] text-[var(--color-faint)]">
                Click a segment to filter the ledger.
              </div>
            </Zone>

            <Zone label="Median QEI">
              <div className="mt-1 flex items-baseline gap-1.5">
                <span
                  className="num text-[32px] font-bold leading-none"
                  style={{
                    color: cohort.medianQei == null ? 'var(--color-faint)'
                      : cohort.medianQei >= cohort.qeiCutoff ? 'var(--color-pass)' : 'var(--color-fail)',
                  }}
                >
                  {cohort.medianQei == null ? '—' : cohort.medianQei.toFixed(2)}
                </span>
                <span className="text-[11px] text-[var(--color-faint)]">of 1.00</span>
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
            </Zone>

            <Zone label="Leading problem">
              <div className="mt-1 truncate text-[19px] font-bold leading-tight tracking-tight">
                {worst ? worst.label : 'None'}
              </div>
              <div className="mt-2.5 h-1.5 overflow-hidden rounded-full bg-[var(--color-well)]">
                <div
                  className="h-full rounded-full bg-[var(--color-none)]"
                  style={{ width: `${((worst?.n || 0) / Math.max(cohort.total, 1)) * 100}%` }}
                />
              </div>
              <div className="mt-2 text-[11.5px] text-[var(--color-faint)]">
                {worst ? `flagged ${worst.n} of ${cohort.total} scans` : 'no check flagged a scan'}
              </div>
            </Zone>
          </div>
        </Card>

        {/* The ledger is the triage surface: full-bleed, dense, and the loudest
            thing on it should be a FAIL - not a row of buttons. */}
        <Card className="col-span-12 overflow-hidden p-0 shadow-[var(--shadow-e1)]">
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[var(--color-line)] px-5 py-3">
            <h2 className="text-[13px] font-semibold uppercase tracking-[0.09em] text-[var(--color-faint)]">
              Participant ledger
            </h2>
            <div className="flex items-center gap-3">
              <span className="num font-mono text-[11px] text-[var(--color-faint)]">
                {rows.length === cohort.total
                  ? `${cohort.total} scans`
                  : `${rows.length} of ${cohort.total}`}
              </span>
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Find a participant…"
                aria-label="Filter participants by name"
                className="no-print w-44 rounded-full border border-[var(--color-line)] bg-[var(--color-paper)] px-3 py-1 text-xs transition-colors focus:border-[var(--color-accent)]"
              />
            </div>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full table-fixed border-collapse text-sm">
              <colgroup>
                <col style={{ width: '210px' }} />
                <col style={{ width: '96px' }} />
                <col style={{ width: '188px' }} />
                <col />
                <col style={{ width: '32px' }} />
              </colgroup>
              <thead>
                <tr className="border-b border-[var(--color-line)] text-left font-mono text-[10px] uppercase tracking-wider text-[var(--color-faint)]">
                  <Th id="sid" sort={sort} setSort={setSort} className="pl-5">Participant</Th>
                  <Th id="severity" sort={sort} setSort={setSort}>Verdict</Th>
                  <Th id="qei" sort={sort} setSort={setSort}>QEI</Th>
                  <Th id="flag" sort={sort} setSort={setSort}>Leading flag</Th>
                  <th className="pr-5" />
                </tr>
              </thead>
              <tbody>
                {rows.map((s) => (
                  <tr
                    key={s.sid}
                    className="group relative h-10 border-b border-[var(--color-line)] transition-colors last:border-0 hover:bg-[var(--color-paper)] focus-within:bg-[var(--color-paper)]"
                  >
                    <td className="truncate py-0 pl-5 align-middle">
                      {/* the id is the link: one accent-free target, no button row */}
                      {/* the anchor stays real for keyboard and right-click, but
                          its hit area is stretched over the entire row */}
                      <Link
                        to={`/subject/${encodeURIComponent(s.sid)}`}
                        className="font-mono font-semibold after:absolute after:inset-0 hover:text-[var(--color-accent-600)] hover:underline"
                      >
                        {s.sid}
                      </Link>
                    </td>
                    <td className="py-0 align-middle"><VerdictPill verdict={s.verdict} /></td>
                    <td className="py-0 align-middle">
                      {s.qei == null ? (
                        <span className="text-[var(--color-faint)]">—</span>
                      ) : (
                        <div className="flex items-center gap-2.5 pr-4">
                          <span className="num w-9 text-right font-medium">{s.qei.toFixed(2)}</span>
                          <div className="relative h-1.5 flex-1 rounded-full bg-[var(--color-well)]">
                            {/* the published cut-off on every row, so rows compare directly */}
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
                    <td className={`truncate py-0 align-middle ${s.incomplete ? 'text-[var(--color-faint)]' : ''}`}>
                      {s.flag}
                    </td>
                  </tr>
                ))}
                {rows.length === 0 && (
                  <tr>
                    <td colSpan="5" className="py-12 text-center text-sm text-[var(--color-faint)]">
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
        </Card>
      </div>
    </>
  )
}
