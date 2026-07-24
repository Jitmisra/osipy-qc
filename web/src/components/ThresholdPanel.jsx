import { useEffect, useState } from 'react'
import { fetchConfig, applyConfig } from '../lib/api.js'
import { Button, Spinner, ErrorNote } from '../lib/ui.jsx'

const PROVENANCE = {
  published: { color: 'var(--color-pass)', label: 'Published', hint: 'A peer-reviewed paper states this number for this purpose.' },
  implementation: { color: 'var(--color-info)', label: 'Implementation', hint: "A reference pipeline's code uses it — code, not a paper." },
  uncalibrated: { color: 'var(--color-warn)', label: 'Uncalibrated', hint: 'An engineering default with no published source. Never fails a scan on its own.' },
}

/**
 * Live threshold editor. Changing a value re-grades the whole cohort on the
 * server from the already-loaded scans, so nothing is re-read from disk.
 *
 * Each field shows its provenance, because the honest answer to "why this
 * number?" belongs at the point where someone is about to change it.
 */
export default function ThresholdPanel({ open, onClose, onApplied }) {
  const [config, setConfig] = useState(null)
  const [error, setError] = useState(null)
  const [values, setValues] = useState({})
  const [population, setPopulation] = useState('adult')
  const [strict, setStrict] = useState(true)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    if (!open) return
    setError(null)
    fetchConfig()
      .then((c) => {
        setConfig(c)
        setPopulation(c.population)
        setStrict(c.strict)
        const v = {}
        c.groups.forEach((g) => g.fields.forEach((f) => { v[f.name] = f.value }))
        setValues(v)
      })
      .catch(setError)
  }, [open])

  useEffect(() => {
    const onKey = (e) => e.key === 'Escape' && onClose()
    if (open) document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [open, onClose])

  // Picking a population resets the bands to that population's own values, so
  // what is shown (and submitted) always matches the chosen profile.
  function choosePopulation(pop) {
    setPopulation(pop)
    const preset = config?.populationValues?.[pop]
    if (preset) setValues((v) => ({ ...v, ...preset }))
  }

  async function apply(reset = false) {
    setBusy(true)
    setError(null)
    try {
      const payload = reset ? {} : { population, strict, ...values }
      const cohort = await applyConfig(payload)
      onApplied?.(cohort)
      onClose()
    } catch (e) {
      setError(e)
    } finally {
      setBusy(false)
    }
  }

  const changed = config?.groups
    ?.flatMap((g) => g.fields)
    .filter((f) => Number(values[f.name]) !== Number(f.value)).length || 0

  return (
    <>
      <div
        onClick={onClose}
        className={`no-print fixed inset-0 z-30 bg-black/25 transition-opacity ${
          open ? 'opacity-100' : 'pointer-events-none opacity-0'
        }`}
      />
      <aside
        role="dialog"
        aria-label="Grading thresholds"
        aria-hidden={!open}
        className={`no-print fixed right-0 top-0 z-40 flex h-screen w-[400px] max-w-[94vw] flex-col border-l border-[var(--color-line)] bg-[var(--color-surface)] shadow-2xl transition-transform duration-200 ${
          open ? 'translate-x-0' : 'translate-x-full'
        }`}
      >
        <header className="flex items-start justify-between border-b border-[var(--color-line)] px-5 py-4">
          <div>
            <h2 className="text-base font-bold">Grading thresholds</h2>
            <p className="mt-0.5 text-xs text-[var(--color-muted)]">
              Every cut-off that grades a CBF map. Changes re-grade the cohort immediately.
            </p>
          </div>
          <button onClick={onClose} aria-label="Close" className="-mr-1 -mt-1 p-1 text-xl leading-none text-[var(--color-faint)] hover:text-[var(--color-ink)]">
            ×
          </button>
        </header>

        <div className="flex-1 overflow-y-auto px-5 py-4">
          {error && <ErrorNote error={error} />}
          {!config && !error && <Spinner label="Loading thresholds" />}

          {config && (
            <>
              <label className="mb-1.5 block font-mono text-[11px] uppercase tracking-wider text-[var(--color-muted)]">
                Population
              </label>
              <select
                value={population}
                onChange={(e) => choosePopulation(e.target.value)}
                className="w-full rounded-[9px] border border-[var(--color-line)] bg-[var(--color-paper)] px-3 py-2 font-mono text-sm"
              >
                {config.populations.map((p) => (
                  <option key={p} value={p}>{p}</option>
                ))}
              </select>
              <p className="mt-1.5 text-[11px] text-[var(--color-faint)]">
                Selecting a population resets the CBF bands to that profile.
              </p>

              <div className="mt-4 flex flex-wrap gap-x-4 gap-y-1.5 border-y border-[var(--color-line)] py-2.5">
                {Object.entries(PROVENANCE).map(([k, p]) => (
                  <span key={k} title={p.hint} className="flex items-center gap-1.5 font-mono text-[10px] text-[var(--color-muted)]">
                    <span className="inline-block h-2 w-2 rounded-full" style={{ background: p.color }} />
                    {p.label}
                  </span>
                ))}
              </div>

              {config.groups.map((g) => (
                <section key={g.title} className="mt-5">
                  <h3 className="mb-2 flex items-center gap-2 font-mono text-[11px] font-bold uppercase tracking-[0.08em] text-[var(--color-accent-600)]">
                    {g.title}
                    <span className="h-px flex-1 bg-[var(--color-line)]" />
                  </h3>
                  <div className="grid grid-cols-2 gap-x-3 gap-y-2.5">
                    {g.fields.map((f) => {
                      const p = PROVENANCE[f.provenance] || PROVENANCE.uncalibrated
                      return (
                        <div key={f.name}>
                          <label
                            htmlFor={`th-${f.name}`}
                            title={`${p.label} — ${f.citation}`}
                            className="mb-1 flex items-center gap-1.5 font-mono text-[10px] uppercase tracking-wide text-[var(--color-muted)]"
                          >
                            <span className="inline-block h-[7px] w-[7px] flex-none rounded-full" style={{ background: p.color }} />
                            <span className="truncate">{f.label}</span>
                          </label>
                          <input
                            id={`th-${f.name}`}
                            type="number"
                            step="any"
                            value={values[f.name] ?? ''}
                            onChange={(e) =>
                              setValues((v) => ({ ...v, [f.name]: e.target.value }))
                            }
                            className="w-full rounded-[8px] border border-[var(--color-line)] bg-[var(--color-paper)] px-2.5 py-1.5 font-mono text-[13px] tabular-nums focus:border-[var(--color-accent)]"
                          />
                        </div>
                      )
                    })}
                  </div>
                </section>
              ))}

              <label className="mt-5 flex items-start gap-2.5 text-sm">
                <input
                  type="checkbox"
                  checked={strict}
                  onChange={(e) => setStrict(e.target.checked)}
                  className="mt-0.5 h-4 w-4 accent-[var(--color-accent)]"
                />
                <span>
                  <b>Strict grading</b>
                  <span className="block text-xs text-[var(--color-muted)]">
                    Allow an uncalibrated cut-off to raise a failure. Turn this off and
                    those become warnings instead.
                  </span>
                </span>
              </label>
            </>
          )}
        </div>

        <footer className="flex items-center gap-2.5 border-t border-[var(--color-line)] px-5 py-4">
          <Button variant="primary" onClick={() => apply(false)} disabled={busy || !config} className="flex-1">
            {busy ? 'Re-grading…' : `Apply${changed ? ` (${changed})` : ''}`}
          </Button>
          <Button onClick={() => apply(true)} disabled={busy || !config}>Reset</Button>
        </footer>
      </aside>
    </>
  )
}
