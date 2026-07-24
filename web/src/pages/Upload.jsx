import { useRef, useState } from 'react'
import { uploadScan } from '../lib/api.js'
import { Card, SectionTitle, Button, ErrorNote } from '../lib/ui.jsx'

const FIELDS = [
  { name: 'cbf', label: 'CBF map', required: true, hint: 'The quantified perfusion map (.nii or .nii.gz).' },
  { name: 'gm', label: 'Grey-matter map', hint: 'Probability map in the same space. Needed for QEI.' },
  { name: 'wm', label: 'White-matter map', hint: 'Probability map. Needed for the GM/WM ratio.' },
  { name: 'csf', label: 'CSF map', hint: 'Optional; improves the QEI dispersion term.' },
]

const CLI = `# grade a whole cohort and open this dashboard
osipy-qc --dashboard ./cohort

# one scan, written out as a single self-contained HTML report
osipy-qc ./scan --html report.html

# try it on synthetic data, no data of your own needed
osipy-qc --dashboard-demo`

const PY = `from osipy_qc import grade_cbf

report = grade_cbf("cbf.nii.gz", gm="gm.nii.gz", wm="wm.nii.gz")
print(report.overall.value)   # 'PASS' | 'WARN' | 'FAIL'
report.to_dict()              # the full per-check result`

function Snippet({ title, code }) {
  const [copied, setCopied] = useState(false)
  return (
    <div>
      <div className="mb-1.5 flex items-center justify-between">
        <span className="font-mono text-[10px] font-bold uppercase tracking-[0.08em] text-[var(--color-accent-600)]">
          {title}
        </span>
        <button
          onClick={() => {
            navigator.clipboard?.writeText(code)
            setCopied(true)
            setTimeout(() => setCopied(false), 1400)
          }}
          className="font-mono text-[10px] text-[var(--color-muted)] hover:text-[var(--color-ink)]"
        >
          {copied ? 'copied' : 'copy'}
        </button>
      </div>
      <pre className="overflow-x-auto rounded-[10px] bg-[#161311] p-3.5 font-mono text-[12.5px] leading-relaxed text-[#ede6de]">
        {code}
      </pre>
    </div>
  )
}

export default function Upload() {
  const [files, setFiles] = useState({})
  const [population, setPopulation] = useState('adult')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const formRef = useRef(null)

  async function submit(e) {
    e.preventDefault()
    setError(null)
    if (!files.cbf) {
      setError(new Error('A CBF map is required.'))
      return
    }
    setBusy(true)
    try {
      const fd = new FormData()
      FIELDS.forEach((f) => files[f.name] && fd.append(f.name, files[f.name]))
      fd.append('population', population)
      const html = await uploadScan(fd)
      // the server returns a complete, self-contained report page
      const w = window.open('', '_blank')
      if (w) {
        w.document.write(html)
        w.document.close()
      } else {
        const url = URL.createObjectURL(new Blob([html], { type: 'text/html' }))
        window.location.href = url
      }
    } catch (err) {
      setError(err)
    } finally {
      setBusy(false)
    }
  }

  return (
    <>
      <div className="mb-5">
        <h1 className="text-[28px] font-bold tracking-tight">Grade a new scan</h1>
        <p className="mt-1 max-w-2xl text-sm text-[var(--color-muted)]">
          Upload one CBF map to get a full quality report. Everything runs locally on your own
          machine — nothing is sent anywhere.
        </p>
      </div>

      <div className="grid gap-4 lg:grid-cols-[1fr_420px]">
        <Card className="p-5">
          <SectionTitle>Upload</SectionTitle>
          <form ref={formRef} onSubmit={submit} className="flex flex-col gap-3">
            {FIELDS.map((f) => (
              <label key={f.name} className="block">
                <span className="mb-1 flex items-baseline gap-2">
                  <span className="text-[13px] font-semibold">{f.label}</span>
                  {f.required
                    ? <span className="font-mono text-[10px] text-[var(--color-accent-600)]">required</span>
                    : <span className="font-mono text-[10px] text-[var(--color-faint)]">optional</span>}
                </span>
                <input
                  type="file"
                  accept=".nii,.nii.gz,.gz"
                  onChange={(e) => setFiles((s) => ({ ...s, [f.name]: e.target.files?.[0] }))}
                  className="block w-full cursor-pointer rounded-[9px] border border-dashed border-[var(--color-line)] bg-[var(--color-paper)] p-2.5 text-[13px] file:mr-3 file:rounded-md file:border-0 file:bg-[var(--color-well)] file:px-2.5 file:py-1 file:text-[12px] file:font-semibold hover:border-[var(--color-faint)]"
                />
                <span className="mt-1 block text-[11.5px] text-[var(--color-faint)]">{f.hint}</span>
              </label>
            ))}

            <label className="block">
              <span className="mb-1 block text-[13px] font-semibold">Population</span>
              <select
                value={population}
                onChange={(e) => setPopulation(e.target.value)}
                className="w-full rounded-[9px] border border-[var(--color-line)] bg-[var(--color-paper)] px-3 py-2 font-mono text-sm"
              >
                <option value="adult">adult</option>
                <option value="neonate">neonate</option>
              </select>
              <span className="mt-1 block text-[11.5px] text-[var(--color-faint)]">
                A newborn's normal CBF is far below an adult's, so the bands differ.
              </span>
            </label>

            {error && <ErrorNote error={error} />}

            <Button variant="primary" type="submit" disabled={busy} className="mt-1 w-full py-2.5">
              {busy ? 'Grading…' : 'Grade this scan'}
            </Button>
            <p className="text-center text-[11.5px] text-[var(--color-faint)]">
              The report opens in a new tab and is a single self-contained file you can save or email.
            </p>
          </form>
        </Card>

        <Card className="p-5">
          <SectionTitle>Or run it from your own code</SectionTitle>
          <p className="-mt-1 mb-3 text-xs text-[var(--color-muted)]">
            The library is pure NumPy and nibabel: no framework, no build step.
          </p>
          <div className="flex flex-col gap-4">
            <Snippet title="Command line" code={CLI} />
            <Snippet title="Python" code={PY} />
          </div>
        </Card>
      </div>
    </>
  )
}
