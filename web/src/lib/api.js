// Thin client for the Python JSON API. Every payload here is produced by
// osipy_qc/api.py, which in turn only reshapes QCReport.to_dict() — the browser
// never recomputes any quality metric.

async function get(path) {
  const res = await fetch(path, { headers: { Accept: 'application/json' } })
  if (!res.ok) {
    let detail = res.statusText
    try {
      detail = (await res.json()).error || detail
    } catch {
      /* non-JSON error body; keep the status text */
    }
    throw new Error(detail)
  }
  return res.json()
}

export const fetchCohort = () => get('/api/cohort')
export const fetchSubject = (sid) => get(`/api/subject/${encodeURIComponent(sid)}`)
export const fetchConfig = () => get('/api/config')
export const fetchProvenance = () => get('/api/provenance')
export const fetchChecks = () => get('/api/checks')

/** Apply threshold overrides and re-grade the whole cohort server-side. */
export async function applyConfig(overrides) {
  const res = await fetch('/api/config', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(overrides),
  })
  if (!res.ok) {
    let detail = res.statusText
    try {
      detail = (await res.json()).error || detail
    } catch {
      /* keep status text */
    }
    throw new Error(detail)
  }
  return res.json()
}

export const figureUrl = (sid, name) =>
  `/api/subject/${encodeURIComponent(sid)}/figure/${name}`

/** Upload one scan for grading; resolves to the report HTML. */
/** Grade an uploaded scan. Returns the same payload shape as fetchSubject. */
export async function uploadScan(formData) {
  const res = await fetch('/run', { method: 'POST', body: formData })
  const data = await res.json().catch(() => null)
  if (!res.ok) throw new Error(data?.error || 'The server could not grade that upload.')
  return data
}
