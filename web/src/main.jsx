import { StrictMode, useCallback, useEffect, useState } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'

import './index.css'
import { fetchCohort } from './lib/api.js'
import Layout from './components/Layout.jsx'
import ThresholdPanel from './components/ThresholdPanel.jsx'
import Overview from './pages/Overview.jsx'
import Subject from './pages/Subject.jsx'
import Methodology from './pages/Methodology.jsx'
import Upload from './pages/Upload.jsx'
import { OverviewSkeleton, ErrorNote, EmptyState } from './lib/ui.jsx'

function App() {
  const [cohort, setCohort] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(true)
  const [panelOpen, setPanelOpen] = useState(false)

  const load = useCallback(() => {
    setLoading(true)
    fetchCohort()
      .then((c) => { setCohort(c); setError(null) })
      // no cohort is a valid state: the server may be running in single-scan mode
      .catch((e) => setError(e))
      .finally(() => setLoading(false))
  }, [])

  useEffect(load, [load])

  const openThresholds = useCallback(() => setPanelOpen(true), [])

  return (
    <Layout cohort={cohort} onOpenThresholds={openThresholds}>
      <Routes>
        <Route
          path="/"
          element={
            loading ? <OverviewSkeleton />
              : cohort ? <Overview cohort={cohort} onOpenThresholds={openThresholds} />
              : <NoCohort error={error} />
          }
        />
        <Route path="/subject/:sid" element={<Subject cohort={cohort} />} />
        <Route path="/methodology" element={<Methodology />} />
        <Route path="/upload" element={<Upload />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>

      <ThresholdPanel
        open={panelOpen}
        onClose={() => setPanelOpen(false)}
        onApplied={(c) => setCohort(c)}
      />
    </Layout>
  )
}

/** Running without a cohort is normal (single-scan mode), not an error state. */
function NoCohort({ error }) {
  const noCohort = /no cohort/i.test(String(error?.message || ''))
  if (!noCohort && error) {
    return <ErrorNote error={error} hint="Is the osipy-qc server still running?" />
  }
  return (
    <EmptyState title="No cohort loaded">
      Start the server with <code className="font-mono text-[var(--color-accent-600)]">osipy-qc --dashboard &lt;folder&gt;</code>{' '}
      to grade a batch, or grade a single scan from the “Grade a new scan” page.
    </EmptyState>
  )
}

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </StrictMode>,
)
