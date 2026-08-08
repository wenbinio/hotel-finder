function completedCount(progress) {
  return Number(progress?.completed ?? progress?.completedCount ?? 0)
}

function totalCount(progress) {
  return Math.max(1, Number(progress?.total ?? progress?.totalCount ?? 0))
}

function partialPrice(item) {
  return item.cheapestPrice ?? item.cheapest_price
}

function warningText(warning) {
  if (typeof warning === 'string') return warning
  if (warning === null || warning === undefined) return 'Unknown sweep warning'
  if (typeof warning !== 'object') return String(warning)
  if (typeof warning.message === 'string' && warning.message.trim()) return warning.message.trim()

  const summary = [warning.code, warning.location]
    .filter(value => typeof value === 'string' && value.trim())
    .map(value => value.trim())
  if (summary.length) return summary.join(' — ')

  try {
    return JSON.stringify(warning) || 'Unknown sweep warning'
  } catch {
    return 'Unknown sweep warning'
  }
}

export default function SweepProgress({ job, onCancel }) {
  if (!job) return null

  const progress = job.progress || {}
  const completed = completedCount(progress)
  const total = totalCount(progress)
  const partial = job.partial || []
  const warnings = job.warnings || []
  const cancellable = job.status === 'queued' || job.status === 'running'
  const jobId = job.jobId ?? job.id

  return (
    <section className="sweep-progress" aria-labelledby="sweep-progress-title">
      <div className="table-header">
        <h2 id="sweep-progress-title">Date sweep {job.status}</h2>
        {cancellable && jobId && (
          <button className="cancel-btn" type="button" onClick={() => onCancel?.(jobId)}>
            Cancel sweep
          </button>
        )}
      </div>
      <p className="progress-status" role="status" aria-live="polite">
        {completed} of {total} searches complete
        {progress.currentLocation ? `; searching ${progress.currentLocation}` : ''}.
      </p>
      <progress value={Math.min(completed, total)} max={total}>{completed} of {total}</progress>
      {partial.length > 0 && (
        <ul className="prog-results" aria-label="Completed date summaries">
          {partial.map(item => {
            const price = partialPrice(item)
            const location = item.location ? ` in ${item.location}` : ''
            return (
              <li key={`${item.checkin}-${item.location || ''}`} className="prog-chip">
                {item.checkin}: {price === undefined || price === null ? 'no available rate' : `$${price}`}{location}
              </li>
            )
          })}
        </ul>
      )}
      {warnings.length > 0 && (
        <ul className="warning-list" aria-label="Sweep warnings">
          {warnings.map((warning, index) => {
            const text = warningText(warning)
            return <li key={`${index}-${text}`}>{text}</li>
          })}
        </ul>
      )}
    </section>
  )
}
