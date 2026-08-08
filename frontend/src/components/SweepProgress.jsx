function completedCount(progress) {
  return Number(progress?.completed ?? progress?.completedCount ?? 0)
}

function totalCount(progress) {
  return Math.max(1, Number(progress?.total ?? progress?.totalCount ?? 0))
}

function partialPrice(item) {
  return item.cheapestPrice ?? item.cheapest_price
}

export default function SweepProgress({ job, onCancel }) {
  if (!job) return null

  const progress = job.progress || {}
  const completed = completedCount(progress)
  const total = totalCount(progress)
  const partial = job.partial || []
  const warnings = job.warnings || []
  const cancellable = job.status === 'queued' || job.status === 'running'

  return (
    <section className="sweep-progress" aria-labelledby="sweep-progress-title">
      <div className="table-header">
        <h2 id="sweep-progress-title">Date sweep {job.status}</h2>
        {cancellable && (
          <button className="cancel-btn" type="button" onClick={() => onCancel?.(job.id)}>
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
          {warnings.map(warning => <li key={warning}>{warning}</li>)}
        </ul>
      )}
    </section>
  )
}
