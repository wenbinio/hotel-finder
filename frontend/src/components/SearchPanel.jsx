import { useState } from 'react'
import { defaultSearchDates, nightsBetween } from '../lib/dates'

function formatToday(now) {
  const year = now.getFullYear()
  const month = String(now.getMonth() + 1).padStart(2, '0')
  const day = String(now.getDate()).padStart(2, '0')
  return `${year}-${month}-${day}`
}

export default function SearchPanel({
  now = new Date(),
  value,
  onChange,
  onSubmit,
  loading = false,
}) {
  const [localValue, setLocalValue] = useState(() => ({
    ...defaultSearchDates(now), minStars: 5, maxFlight: 300,
  }))
  const [validationError, setValidationError] = useState('')
  const query = { ...localValue, ...value }
  const nights = nightsBetween(query.checkin, query.checkout)

  const update = changes => {
    const next = { ...query, ...changes }
    if (value === undefined) setLocalValue(next)
    onChange?.(next)
    setValidationError('')
  }

  const submit = event => {
    event.preventDefault()
    if (nights < 1) {
      setValidationError('Check-out must be after check-in.')
      return
    }
    onSubmit?.({ ...query, nights })
  }

  return (
    <section className="search-panel" aria-labelledby="search-panel-title">
      <h2 id="search-panel-title">Search hotels</h2>
      <form onSubmit={submit} noValidate>
        <fieldset className="fields">
          <legend>Search parameters</legend>
          <label>
            Check-in
            <input
              type="date"
              value={query.checkin}
              min={formatToday(now)}
              onChange={event => update({ checkin: event.target.value })}
            />
          </label>
          <label>
            Check-out
            <input
              type="date"
              value={query.checkout}
              min={query.checkin || formatToday(now)}
              onChange={event => update({ checkout: event.target.value })}
            />
          </label>
          <output className="nights" aria-live="polite">{nights} nights</output>
          <fieldset className="star-selector">
            <legend>Minimum stars</legend>
            {[3, 4, 5].map(stars => (
              <button
                key={stars}
                type="button"
                className={`star-btn ${query.minStars === stars ? 'active' : ''}`}
                aria-pressed={query.minStars === stars}
                onClick={() => update({ minStars: stars })}
              >
                {stars}-star
              </button>
            ))}
          </fieldset>
          <label>
            Maximum flight estimate (USD)
            <input
              type="number"
              value={query.maxFlight}
              min="0"
              step="25"
              onChange={event => update({ maxFlight: Number(event.target.value) })}
            />
          </label>
          <button className="search-btn" type="submit" disabled={loading}>
            {loading ? 'Searching…' : 'Search all destinations'}
          </button>
        </fieldset>
        {validationError && <p className="inline-error" role="alert">{validationError}</p>}
      </form>
    </section>
  )
}
