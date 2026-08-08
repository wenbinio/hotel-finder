import { useState } from 'react'
import { defaultSweepDates } from '../lib/dates'

function flattenDestinations(destinations) {
  if (Array.isArray(destinations)) return destinations
  return Object.values(destinations || {}).flat()
}

function destinationName(destination) {
  return typeof destination === 'string' ? destination : destination.name
}

export default function DateSweep({
  destinations,
  now = new Date(),
  value,
  onChange,
  onSubmit,
  loading = false,
}) {
  const [localValue, setLocalValue] = useState(() => ({
    ...defaultSweepDates(now), mode: 'all', category: 'beachfront', location: '',
  }))
  const [validationError, setValidationError] = useState('')
  const sweep = { ...localValue, ...value }
  const allDestinations = flattenDestinations(destinations)
  const allLocations = allDestinations.map(destinationName).filter(Boolean)
  const selectedLocations = sweep.mode === 'single'
    ? (sweep.location.trim() ? [sweep.location.trim()] : [])
    : sweep.mode === 'category'
      ? (destinations?.[sweep.category] || []).map(destinationName).filter(Boolean)
      : allLocations
  const validRange = sweep.startDate && sweep.endDate && sweep.endDate > sweep.startDate
  const canSubmit = !loading && validRange && selectedLocations.length > 0 && sweep.sampleCount >= 1

  const update = changes => {
    const next = { ...sweep, ...changes }
    if (value === undefined) setLocalValue(next)
    onChange?.(next)
    setValidationError('')
  }

  const submit = event => {
    event.preventDefault()
    if (!validRange) {
      setValidationError('The end date must be after the start date.')
      return
    }
    if (!selectedLocations.length) {
      setValidationError('Choose a location before starting a sweep.')
      return
    }
    onSubmit?.({
      locations: selectedLocations,
      startDate: sweep.startDate,
      endDate: sweep.endDate,
      nights: Number(sweep.nights),
      sampleCount: Number(sweep.sampleCount),
    })
  }

  return (
    <section className="search-panel date-sweep" aria-labelledby="date-sweep-title">
      <h2 id="date-sweep-title">Find cheapest dates</h2>
      <form onSubmit={submit} noValidate>
        <fieldset>
          <legend>Sweep scope</legend>
          <div className="sweep-modes">
            {[
              ['all', 'All destinations'],
              ['category', 'By category'],
              ['single', 'Single location'],
            ].map(([mode, label]) => (
              <button
                key={mode}
                type="button"
                className={sweep.mode === mode ? 'active' : ''}
                aria-pressed={sweep.mode === mode}
                onClick={() => update({ mode })}
              >
                {label}
              </button>
            ))}
          </div>
          <div className="fields">
            {sweep.mode === 'category' && (
              <label>
                Category
                <select value={sweep.category} onChange={event => update({ category: event.target.value })}>
                  {Object.keys(destinations || {}).map(category => (
                    <option key={category} value={category}>{category.replace('_', ' ')}</option>
                  ))}
                </select>
              </label>
            )}
            {sweep.mode === 'single' && (
              <label>
                Destination
                <input
                  type="text"
                  value={sweep.location}
                  list="destination-options"
                  onChange={event => update({ location: event.target.value })}
                />
                <datalist id="destination-options">
                  {allLocations.map(location => <option key={location} value={location} />)}
                </datalist>
              </label>
            )}
            <label>
              From
              <input type="date" value={sweep.startDate} min={defaultSweepDates(now).startDate} onChange={event => update({ startDate: event.target.value })} />
            </label>
            <label>
              To
              <input type="date" value={sweep.endDate} min={sweep.startDate} onChange={event => update({ endDate: event.target.value })} />
            </label>
            <label>
              Nights
              <input type="number" value={sweep.nights} min="1" max="30" onChange={event => update({ nights: Number(event.target.value) })} />
            </label>
            <label>
              Samples
              <input type="number" value={sweep.sampleCount} min="1" max="200" onChange={event => update({ sampleCount: Number(event.target.value) })} />
            </label>
            <button className="search-btn sweep" type="submit" disabled={!canSubmit}>
              {loading ? 'Sweeping…' : 'Find cheapest dates'}
            </button>
          </div>
        </fieldset>
        <p className="sweep-note">Six dates are sampled across the next 90 days by default.</p>
        {validationError && <p className="inline-error" role="alert">{validationError}</p>}
      </form>
    </section>
  )
}
