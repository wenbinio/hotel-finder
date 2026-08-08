import { useState } from 'react'
import { defaultSweepDates } from '../lib/dates'

const MAX_NIGHTS = 30
const MAX_SAMPLE_COUNT = 10
const MAX_LOGICAL_SEARCHES = 200

function flattenDestinations(destinations) {
  if (Array.isArray(destinations)) return destinations
  return Object.values(destinations || {}).flat()
}

function destinationName(destination) {
  return typeof destination === 'string' ? destination : destination.name
}

function normalizedLocation(value) {
  return String(value || '').trim().toLocaleLowerCase()
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
  const canonicalSingleLocation = allLocations.find(
    location => normalizedLocation(location) === normalizedLocation(sweep.location),
  )
  const selectedLocations = sweep.mode === 'single'
    ? (canonicalSingleLocation ? [canonicalSingleLocation] : [])
    : sweep.mode === 'category'
      ? (destinations?.[sweep.category] || []).map(destinationName).filter(Boolean)
      : allLocations
  const validRange = sweep.startDate && sweep.endDate && sweep.endDate > sweep.startDate
  const nights = Number(sweep.nights)
  const sampleCount = Number(sweep.sampleCount)
  const validNights = Number.isInteger(nights) && nights >= 1 && nights <= MAX_NIGHTS
  const validSampleCount = Number.isInteger(sampleCount) && sampleCount >= 1 && sampleCount <= MAX_SAMPLE_COUNT
  const logicalSearches = selectedLocations.length * sampleCount
  const constraintError = !validRange
    ? 'The end date must be after the start date.'
    : sweep.mode === 'single' && !canonicalSingleLocation
      ? 'Choose a listed destination before starting a sweep.'
      : selectedLocations.length === 0
        ? 'Choose at least one destination before starting a sweep.'
        : !validNights
          ? `Nights must be between 1 and ${MAX_NIGHTS}.`
          : !validSampleCount
            ? `Samples must be between 1 and ${MAX_SAMPLE_COUNT}.`
            : logicalSearches > MAX_LOGICAL_SEARCHES
              ? `Sweep must not exceed ${MAX_LOGICAL_SEARCHES} logical hotel searches.`
              : ''
  const canSubmit = !loading && !constraintError

  const update = changes => {
    const next = { ...sweep, ...changes }
    if (value === undefined) setLocalValue(next)
    onChange?.(next)
    setValidationError('')
  }

  const submit = event => {
    event.preventDefault()
    if (constraintError) {
      setValidationError(constraintError)
      return
    }
    onSubmit?.({
      locations: selectedLocations,
      startDate: sweep.startDate,
      endDate: sweep.endDate,
      nights,
      sampleCount,
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
              <input type="number" value={sweep.nights} min="1" max={MAX_NIGHTS} onChange={event => update({ nights: Number(event.target.value) })} />
            </label>
            <label>
              Samples
              <input type="number" value={sweep.sampleCount} min="1" max={MAX_SAMPLE_COUNT} onChange={event => update({ sampleCount: Number(event.target.value) })} />
            </label>
            <button className="search-btn sweep" type="submit" disabled={!canSubmit}>
              {loading ? 'Sweeping…' : 'Find cheapest dates'}
            </button>
          </div>
        </fieldset>
        <p className="sweep-note">Six dates are sampled across the next 90 days by default.</p>
        {(validationError || constraintError) && (
          <p className="inline-error" role="alert">{validationError || constraintError}</p>
        )}
      </form>
    </section>
  )
}
