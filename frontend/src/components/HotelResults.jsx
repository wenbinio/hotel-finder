import { useState } from 'react'
import { bestNightlyRate, sortHotels, stableHotelKey, tripTotal } from '../lib/pricing'

function providerEntries(providers) {
  return Object.entries(providers || {})
    .map(([name, value]) => ({
      name,
      rate: typeof value === 'object' && value !== null ? value.rate : value,
      url: typeof value === 'object' && value !== null ? value.url : undefined,
    }))
    .filter(provider => Number.isFinite(provider.rate))
    .sort((left, right) => left.rate - right.rate)
}

function ProviderRates({ providers }) {
  const entries = providerEntries(providers)
  if (!entries.length) return '—'
  const best = entries[0].rate

  return (
    <div className="providers">
      {entries.map(provider => {
        const className = `prov ${provider.rate === best ? 'prov-best' : ''}`
        const label = `${provider.name}: $${provider.rate}`
        return provider.url ? (
          <a key={provider.name} className={className} href={provider.url} target="_blank" rel="noreferrer">
            {label}
          </a>
        ) : <span key={provider.name} className={className}>{label}</span>
      })}
    </div>
  )
}

function resultList(results) {
  if (Array.isArray(results)) return results
  if (Array.isArray(results?.results)) return results.results
  if (results?.results && typeof results.results === 'object') return Object.values(results.results).flat()
  return []
}

export default function HotelResults({
  results,
  query = {},
  elapsedSeconds,
  warnings = [],
  nights = 1,
  sortKey,
  onSortChange,
  onCompare,
  comparing = false,
}) {
  const [localSortKey, setLocalSortKey] = useState('price')
  const activeSortKey = sortKey || localSortKey
  const hotels = resultList(results)
  const sorted = sortHotels(hotels, activeSortKey, nights)
  const minStars = query.minStars ?? 5
  const changeSort = nextSort => {
    if (sortKey === undefined) setLocalSortKey(nextSort)
    onSortChange?.(nextSort)
  }

  return (
    <section className="hotel-results" aria-labelledby="hotel-results-title">
      <div className="table-header">
        <div>
          <h2 id="hotel-results-title">{minStars}-star hotels and above</h2>
          {elapsedSeconds !== undefined && elapsedSeconds !== null && (
            <p className="result-timing">Results received in {elapsedSeconds} seconds.</p>
          )}
        </div>
        {onCompare && (
          <button className="compare-btn" type="button" disabled={comparing} onClick={() => onCompare(sorted.slice(0, 15))}>
            {comparing ? 'Comparing…' : 'Compare top 15 providers'}
          </button>
        )}
      </div>
      {warnings.length > 0 && (
        <ul className="warning-list" aria-live="polite" aria-label="Search warnings">
          {warnings.map(warning => <li key={warning}>{warning}</li>)}
        </ul>
      )}
      <fieldset className="sort-bar">
        <legend>Sort results</legend>
        {[
          ['price', 'Google nightly rate'],
          ['best', 'Best nightly rate'],
          ['total', 'Total trip'],
          ['rating', 'Rating'],
        ].map(([key, label]) => (
          <button
            key={key}
            type="button"
            className={activeSortKey === key ? 'active' : ''}
            aria-pressed={activeSortKey === key}
            onClick={() => changeSort(key)}
          >
            {label}
          </button>
        ))}
      </fieldset>
      {sorted.length === 0 ? <p className="empty" role="status">No hotels match this search.</p> : (
        <div className="table-wrap">
          <table>
            <caption>Hotel results</caption>
            <thead>
              <tr>
                <th scope="col">#</th>
                <th scope="col">Hotel</th>
                <th scope="col">Location</th>
                <th scope="col">Stars</th>
                <th scope="col">Google nightly rate</th>
                <th scope="col">Provider rates</th>
                <th scope="col">Best nightly rate</th>
                <th scope="col">Estimated flight RT</th>
                <th scope="col">{nights}-night total</th>
              </tr>
            </thead>
            <tbody>
              {sorted.slice(0, 50).map((hotel, index) => {
                const bestRate = bestNightlyRate(hotel)
                const total = tripTotal(hotel, nights)
                return (
                  <tr key={stableHotelKey(hotel)}>
                    <td className="rank">{index + 1}</td>
                    <td className="name">
                      {hotel.url
                        ? <a href={hotel.url} target="_blank" rel="noreferrer">{hotel.name}</a>
                        : hotel.name}
                    </td>
                    <td>{hotel.location || '—'}</td>
                    <td className="stars">{hotel.star_class ? `${hotel.star_class}-star` : '—'}</td>
                    <td className="price">{Number.isFinite(hotel.price) ? `$${hotel.price}` : '—'}</td>
                    <td><ProviderRates providers={hotel.providers} /></td>
                    <td className="price">{bestRate === null ? '—' : `$${bestRate}`}</td>
                    <td className="flight">{Number.isFinite(hotel.flight_cost) ? `$${hotel.flight_cost}` : '—'}</td>
                    <td className="total">{total === null ? '—' : `$${total}`}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </section>
  )
}
