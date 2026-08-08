import { useState, useCallback, useEffect, useRef } from 'react'
import './App.css'

const API = import.meta.env.DEV ? 'http://localhost:5001/api' : '/api'

const AMENITY_TIERS = {
  all: 'All 5-Star',
  pool_spa_dining: 'Pool + Spa + Dining',
  room_only: 'Room Quality',
  full_resort: 'Full Resort',
}

const TIER_KEYWORDS = {
  pool_spa_dining: { required: ['Pool', 'Spa'], any: ['Restaurant', 'Bar', 'Dining', 'Breakfast'] },
  full_resort: { required: ['Pool', 'Spa', 'Restaurant'], any: ['Beach', 'Fitness', 'Gym', 'Kid-friendly', 'Airport shuttle'] },
}

function matchesTier(hotel, tier) {
  if (tier === 'all') return true
  if (tier === 'room_only') return true
  const rules = TIER_KEYWORDS[tier]
  if (!rules) return true
  const amenStr = (hotel.amenities || []).join(' ').toLowerCase()
  const hasRequired = rules.required.every(k => amenStr.includes(k.toLowerCase()))
  const hasAny = rules.any.some(k => amenStr.includes(k.toLowerCase()))
  return hasRequired && hasAny
}

function SearchPanel({ onSearch, loading, nights, setNights, minStars, setMinStars }) {
  const [checkin, setCheckin] = useState('2026-06-15')
  const [checkout, setCheckout] = useState('2026-06-16')
  const [maxFlight, setMaxFlight] = useState(300)

  return (
    <div className="search-panel">
      <h2>Search Parameters</h2>
      <div className="fields">
        <label>
          Check-in
          <input type="date" value={checkin} onChange={e => setCheckin(e.target.value)} />
        </label>
        <label>
          Check-out
          <input type="date" value={checkout} onChange={e => setCheckout(e.target.value)} />
        </label>
        <label>
          Nights (for total)
          <input type="number" value={nights} min={1} max={30} step={1}
            onChange={e => setNights(Math.max(1, Number(e.target.value)))} />
        </label>
        <label>
          Min Stars
          <div className="star-selector">
            {[3, 4, 5].map(s => (
              <button key={s} className={`star-btn ${minStars === s ? 'active' : ''}`}
                onClick={() => setMinStars(s)}>
                {s}{'★'.repeat(s)}
              </button>
            ))}
          </div>
        </label>
        <label>
          Max Flight (USD)
          <input type="number" value={maxFlight} min={50} max={500} step={25}
            onChange={e => setMaxFlight(Number(e.target.value))} />
        </label>
        <button className="search-btn" disabled={loading}
          onClick={() => onSearch({ checkin, checkout, maxFlight, minStars })}>
          {loading ? 'Searching...' : 'Search All Destinations'}
        </button>
      </div>
    </div>
  )
}

function DateSweep({ destinations, onSweep, loading }) {
  const [mode, setMode] = useState('all')          // 'all' | 'category' | 'single'
  const [category, setCategory] = useState('beachfront')
  const [singleDest, setSingleDest] = useState('')
  const [startDate, setStartDate] = useState('2026-05-01')
  const [endDate, setEndDate] = useState('2026-10-31')
  const [nights, setNights] = useState(1)

  const allDests = destinations
    ? [...(destinations.beachfront || []), ...(destinations.non_beachfront || [])]
    : []

  const getSelectedLocations = () => {
    if (mode === 'all') return null  // null = all
    if (mode === 'category') {
      const list = destinations?.[category] || []
      return list.map(d => d.name)
    }
    return singleDest ? [singleDest] : null
  }

  return (
    <div className="search-panel date-sweep">
      <h2>Find Cheapest Dates</h2>
      <div className="sweep-modes">
        <button className={mode === 'all' ? 'active' : ''} onClick={() => setMode('all')}>
          All Destinations
        </button>
        <button className={mode === 'category' ? 'active' : ''} onClick={() => setMode('category')}>
          By Category
        </button>
        <button className={mode === 'single' ? 'active' : ''} onClick={() => setMode('single')}>
          Single Location
        </button>
      </div>
      <div className="fields">
        {mode === 'category' && (
          <label>
            Category
            <select value={category} onChange={e => setCategory(e.target.value)}>
              <option value="beachfront">Beachfront</option>
              <option value="non_beachfront">Non-Beachfront</option>
            </select>
          </label>
        )}
        {mode === 'single' && (
          <label>
            Destination
            <input type="text" value={singleDest} list="dest-list"
              onChange={e => setSingleDest(e.target.value)}
              placeholder="Type or select..." />
            <datalist id="dest-list">
              {allDests.map(d => <option key={d.name} value={d.name} />)}
            </datalist>
          </label>
        )}
        <label>
          From
          <input type="date" value={startDate} onChange={e => setStartDate(e.target.value)} />
        </label>
        <label>
          To
          <input type="date" value={endDate} onChange={e => setEndDate(e.target.value)} />
        </label>
        <label>
          Nights
          <input type="number" value={nights} min={1} max={14}
            onChange={e => setNights(Number(e.target.value))} />
        </label>
        <button className="search-btn sweep" disabled={loading}
          onClick={() => onSweep({
            locations: getSelectedLocations(),
            startDate, endDate, nights,
            mode,
          })}>
          {loading ? 'Sweeping...' : 'Find Cheapest Dates'}
        </button>
      </div>
      {mode === 'all' && (
        <p className="sweep-note">Searches cheapest 5-star across ALL destinations for each date sample. Intensive — takes ~2 min.</p>
      )}
    </div>
  )
}

function SweepProgress({ active }) {
  const [prog, setProg] = useState(null)
  const intervalRef = useRef(null)

  useEffect(() => {
    if (active) {
      intervalRef.current = setInterval(async () => {
        try {
          const res = await fetch(`${API}/sweep-progress`)
          const data = await res.json()
          setProg(data)
        } catch {}
      }, 1500)
    } else {
      clearInterval(intervalRef.current)
      setProg(null)
    }
    return () => clearInterval(intervalRef.current)
  }, [active])

  if (!active || !prog?.active) return null

  const dateProgress = prog.total_dates > 0 ? (prog.dates_done?.length || 0) / prog.total_dates : 0
  const destProgress = prog.total_dests > 0 ? (prog.dests_done || 0) / prog.total_dests : 0

  return (
    <div className="sweep-progress">
      <div className="prog-header">
        <span>Date {prog.dates_done?.length || 0} of {prog.total_dates}</span>
        <span className="prog-dest">
          {prog.current_dest && `Searching: ${prog.current_dest} (${prog.dests_done}/${prog.total_dests})`}
        </span>
      </div>
      <div className="prog-bar-outer">
        <div className="prog-bar-inner" style={{ width: `${dateProgress * 100}%` }} />
        <div className="prog-bar-sub" style={{ width: `${(dateProgress + destProgress / prog.total_dates) * 100}%` }} />
      </div>
      {prog.dates_done?.length > 0 && (
        <div className="prog-results">
          {prog.dates_done.map((d, i) => (
            <span key={i} className="prog-chip">
              {d.checkin?.slice(5)}: {d.cheapest_price ? `$${d.cheapest_price}` : '—'}
              {d.location && ` (${d.location})`}
            </span>
          ))}
        </div>
      )}
    </div>
  )
}

function DateChart({ data }) {
  if (!data?.dates?.length) return null
  const valid = data.dates.filter(d => d.cheapest_price != null)
  if (!valid.length) return null
  const maxP = Math.max(...valid.map(d => d.cheapest_price))
  const minP = Math.min(...valid.map(d => d.cheapest_price))

  const title = data.location || data.locations?.join(', ') || 'All Destinations'

  return (
    <div className="date-chart">
      <h3>Price by Date — {title}</h3>
      {data.cheapestDate && (
        <p className="best-date">
          Cheapest: <strong>${data.cheapestDate.cheapest_price}/night</strong> on{' '}
          {data.cheapestDate.checkin}
          {data.cheapestDate.cheapest_hotel && ` (${data.cheapestDate.cheapest_hotel})`}
          {data.cheapestDate.location && ` — ${data.cheapestDate.location}`}
        </p>
      )}
      <div className="bars">
        {data.dates.map((d, i) => {
          const p = d.cheapest_price
          const pct = p != null ? ((p - minP) / (maxP - minP || 1)) * 100 : 0
          const isCheapest = data.cheapestDate && d.checkin === data.cheapestDate.checkin
          return (
            <div key={i} className={`bar-col ${isCheapest ? 'cheapest' : ''}`}
              title={d.cheapest_hotel ? `${d.cheapest_hotel} (${d.location || ''}) — $${p}` : ''}>
              <span className="bar-price">{p != null ? `$${p}` : '—'}</span>
              <div className="bar" style={{ height: `${20 + pct * 0.8}%` }} />
              <span className="bar-date">{d.checkin.slice(5)}</span>
            </div>
          )
        })}
      </div>
    </div>
  )
}

function ProviderPrices({ providers }) {
  if (!providers || !Object.keys(providers).length) return null
  // Handle both {rate, url} objects and plain numbers
  const entries = Object.entries(providers).map(([name, val]) => {
    const rate = typeof val === 'object' ? val.rate : val
    const url = typeof val === 'object' ? val.url : ''
    return { name, rate, url }
  }).sort((a, b) => a.rate - b.rate)
  const cheapest = entries[0]?.rate
  return (
    <div className="providers">
      {entries.map(({ name, rate, url }) => (
        url ? (
          <a key={name} href={url} target="_blank" rel="noreferrer"
            className={`prov prov-link ${rate === cheapest ? 'prov-best' : ''}`}>
            {name}: ${rate}
          </a>
        ) : (
          <span key={name} className={`prov ${rate === cheapest ? 'prov-best' : ''}`}>
            {name}: ${rate}
          </span>
        )
      ))}
    </div>
  )
}

function HotelTable({ hotels, title, sortKey, setSortKey, nights, onCompare, comparing }) {
  if (!hotels?.length) return <p className="empty">No results yet.</p>

  const sorted = [...hotels].sort((a, b) => {
    if (sortKey === 'total') return (a.price * nights + a.flight_cost) - (b.price * nights + b.flight_cost)
    if (sortKey === 'rating') return (b.rating || 0) - (a.rating || 0)
    if (sortKey === 'best') {
      const ratesA = a.providers ? Object.values(a.providers).map(v => typeof v === 'object' ? v.rate : v) : []
      const ratesB = b.providers ? Object.values(b.providers).map(v => typeof v === 'object' ? v.rate : v) : []
      const bestA = Math.min(a.price, ...ratesA)
      const bestB = Math.min(b.price, ...ratesB)
      return bestA - bestB
    }
    return a.price - b.price
  })

  const hasProviders = sorted.some(h => h.providers && Object.keys(h.providers).length)

  return (
    <div className="table-wrap">
      <div className="table-header">
        <h3>{title} ({hotels.length} results)</h3>
        {onCompare && (
          <button className="compare-btn" disabled={comparing}
            onClick={() => onCompare(sorted.slice(0, 15))}>
            {comparing ? 'Comparing...' : 'Compare Top 15 Across Providers'}
          </button>
        )}
      </div>
      <div className="sort-bar">
        Sort:
        {['price', 'total', 'rating', ...(hasProviders ? ['best'] : [])].map(k => (
          <button key={k} className={sortKey === k ? 'active' : ''}
            onClick={() => setSortKey(k)}>
            {k === 'price' ? 'Nightly (Google)' : k === 'total' ? 'Total Trip' : k === 'rating' ? 'Rating' : 'Best Provider'}
          </button>
        ))}
      </div>
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>Hotel</th>
            <th>Location</th>
            <th>Stars</th>
            <th>Google $/Night</th>
            {hasProviders && <th>Other Providers</th>}
            <th>Flight RT</th>
            <th>{nights}-Night Total</th>
            <th>Rating</th>
            <th>Amenities</th>
          </tr>
        </thead>
        <tbody>
          {sorted.slice(0, 50).map((h, i) => {
            const provRates = h.providers
              ? Object.values(h.providers).map(v => typeof v === 'object' ? v.rate : v)
              : []
            const bestPrice = provRates.length
              ? Math.min(h.price, ...provRates)
              : h.price
            const total = bestPrice * nights + h.flight_cost
            const hasCheaper = h.providers && bestPrice < h.price
            return (
              <tr key={i}>
                <td className="rank">{i + 1}</td>
                <td className="name">
                  {h.url ? <a href={h.url} target="_blank" rel="noreferrer">{h.name}</a> : h.name}
                </td>
                <td>{h.location}</td>
                <td className="stars">{'★'.repeat(h.star_class || 5)}</td>
                <td className={`price ${hasCheaper ? 'price-beaten' : ''}`}>${h.price}</td>
                {hasProviders && (
                  <td><ProviderPrices providers={h.providers} /></td>
                )}
                <td className="flight">${h.flight_cost}</td>
                <td className="total">${total}</td>
                <td className="rating">{h.rating || '—'}</td>
                <td className="amenities">{(h.amenities || []).slice(0, 4).join(', ')}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

function App() {
  const [destinations, setDestinations] = useState(null)
  const [results, setResults] = useState(null)
  const [dateData, setDateData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [sweepLoading, setSweepLoading] = useState(false)
  const [tab, setTab] = useState('beachfront')
  const [tier, setTier] = useState('all')
  const [sortKey, setSortKey] = useState('price')
  const [nights, setNights] = useState(5)
  const [minStars, setMinStars] = useState(5)
  const [comparing, setComparing] = useState(false)
  const [error, setError] = useState(null)
  const [searchTime, setSearchTime] = useState(null)

  // Load destinations on mount
  useEffect(() => {
    fetch(`${API}/destinations`).then(r => r.json()).then(setDestinations).catch(() => {})
  }, [])

  const handleSearch = useCallback(async (params) => {
    setLoading(true)
    setError(null)
    const t0 = Date.now()
    try {
      const res = await fetch(`${API}/search-all`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
      })
      const data = await res.json()
      if (data.error) throw new Error(data.error)
      setResults(data)
      setSearchTime(((Date.now() - t0) / 1000).toFixed(1))
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [])

  const handleSweep = useCallback(async (params) => {
    setSweepLoading(true)
    setError(null)
    try {
      const res = await fetch(`${API}/cheapest-dates`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
      })
      const data = await res.json()
      if (data.error) throw new Error(data.error)
      setDateData(data)
      // If the sweep returned full hotel results for the best date, populate the main table
      if (data.bestDateResults && (data.totalBeachfront > 0 || data.totalNonBeachfront > 0)) {
        setResults({
          checkin: data.cheapestDate?.checkin || params.startDate,
          checkout: data.cheapestDate?.checkout || '',
          results: data.bestDateResults,
          totalBeachfront: data.totalBeachfront,
          totalNonBeachfront: data.totalNonBeachfront,
        })
        setSearchTime(null)
      }
    } catch (e) {
      setError(e.message)
    } finally {
      setSweepLoading(false)
    }
  }, [])

  const handleCompare = useCallback(async (hotels) => {
    setComparing(true)
    try {
      const res = await fetch(`${API}/compare-prices`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ hotels }),
      })
      const data = await res.json()
      if (data.hotels) {
        // Merge provider data back into results
        const key = tab === 'beachfront' ? 'beachfront' : 'non_beachfront'
        const provMap = {}
        data.hotels.forEach(h => { if (h.providers) provMap[h.name] = h.providers })
        const updated = results.results[key].map(h => ({
          ...h,
          providers: provMap[h.name] || h.providers,
        }))
        setResults(prev => ({
          ...prev,
          results: { ...prev.results, [key]: updated },
        }))
      }
    } catch (e) {
      setError(e.message)
    } finally {
      setComparing(false)
    }
  }, [tab, results])

  const currentHotels = results?.results?.[tab === 'beachfront' ? 'beachfront' : 'non_beachfront'] || []
  const filtered = currentHotels.filter(h => matchesTier(h, tier))

  return (
    <div className="app">
      <header>
        <h1>5-Star Hotel Finder</h1>
        <p className="subtitle">Live Google Hotels data | Flights from Singapore</p>
      </header>

      <SearchPanel onSearch={handleSearch} loading={loading} nights={nights} setNights={setNights} minStars={minStars} setMinStars={setMinStars} />
      <DateSweep destinations={destinations} onSweep={handleSweep} loading={sweepLoading} />

      {error && <div className="error">{error}</div>}

      <SweepProgress active={sweepLoading} />

      {dateData && <DateChart data={dateData} />}

      {results && (
        <>
          <div className="meta">
            Found {results.totalBeachfront} beachfront + {results.totalNonBeachfront} non-beachfront
            {' '}5-star hotels in {searchTime}s
            <span className="checkin-label">Check-in: {results.checkin}</span>
          </div>

          <div className="tabs">
            <button className={tab === 'beachfront' ? 'active' : ''} onClick={() => setTab('beachfront')}>
              Beachfront ({results.totalBeachfront})
            </button>
            <button className={tab === 'non_beachfront' ? 'active' : ''} onClick={() => setTab('non_beachfront')}>
              Non-Beachfront ({results.totalNonBeachfront})
            </button>
          </div>

          <div className="tier-filters">
            {Object.entries(AMENITY_TIERS).map(([k, v]) => (
              <button key={k} className={tier === k ? 'active' : ''} onClick={() => setTier(k)}>
                {v}
              </button>
            ))}
          </div>

          <HotelTable
            hotels={filtered}
            title={`${tab === 'beachfront' ? 'Beachfront' : 'Non-Beachfront'} — ${AMENITY_TIERS[tier]}`}
            sortKey={sortKey}
            setSortKey={setSortKey}
            nights={nights}
            onCompare={handleCompare}
            comparing={comparing}
          />
        </>
      )}
    </div>
  )
}

export default App
