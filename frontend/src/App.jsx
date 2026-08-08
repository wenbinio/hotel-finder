import { useCallback, useEffect, useRef, useState } from 'react'
import './App.css'
import DateChart from './components/DateChart'
import DateSweep from './components/DateSweep'
import HotelResults from './components/HotelResults'
import SearchPanel from './components/SearchPanel'
import SweepProgress from './components/SweepProgress'
import { useLatestRequest } from './hooks/useLatestRequest'
import { cancelSweep, createSweep, fetchJson, getSweep } from './lib/api'
import { stableHotelKey } from './lib/pricing'

const POLL_DELAY_MS = 750
const ACTIVE_SWEEP_STATUSES = new Set(['queued', 'running'])

function isAbort(error) {
  return error?.name === 'AbortError'
}

function messageFor(error) {
  return error?.message || 'An unexpected error occurred'
}

function warningText(warning) {
  if (typeof warning === 'string') return warning
  if (warning?.message) return warning.message
  if (warning?.location && warning?.code) return `${warning.location}: ${warning.code}`
  if (warning?.code) return warning.code
  try {
    return JSON.stringify(warning)
  } catch {
    return String(warning)
  }
}

function responseWarnings(response) {
  return [
    ...(response?.warnings || []),
    ...(response?.failedDestinations || []),
    ...(response?.failedHotels || []),
  ].map(warningText).filter(Boolean)
}

function normalizeSearchResult(response, query) {
  if (Array.isArray(response?.hotels)) {
    return {
      ...response,
      checkin: response.checkin || query.checkin,
      checkout: response.checkout || query.checkout,
      results: response.hotels,
    }
  }
  return response
}

function groupedHotels(result) {
  return result?.results && !Array.isArray(result.results) && typeof result.results === 'object'
    ? result.results
    : null
}

function initialTab(result) {
  const groups = groupedHotels(result)
  if (!groups) return 'all'
  if ((groups.beachfront || []).length > 0) return 'beachfront'
  return 'non_beachfront'
}

function mergeComparison(result, comparedHotels) {
  const comparedByKey = new Map(
    (comparedHotels || []).map(hotel => [stableHotelKey(hotel), hotel]),
  )
  const mergeHotel = hotel => {
    const compared = comparedByKey.get(stableHotelKey(hotel))
    return compared ? { ...hotel, providers: compared.providers || hotel.providers } : hotel
  }

  if (Array.isArray(result?.results)) {
    return { ...result, results: result.results.map(mergeHotel) }
  }
  const groups = groupedHotels(result)
  if (groups) {
    return {
      ...result,
      results: Object.fromEntries(
        Object.entries(groups).map(([category, hotels]) => [category, hotels.map(mergeHotel)]),
      ),
    }
  }
  return result
}

function abortCurrent(begin) {
  const replacement = begin()
  replacement.finish()
}

export default function App() {
  const [destinations, setDestinations] = useState({})
  const [query, setQuery] = useState(null)
  const [result, setResult] = useState(null)
  const [searching, setSearching] = useState(false)
  const [comparing, setComparing] = useState(false)
  const [error, setError] = useState(null)
  const [warnings, setWarnings] = useState([])
  const [sweepJob, setSweepJob] = useState(null)
  const [tab, setTab] = useState('beachfront')
  const [sortKey, setSortKey] = useState('price')

  const { begin: beginDestinations } = useLatestRequest()
  const { begin: beginSearch } = useLatestRequest()
  const { begin: beginComparison } = useLatestRequest()
  const { begin: beginSweep } = useLatestRequest()
  const { begin: beginCancellation } = useLatestRequest()
  const pollTimer = useRef(null)
  const sweepInput = useRef(null)
  const pollSweepRef = useRef(null)

  const clearPollTimer = useCallback(() => {
    if (pollTimer.current !== null) {
      clearTimeout(pollTimer.current)
      pollTimer.current = null
    }
  }, [])

  useEffect(() => () => clearPollTimer(), [clearPollTimer])

  useEffect(() => {
    const request = beginDestinations()

    async function loadDestinations() {
      try {
        const response = await fetchJson('/api/destinations', { signal: request.signal })
        if (request.isCurrent()) setDestinations(response)
      } catch (loadError) {
        if (request.isCurrent() && !isAbort(loadError)) {
          setError({ operation: 'Destination loading', message: messageFor(loadError) })
        }
      } finally {
        request.finish()
      }
    }

    loadDestinations()
  }, [beginDestinations])

  const applySweepResult = useCallback(sweepResult => {
    if (!sweepResult?.bestDateResults) return
    const cheapestDate = sweepResult.cheapestDate || {}
    const nextResult = {
      checkin: cheapestDate.checkin,
      checkout: cheapestDate.checkout,
      results: sweepResult.bestDateResults,
      totalBeachfront: sweepResult.totalBeachfront,
      totalNonBeachfront: sweepResult.totalNonBeachfront,
    }
    const nextQuery = {
      checkin: cheapestDate.checkin,
      checkout: cheapestDate.checkout,
      minStars: 5,
      nights: sweepInput.current?.nights || 1,
    }
    setResult(nextResult)
    setQuery(nextQuery)
    setTab(initialTab(nextResult))
    setSortKey('price')
  }, [])

  const pollSweep = useCallback(async (jobId, request) => {
    try {
      const snapshot = await getSweep(jobId, request.signal)
      if (!request.isCurrent()) return

      setSweepJob(snapshot)

      if (ACTIVE_SWEEP_STATUSES.has(snapshot.status)) {
        pollTimer.current = setTimeout(() => {
          pollTimer.current = null
          pollSweepRef.current?.(jobId, request)
        }, POLL_DELAY_MS)
        return
      }

      if (snapshot.status === 'completed') applySweepResult(snapshot.result)
      if (snapshot.status === 'failed') {
        setError({
          operation: 'Date sweep',
          message: warningText(snapshot.warnings?.[0]) || 'The sweep stopped before completing',
        })
      }
      request.finish()
    } catch (pollError) {
      if (!request.isCurrent() || isAbort(pollError)) return
      setSweepJob(previous => previous ? { ...previous, status: 'failed' } : null)
      setError({ operation: 'Date sweep', message: messageFor(pollError) })
      request.finish()
    }
  }, [applySweepResult])

  useEffect(() => {
    pollSweepRef.current = pollSweep
  }, [pollSweep])

  const stopSweep = useCallback(() => {
    clearPollTimer()
    abortCurrent(beginSweep)
    abortCurrent(beginCancellation)
    setSweepJob(null)
  }, [beginCancellation, beginSweep, clearPollTimer])

  const handleSearch = useCallback(async searchQuery => {
    stopSweep()
    abortCurrent(beginComparison)
    setComparing(false)

    const request = beginSearch()
    setSearching(true)
    setError(null)
    setWarnings([])

    try {
      const endpoint = searchQuery.location ? '/api/search' : '/api/search-all'
      const response = await fetchJson(endpoint, {
        method: 'POST',
        body: searchQuery,
        signal: request.signal,
      })
      if (!request.isCurrent()) return

      const nextResult = normalizeSearchResult(response, searchQuery)
      setQuery(searchQuery)
      setResult(nextResult)
      setWarnings(responseWarnings(response))
      setTab(initialTab(nextResult))
      setSortKey('price')
    } catch (searchError) {
      if (request.isCurrent() && !isAbort(searchError)) {
        setError({ operation: 'Hotel search', message: messageFor(searchError) })
      }
    } finally {
      if (request.isCurrent()) setSearching(false)
      request.finish()
    }
  }, [beginComparison, beginSearch, stopSweep])

  const handleCompare = useCallback(async hotels => {
    if (!query || !result || ACTIVE_SWEEP_STATUSES.has(sweepJob?.status)) return

    const request = beginComparison()
    setComparing(true)
    setError(null)

    try {
      const response = await fetchJson('/api/compare-prices', {
        method: 'POST',
        body: {
          checkin: query.checkin,
          checkout: query.checkout,
          hotels,
        },
        signal: request.signal,
      })
      if (!request.isCurrent()) return
      setResult(previous => mergeComparison(previous, response.hotels))
      setWarnings(responseWarnings(response))
    } catch (compareError) {
      if (request.isCurrent() && !isAbort(compareError)) {
        setError({ operation: 'Provider comparison', message: messageFor(compareError) })
      }
    } finally {
      if (request.isCurrent()) setComparing(false)
      request.finish()
    }
  }, [beginComparison, query, result, sweepJob?.status])

  const handleSweep = useCallback(async parameters => {
    clearPollTimer()
    abortCurrent(beginSearch)
    abortCurrent(beginComparison)
    abortCurrent(beginCancellation)
    setSearching(false)
    setComparing(false)

    const request = beginSweep()
    sweepInput.current = parameters
    setSweepJob({ jobId: null, status: 'queued' })
    setError(null)
    setWarnings([])

    try {
      const created = await createSweep(parameters, request.signal)
      if (!request.isCurrent()) return
      if (!created?.jobId) throw new Error('The server did not return a sweep job ID')

      setSweepJob(created)
      await pollSweep(created.jobId, request)
    } catch (sweepError) {
      if (!request.isCurrent() || isAbort(sweepError)) return
      setSweepJob(previous => previous ? { ...previous, status: 'failed' } : null)
      setError({ operation: 'Date sweep', message: messageFor(sweepError) })
      request.finish()
    }
  }, [beginCancellation, beginComparison, beginSearch, beginSweep, clearPollTimer, pollSweep])

  const handleCancel = useCallback(async jobId => {
    if (!jobId || !ACTIVE_SWEEP_STATUSES.has(sweepJob?.status) || sweepJob.cancelRequested) return
    const request = beginCancellation()
    setError(null)

    try {
      const cancellation = await cancelSweep(jobId, request.signal)
      if (!request.isCurrent()) return
      setSweepJob(previous => {
        if (!previous || previous.jobId !== jobId) return previous
        return {
          ...previous,
          ...cancellation,
          progress: cancellation.progress || previous.progress,
          partial: cancellation.partial || previous.partial,
          result: cancellation.result ?? previous.result,
          warnings: cancellation.warnings || previous.warnings,
        }
      })
    } catch (cancelError) {
      if (request.isCurrent() && !isAbort(cancelError)) {
        setError({ operation: 'Sweep cancellation', message: messageFor(cancelError) })
      }
    } finally {
      request.finish()
    }
  }, [beginCancellation, sweepJob])

  const sweepActive = ACTIVE_SWEEP_STATUSES.has(sweepJob?.status)
  const groups = groupedHotels(result)
  const displayedHotels = groups ? (groups[tab] || []) : (result?.results || [])
  const nights = query?.nights || 1

  return (
    <main className="app">
      <header>
        <h1>5-Star Hotel Finder</h1>
        <p className="subtitle">Live Google Hotels data | Flights from Singapore</p>
      </header>

      <fieldset className="operation-group" disabled={sweepActive} aria-label="Hotel search controls">
        <SearchPanel onSubmit={handleSearch} />
      </fieldset>
      <DateSweep destinations={destinations} onSubmit={handleSweep} loading={sweepActive} />

      {searching && <p className="operation-status" role="status">Searching for hotels…</p>}
      {comparing && <p className="operation-status" role="status">Comparing provider prices…</p>}
      {error && (
        <div className="error" role="alert">
          <strong>{error.operation} failed:</strong> {error.message}
        </div>
      )}

      <SweepProgress job={sweepJob} onCancel={handleCancel} />
      {sweepActive && sweepJob?.cancelRequested && (
        <p className="operation-status" role="status">Cancellation requested…</p>
      )}
      {sweepJob?.status === 'completed' && <DateChart data={sweepJob.result} />}

      {result && query && (
        <>
          <p className="meta">
            Stay: {query.checkin} to {query.checkout}
          </p>
          {groups && (
            <nav className="tabs" aria-label="Hotel categories">
              {[
                ['beachfront', 'Beachfront'],
                ['non_beachfront', 'Non-beachfront'],
              ].map(([category, label]) => (
                <button
                  key={category}
                  type="button"
                  className={tab === category ? 'active' : ''}
                  aria-pressed={tab === category}
                  onClick={() => setTab(category)}
                >
                  {label} ({(groups[category] || []).length})
                </button>
              ))}
            </nav>
          )}
          <HotelResults
            results={displayedHotels}
            query={query}
            warnings={warnings}
            nights={nights}
            sortKey={sortKey}
            onSortChange={setSortKey}
            onCompare={handleCompare}
            comparing={comparing || sweepActive}
          />
        </>
      )}
    </main>
  )
}
