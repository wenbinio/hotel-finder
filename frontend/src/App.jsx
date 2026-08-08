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
const TERMINAL_SWEEP_STATUSES = new Set(['cancelled', 'completed', 'failed'])

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

function warningRecord(warning) {
  if (typeof warning === 'string') {
    const text = warning.trim()
    return text ? { identity: JSON.stringify(['message', text.toLocaleLowerCase()]), text } : null
  }

  const code = typeof warning?.code === 'string' ? warning.code.trim() : ''
  const location = typeof warning?.location === 'string' ? warning.location.trim() : ''
  const name = typeof warning?.name === 'string' ? warning.name.trim() : ''
  const source = typeof warning?.source === 'string' ? warning.source.trim() : ''
  const message = typeof warning?.message === 'string' ? warning.message.trim() : ''
  const entity = location || name
  const qualifier = [entity, source, code].filter(Boolean).join(' — ')
  const text = message || qualifier || warningText(warning)
  if (!text) return null

  const normalized = value => value.toLocaleLowerCase()
  const identity = location
    ? JSON.stringify(['destination', normalized(code), normalized(location)])
    : name
      ? JSON.stringify(['hotel', normalized(code), normalized(name), normalized(source)])
      : code || source
        ? JSON.stringify(['structured', normalized(code), normalized(source)])
    : JSON.stringify(['message', text.toLocaleLowerCase()])
  return { identity, text, qualifier }
}

function responseWarningRecords(response) {
  const unique = new Map()
  for (const field of ['warnings', 'failedDestinations', 'failedHotels']) {
    const value = response?.[field]
    const values = value === null || value === undefined
      ? []
      : Array.isArray(value) ? value : [value]
    for (const warning of values) {
      const record = warningRecord(warning)
      if (record && !unique.has(record.identity)) unique.set(record.identity, record)
    }
  }

  return [...unique.values()]
}

function warningMessages(...groups) {
  const unique = new Map()
  for (const record of groups.flat()) {
    if (record && !unique.has(record.identity)) unique.set(record.identity, record)
  }
  const records = [...unique.values()]
  const textCounts = new Map()
  for (const record of records) {
    const normalizedText = record.text.toLocaleLowerCase()
    textCounts.set(normalizedText, (textCounts.get(normalizedText) || 0) + 1)
  }

  return records.map(record => {
    const hasDuplicateText = textCounts.get(record.text.toLocaleLowerCase()) > 1
    return hasDuplicateText && record.qualifier
      ? `${record.text} (${record.qualifier})`
      : record.text
  })
}

function hasHotels(results) {
  if (Array.isArray(results)) return results.length > 0
  return results && typeof results === 'object'
    ? Object.values(results).some(hotels => Array.isArray(hotels) && hotels.length > 0)
    : false
}

function sweepFailureMessage(snapshot) {
  const message = snapshot?.error?.message
  const code = snapshot?.error?.code
  if (message) return code ? `${message} (${code})` : message
  return warningText(snapshot?.warnings?.[0]) || 'The sweep stopped before completing'
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

function applySweepSnapshot(previous, snapshot) {
  if (!previous) return snapshot
  if (previous.jobId && snapshot.jobId && previous.jobId !== snapshot.jobId) return previous
  if (TERMINAL_SWEEP_STATUSES.has(previous.status)) return previous
  return {
    ...snapshot,
    cancelRequested: Boolean(previous.cancelRequested || snapshot.cancelRequested),
  }
}

export default function App() {
  const [destinations, setDestinations] = useState(null)
  const [destinationsStatus, setDestinationsStatus] = useState('loading')
  const [query, setQuery] = useState(null)
  const [result, setResult] = useState(null)
  const [searching, setSearching] = useState(false)
  const [comparing, setComparing] = useState(false)
  const [destinationError, setDestinationError] = useState(null)
  const [error, setError] = useState(null)
  const [searchWarnings, setSearchWarnings] = useState([])
  const [comparisonWarnings, setComparisonWarnings] = useState([])
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
  const cancellationInFlight = useRef(null)
  const cancellationObserved = useRef(null)
  const pollRetryAfterCancellation = useRef(null)

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
        if (request.isCurrent()) {
          setDestinations(response)
          setDestinationsStatus('ready')
        }
      } catch (loadError) {
        if (request.isCurrent() && !isAbort(loadError)) {
          setDestinationsStatus('failed')
          setDestinationError({ operation: 'Destination loading', message: messageFor(loadError) })
        }
      } finally {
        request.finish()
      }
    }

    loadDestinations()
  }, [beginDestinations])

  const applySweepResult = useCallback(sweepResult => {
    const cheapestDate = sweepResult?.cheapestDate
    if (!cheapestDate?.checkin || !cheapestDate?.checkout || !hasHotels(sweepResult.bestDateResults)) {
      setResult(null)
      setQuery(null)
      return
    }
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

  const scheduleSweepPoll = useCallback((jobId, request) => {
    clearPollTimer()
    pollTimer.current = setTimeout(() => {
      pollTimer.current = null
      pollSweepRef.current?.(jobId, request)
    }, POLL_DELAY_MS)
  }, [clearPollTimer])

  const resumePollAfterCancellation = useCallback(jobId => {
    const pending = pollRetryAfterCancellation.current
    if (!pending || pending.jobId !== jobId) return
    pollRetryAfterCancellation.current = null
    if (pending.request.isCurrent()) scheduleSweepPoll(jobId, pending.request)
  }, [scheduleSweepPoll])

  const finishTerminalSweep = useCallback(snapshot => {
    clearPollTimer()
    cancellationInFlight.current = null
    cancellationObserved.current = null
    pollRetryAfterCancellation.current = null
    abortCurrent(beginSweep)
    abortCurrent(beginCancellation)
    setSweepJob(previous => applySweepSnapshot(previous, snapshot))
    setError(previous => (
      previous?.operation === 'Sweep cancellation' || previous?.operation === 'Date sweep polling'
        ? null
        : previous
    ))
    if (snapshot.status === 'completed') applySweepResult(snapshot.result)
    if (snapshot.status === 'failed') {
      setError({
        operation: 'Date sweep',
        message: sweepFailureMessage(snapshot),
      })
    }
  }, [applySweepResult, beginCancellation, beginSweep, clearPollTimer])

  const pollSweep = useCallback(async (jobId, request) => {
    try {
      const snapshot = await getSweep(jobId, request.signal)
      if (!request.isCurrent()) return

      if (ACTIVE_SWEEP_STATUSES.has(snapshot.status)) {
        if (snapshot.cancelRequested) cancellationObserved.current = jobId
        setError(previous => previous?.operation === 'Date sweep polling' ? null : previous)
        setSweepJob(previous => applySweepSnapshot(previous, snapshot))
        scheduleSweepPoll(jobId, request)
        return
      }

      if (TERMINAL_SWEEP_STATUSES.has(snapshot.status)) finishTerminalSweep(snapshot)
      else request.finish()
    } catch (pollError) {
      if (!request.isCurrent() || isAbort(pollError)) return
      if (cancellationInFlight.current === jobId) {
        pollRetryAfterCancellation.current = { jobId, request }
        setError({ operation: 'Date sweep polling', message: messageFor(pollError) })
        return
      }
      if (cancellationObserved.current === jobId) {
        setError({ operation: 'Date sweep polling', message: messageFor(pollError) })
        scheduleSweepPoll(jobId, request)
        return
      }
      setSweepJob(previous => previous ? { ...previous, status: 'failed' } : null)
      setError({ operation: 'Date sweep', message: messageFor(pollError) })
      request.finish()
    }
  }, [finishTerminalSweep, scheduleSweepPoll])

  useEffect(() => {
    pollSweepRef.current = pollSweep
  }, [pollSweep])

  const stopSweep = useCallback(() => {
    clearPollTimer()
    abortCurrent(beginSweep)
    abortCurrent(beginCancellation)
    cancellationInFlight.current = null
    cancellationObserved.current = null
    pollRetryAfterCancellation.current = null
    setSweepJob(null)
  }, [beginCancellation, beginSweep, clearPollTimer])

  const handleSearch = useCallback(async searchQuery => {
    stopSweep()
    abortCurrent(beginComparison)
    setComparing(false)

    const request = beginSearch()
    setSearching(true)
    setError(null)
    setQuery(null)
    setResult(null)
    setSearchWarnings([])
    setComparisonWarnings([])

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
      setSearchWarnings(responseWarningRecords(response))
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
    setComparisonWarnings([])

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
      setComparisonWarnings(responseWarningRecords(response))
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
    cancellationInFlight.current = null
    cancellationObserved.current = null
    pollRetryAfterCancellation.current = null
    setSearching(false)
    setComparing(false)

    const request = beginSweep()
    sweepInput.current = parameters
    setSweepJob({ jobId: null, status: 'queued' })
    setError(null)
    setQuery(null)
    setResult(null)
    setSearchWarnings([])
    setComparisonWarnings([])

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
    if (
      !jobId
      || cancellationInFlight.current === jobId
      || !ACTIVE_SWEEP_STATUSES.has(sweepJob?.status)
      || sweepJob.cancelRequested
    ) return
    cancellationInFlight.current = jobId
    const request = beginCancellation()
    setError(null)
    setSweepJob(previous => {
      if (!previous || previous.jobId !== jobId || !ACTIVE_SWEEP_STATUSES.has(previous.status)) {
        return previous
      }
      return { ...previous, cancelRequested: true }
    })

    try {
      const cancellation = await cancelSweep(jobId, request.signal)
      if (!request.isCurrent()) return
      if (TERMINAL_SWEEP_STATUSES.has(cancellation.status)) {
        finishTerminalSweep(cancellation)
        return
      }
      if (cancellation.cancelRequested) cancellationObserved.current = jobId
      const preserveCancelRequested = cancellationObserved.current === jobId
      setSweepJob(previous => {
        if (!previous || previous.jobId !== jobId) return previous
        if (TERMINAL_SWEEP_STATUSES.has(previous.status)) return previous
        return {
          ...previous,
          cancelRequested: preserveCancelRequested
            || (cancellation.cancelRequested ?? previous.cancelRequested),
        }
      })
      resumePollAfterCancellation(jobId)
    } catch (cancelError) {
      if (request.isCurrent() && !isAbort(cancelError)) {
        cancellationInFlight.current = null
        const preserveCancelRequested = cancellationObserved.current === jobId
        setSweepJob(previous => {
          if (!previous || previous.jobId !== jobId || !ACTIVE_SWEEP_STATUSES.has(previous.status)) {
            return previous
          }
          return { ...previous, cancelRequested: preserveCancelRequested }
        })
        setError({ operation: 'Sweep cancellation', message: messageFor(cancelError) })
        resumePollAfterCancellation(jobId)
      }
    } finally {
      if (request.isCurrent()) cancellationInFlight.current = null
      request.finish()
    }
  }, [beginCancellation, finishTerminalSweep, resumePollAfterCancellation, sweepJob])

  const sweepActive = ACTIVE_SWEEP_STATUSES.has(sweepJob?.status)
  const groups = groupedHotels(result)
  const displayedHotels = groups ? (groups[tab] || []) : (result?.results || [])
  const nights = query?.nights || 1
  const warnings = warningMessages(searchWarnings, comparisonWarnings)
  const emptySweepResult = sweepJob?.status === 'completed'
    && (!sweepJob.result?.cheapestDate || !hasHotels(sweepJob.result?.bestDateResults))
  const emptySweepStart = sweepJob?.result?.startDate || sweepInput.current?.startDate
  const emptySweepEnd = sweepJob?.result?.endDate || sweepInput.current?.endDate

  return (
    <main className="app">
      <header>
        <h1>5-Star Hotel Finder</h1>
        <p className="subtitle">Live Google Hotels data | Flights from Singapore</p>
      </header>

      <fieldset className="operation-group" disabled={sweepActive} aria-label="Hotel search controls">
        <SearchPanel
          destinations={destinations || {}}
          singleDestinationAvailable={destinationsStatus === 'ready'}
          onSubmit={handleSearch}
        />
      </fieldset>
      {destinationsStatus === 'loading' && (
        <p className="operation-status" role="status">Loading destinations…</p>
      )}
      {destinationsStatus === 'ready' && (
        <DateSweep destinations={destinations} onSubmit={handleSweep} loading={sweepActive} />
      )}

      {searching && <p className="operation-status" role="status">Searching for hotels…</p>}
      {comparing && <p className="operation-status" role="status">Comparing provider prices…</p>}
      {destinationError && (
        <div className="error" role="alert">
          <strong>{destinationError.operation} failed:</strong> {destinationError.message}{' '}
          Single-destination search and date sweeps are unavailable until destinations load.
        </div>
      )}
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
      {emptySweepResult && (
        <p className="empty" role="status">
          {emptySweepStart && emptySweepEnd
            ? `No priced hotel results were found from ${emptySweepStart} to ${emptySweepEnd}.`
            : 'No priced hotel results were found for the requested date range.'}
        </p>
      )}

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
