import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import App from './App'
import * as api from './lib/api'

const apiMocks = vi.hoisted(() => ({
  cancelSweep: vi.fn(),
  createSweep: vi.fn(),
  fetchJson: vi.fn(),
  getSweep: vi.fn(),
}))

vi.mock('./lib/api', () => apiMocks)

const DESTINATIONS = {
  beachfront: [{ name: 'Phuket', flight_usd: 150 }],
  non_beachfront: [{ name: 'Bangkok', flight_usd: 90 }],
}

function deferred() {
  let resolve
  let reject
  const promise = new Promise((resolvePromise, rejectPromise) => {
    resolve = resolvePromise
    reject = rejectPromise
  })
  return { promise, reject, resolve }
}

function hotel(name, location, price, slug) {
  return {
    name,
    location,
    price,
    flight_cost: 40,
    star_class: 5,
    url: `https://www.google.com/travel/hotels/entity/${slug}`,
  }
}

function searchResult(query, currentHotel, category = 'beachfront') {
  const results = { beachfront: [], non_beachfront: [] }
  results[category] = [currentHotel]
  return {
    checkin: query.checkin,
    checkout: query.checkout,
    results,
    totalBeachfront: results.beachfront.length,
    totalNonBeachfront: results.non_beachfront.length,
  }
}

function setSearchDates(checkin, checkout) {
  fireEvent.change(screen.getByLabelText('Check-in'), { target: { value: checkin } })
  fireEvent.change(screen.getByLabelText('Check-out'), { target: { value: checkout } })
}

async function renderReady() {
  render(<App />)
  await waitFor(() => {
    expect(screen.getByRole('button', { name: /find cheapest dates/i })).toBeEnabled()
  })
}

beforeEach(() => {
  api.fetchJson.mockReset()
  api.createSweep.mockReset()
  api.getSweep.mockReset()
  api.cancelSweep.mockReset()
  api.fetchJson.mockImplementation(path => {
    if (path === '/api/destinations') return Promise.resolve(DESTINATIONS)
    return Promise.reject(new Error(`Unexpected API path: ${path}`))
  })
  vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('App bypassed lib/api')))
})

afterEach(() => {
  cleanup()
  vi.useRealTimers()
  vi.unstubAllGlobals()
})

describe('latest search and comparison ownership', () => {
  it('keeps only the newest search result and its query dates when responses finish out of order', async () => {
    const user = userEvent.setup()
    const searchA = deferred()
    const searchB = deferred()
    api.fetchJson.mockImplementation((path, options) => {
      if (path === '/api/destinations') return Promise.resolve(DESTINATIONS)
      if (path === '/api/search-all') {
        return options.body.checkin === '2026-09-10' ? searchA.promise : searchB.promise
      }
      return Promise.reject(new Error(`Unexpected API path: ${path}`))
    })
    await renderReady()

    setSearchDates('2026-09-10', '2026-09-15')
    await user.click(screen.getByRole('button', { name: /search all destinations/i }))
    setSearchDates('2026-10-20', '2026-10-25')
    await user.click(screen.getByRole('button', { name: /search all destinations/i }))

    await act(async () => {
      searchB.resolve(searchResult(
        { checkin: '2026-10-20', checkout: '2026-10-25' },
        hotel('Newest Hotel', 'Phuket', 180, 'newest'),
      ))
    })
    expect(await screen.findByText('Newest Hotel')).toBeInTheDocument()
    expect(screen.getByText(/2026-10-20 to 2026-10-25/)).toBeInTheDocument()

    await act(async () => {
      searchA.resolve(searchResult(
        { checkin: '2026-09-10', checkout: '2026-09-15' },
        hotel('Stale Hotel', 'Bangkok', 90, 'stale'),
      ))
    })
    expect(screen.queryByText('Stale Hotel')).not.toBeInTheDocument()
    expect(screen.queryByText(/2026-09-10 to 2026-09-15/)).not.toBeInTheDocument()
  })

  it('ignores a stale comparison after a new search and merges the current comparison by validated URL', async () => {
    const user = userEvent.setup()
    const compareA = deferred()
    const compareB = deferred()
    const firstQuery = { checkin: '2026-09-10', checkout: '2026-09-15' }
    const secondQuery = { checkin: '2026-10-20', checkout: '2026-10-25' }
    const oldHotel = hotel('Old Hotel', 'Bangkok', 100, 'old')
    const currentHotel = hotel('Current Hotel', 'Phuket', 200, 'current')

    api.fetchJson.mockImplementation((path, options) => {
      if (path === '/api/destinations') return Promise.resolve(DESTINATIONS)
      if (path === '/api/search-all') {
        const query = options.body
        return Promise.resolve(query.checkin === firstQuery.checkin
          ? searchResult(firstQuery, oldHotel)
          : searchResult(secondQuery, currentHotel))
      }
      if (path === '/api/compare-prices') {
        return options.body.hotels[0].name === 'Old Hotel' ? compareA.promise : compareB.promise
      }
      return Promise.reject(new Error(`Unexpected API path: ${path}`))
    })
    await renderReady()

    setSearchDates(firstQuery.checkin, firstQuery.checkout)
    await user.click(screen.getByRole('button', { name: /search all destinations/i }))
    expect(await screen.findByText('Old Hotel')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: /compare top 15 providers/i }))

    setSearchDates(secondQuery.checkin, secondQuery.checkout)
    await user.click(screen.getByRole('button', { name: /search all destinations/i }))
    expect(await screen.findByText('Current Hotel')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: /compare top 15 providers/i }))

    await act(async () => {
      compareB.resolve({
        hotels: [{
          ...currentHotel,
          name: 'Provider Alias',
          providers: { Agoda: { rate: 160, url: 'https://www.agoda.com/search' } },
        }],
      })
    })
    expect(await screen.findByText('Agoda: $160')).toBeInTheDocument()

    await act(async () => {
      compareA.resolve({ hotels: [{ ...oldHotel, providers: { StaleBeds: { rate: 50 } } }] })
    })
    expect(screen.queryByText(/StaleBeds/)).not.toBeInTheDocument()
    expect(screen.queryByText('Old Hotel')).not.toBeInTheDocument()

    const currentComparison = api.fetchJson.mock.calls.findLast(([path]) => path === '/api/compare-prices')
    expect(currentComparison[1].body).toEqual({
      checkin: secondQuery.checkin,
      checkout: secondQuery.checkout,
      hotels: [currentHotel],
    })
  })

  it('deduplicates warning fields by semantic identity before rendering stable keys', async () => {
    const user = userEvent.setup()
    const duplicateKeyError = vi.spyOn(console, 'error').mockImplementation(() => {})
    const warning = {
      code: 'timeout', location: 'Phuket', message: 'Phuket provider timed out',
    }
    api.fetchJson.mockImplementation((path, options) => {
      if (path === '/api/destinations') return Promise.resolve(DESTINATIONS)
      if (path === '/api/search-all') {
        return Promise.resolve({
          ...searchResult(options.body, hotel('Warning Hotel', 'Phuket', 180, 'warning')),
          warnings: [warning, { ...warning }],
          failedDestinations: [{
            code: 'timeout', location: 'Phuket', message: 'A second summary of the same timeout',
          }],
        })
      }
      return Promise.reject(new Error(`Unexpected API path: ${path}`))
    })

    try {
      await renderReady()
      await user.click(screen.getByRole('button', { name: /search all destinations/i }))

      const warningList = await screen.findByRole('list', { name: 'Search warnings' })
      expect(within(warningList).getAllByRole('listitem')).toHaveLength(1)
      expect(warningList).toHaveTextContent('Phuket provider timed out')
      expect(duplicateKeyError.mock.calls.flat().join(' ')).not.toMatch(/same key|unique key/i)
    } finally {
      duplicateKeyError.mockRestore()
    }
  })

  it('preserves distinct failed hotels while deduplicating a mirrored warning by entity and source', async () => {
    const user = userEvent.setup()
    const mirrored = {
      name: 'Alpha Hotel', code: 'timeout', source: 'Google Hotels', message: 'Provider timed out',
    }
    api.fetchJson.mockImplementation((path, options) => {
      if (path === '/api/destinations') return Promise.resolve(DESTINATIONS)
      if (path === '/api/search-all') {
        return Promise.resolve({
          ...searchResult(options.body, hotel('Available Hotel', 'Phuket', 180, 'available')),
          warnings: [mirrored],
          failedHotels: [
            { ...mirrored },
            {
              name: 'Beta Hotel', code: 'timeout', source: 'Google Hotels', message: 'Provider timed out',
            },
          ],
        })
      }
      return Promise.reject(new Error(`Unexpected API path: ${path}`))
    })
    await renderReady()

    await user.click(screen.getByRole('button', { name: /search all destinations/i }))

    const warningList = await screen.findByRole('list', { name: 'Search warnings' })
    expect(within(warningList).getAllByRole('listitem')).toHaveLength(2)
    expect(warningList).toHaveTextContent('Alpha Hotel')
    expect(warningList).toHaveTextContent('Beta Hotel')
  })

  it('routes accessible single-destination and all-destination searches with their submitted payloads', async () => {
    const user = userEvent.setup()
    const singleHotel = hotel('Exact Phuket Hotel', 'Phuket', 170, 'exact-phuket')
    const allHotel = hotel('All Destinations Hotel', 'Bangkok', 190, 'all-destinations')
    api.fetchJson.mockImplementation((path, options) => {
      if (path === '/api/destinations') return Promise.resolve(DESTINATIONS)
      if (path === '/api/search') {
        return Promise.resolve({
          location: options.body.location,
          checkin: options.body.checkin,
          checkout: options.body.checkout,
          hotels: [singleHotel],
        })
      }
      if (path === '/api/search-all') return Promise.resolve(searchResult(options.body, allHotel))
      return Promise.reject(new Error(`Unexpected API path: ${path}`))
    })
    await renderReady()

    await user.selectOptions(screen.getByLabelText('Search scope'), 'single')
    await user.selectOptions(screen.getByLabelText('Destination'), 'Phuket')
    await user.click(screen.getByRole('button', { name: /search one destination/i }))

    expect(await screen.findByText('Exact Phuket Hotel')).toBeInTheDocument()
    const singleCall = api.fetchJson.mock.calls.find(([path]) => path === '/api/search')
    expect(singleCall[1].body).toMatchObject({
      location: 'Phuket', minStars: 5, maxFlight: 300, nights: 5,
    })

    await user.selectOptions(screen.getByLabelText('Search scope'), 'all')
    await user.click(screen.getByRole('button', { name: /search all destinations/i }))

    expect(await screen.findByText('All Destinations Hotel')).toBeInTheDocument()
    const allCall = api.fetchJson.mock.calls.findLast(([path]) => path === '/api/search-all')
    expect(allCall[1].body).not.toHaveProperty('location')
  })

  it('clears the previous result as soon as a superseding search starts and keeps it clear on failure', async () => {
    const user = userEvent.setup()
    const newerSearch = deferred()
    api.fetchJson.mockImplementation((path, options) => {
      if (path === '/api/destinations') return Promise.resolve(DESTINATIONS)
      if (path === '/api/search-all') {
        if (options.body.checkin === '2026-10-20') return newerSearch.promise
        return Promise.resolve(searchResult(
          options.body,
          hotel('Previous Result', 'Phuket', 180, 'previous-result'),
        ))
      }
      return Promise.reject(new Error(`Unexpected API path: ${path}`))
    })
    await renderReady()

    setSearchDates('2026-09-10', '2026-09-15')
    await user.click(screen.getByRole('button', { name: /search all destinations/i }))
    expect(await screen.findByText('Previous Result')).toBeInTheDocument()

    setSearchDates('2026-10-20', '2026-10-25')
    await user.click(screen.getByRole('button', { name: /search all destinations/i }))
    expect(screen.queryByText('Previous Result')).not.toBeInTheDocument()
    expect(screen.queryByText(/^Stay:/)).not.toBeInTheDocument()

    await act(async () => newerSearch.reject(new Error('new search failed')))
    expect(await screen.findByRole('alert')).toHaveTextContent('Hotel search failed: new search failed')
    expect(screen.queryByText('Previous Result')).not.toBeInTheDocument()
  })

  it('preserves search partial warnings after a successful provider comparison', async () => {
    const user = userEvent.setup()
    const currentHotel = hotel('Warning Merge Hotel', 'Phuket', 180, 'warning-merge')
    api.fetchJson.mockImplementation((path, options) => {
      if (path === '/api/destinations') return Promise.resolve(DESTINATIONS)
      if (path === '/api/search-all') {
        return Promise.resolve({
          ...searchResult(options.body, currentHotel),
          failedDestinations: [{
            code: 'timeout', location: 'Bangkok', message: 'Bangkok search timed out',
          }],
        })
      }
      if (path === '/api/compare-prices') {
        return Promise.resolve({
          hotels: [{ ...currentHotel, providers: { Agoda: { rate: 160 } } }],
          warnings: [{
            code: 'rate_limited', location: 'Phuket', message: 'One provider was rate limited',
          }],
        })
      }
      return Promise.reject(new Error(`Unexpected API path: ${path}`))
    })
    await renderReady()

    await user.click(screen.getByRole('button', { name: /search all destinations/i }))
    expect(await screen.findByText('Bangkok search timed out')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: /compare top 15 providers/i }))

    const warningList = await screen.findByRole('list', { name: 'Search warnings' })
    expect(within(warningList).getAllByRole('listitem')).toHaveLength(2)
    expect(warningList).toHaveTextContent('Bangkok search timed out')
    expect(warningList).toHaveTextContent('One provider was rate limited')
  })
})

describe('destination loading', () => {
  it('shows a loading status without announcing sweep validation before destinations arrive', () => {
    api.fetchJson.mockImplementation(path => {
      if (path === '/api/destinations') return new Promise(() => {})
      return Promise.reject(new Error(`Unexpected API path: ${path}`))
    })

    render(<App />)

    expect(screen.getByText('Loading destinations…')).toHaveAttribute('role', 'status')
    expect(screen.queryByRole('heading', { name: /find cheapest dates/i })).not.toBeInTheDocument()
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  })

  it('shows only the destination operation error when loading fails', async () => {
    api.fetchJson.mockImplementation(path => {
      if (path === '/api/destinations') return Promise.reject(new Error('destination service offline'))
      return Promise.reject(new Error(`Unexpected API path: ${path}`))
    })

    render(<App />)

    const alerts = await screen.findAllByRole('alert')
    expect(alerts).toHaveLength(1)
    expect(alerts[0]).toHaveTextContent('Destination loading failed: destination service offline')
    expect(screen.queryByRole('heading', { name: /find cheapest dates/i })).not.toBeInTheDocument()
  })

  it('keeps destination failure guidance visible while ordinary search errors change', async () => {
    const user = userEvent.setup()
    api.fetchJson.mockImplementation(path => {
      if (path === '/api/destinations') return Promise.reject(new Error('destination service offline'))
      if (path === '/api/search-all') return Promise.reject(new Error('hotel search offline'))
      return Promise.reject(new Error(`Unexpected API path: ${path}`))
    })
    render(<App />)

    const destinationAlert = await screen.findByRole('alert')
    expect(destinationAlert).toHaveTextContent('Destination loading failed')
    expect(destinationAlert).toHaveTextContent(
      'Single-destination search and date sweeps are unavailable',
    )
    expect(screen.getByRole('option', { name: /one destination/i })).toBeDisabled()

    await user.click(screen.getByRole('button', { name: /search all destinations/i }))

    const alerts = await screen.findAllByRole('alert')
    expect(alerts).toHaveLength(2)
    expect(alerts.some(alert => alert.textContent.includes('Destination loading failed'))).toBe(true)
    expect(alerts.some(alert => alert.textContent.includes('Hotel search failed'))).toBe(true)
  })
})

describe('background date sweeps', () => {
  it('polls immediately without overlap, renders running progress, and stops with completed chart and hotels', async () => {
    const firstPoll = deferred()
    api.createSweep.mockResolvedValue({
      jobId: 'sweep-1', status: 'queued', statusUrl: '/api/sweeps/sweep-1',
    })
    api.getSweep
      .mockImplementationOnce(() => firstPoll.promise)
      .mockResolvedValueOnce({
        jobId: 'sweep-1',
        status: 'completed',
        progress: { completed: 2, total: 2 },
        partial: [
          { checkin: '2026-09-01', cheapestPrice: 150, location: 'Phuket' },
          { checkin: '2026-09-08', cheapestPrice: 125, location: 'Bangkok' },
        ],
        result: {
          dates: [
            { checkin: '2026-09-01', cheapestPrice: 150, location: 'Phuket' },
            { checkin: '2026-09-08', cheapestPrice: 125, location: 'Bangkok' },
          ],
          cheapestDate: {
            checkin: '2026-09-08', checkout: '2026-09-13', cheapestPrice: 125, location: 'Bangkok',
          },
          bestDateResults: {
            beachfront: [],
            non_beachfront: [hotel('Sweep Winner', 'Bangkok', 125, 'winner')],
          },
          totalBeachfront: 0,
          totalNonBeachfront: 1,
        },
        warnings: ['One retry was needed'],
        cancelRequested: false,
      })
    await renderReady()
    vi.useFakeTimers()

    fireEvent.click(screen.getByRole('button', { name: /find cheapest dates/i }))
    await act(async () => {})
    expect(api.createSweep).toHaveBeenCalledTimes(1)
    expect(api.getSweep).toHaveBeenCalledTimes(1)
    expect(screen.getByRole('button', { name: /search all destinations/i })).toBeDisabled()

    act(() => vi.advanceTimersByTime(5_000))
    expect(api.getSweep).toHaveBeenCalledTimes(1)

    await act(async () => {
      firstPoll.resolve({
        jobId: 'sweep-1',
        status: 'running',
        progress: { completed: 1, total: 2, currentLocation: 'Bangkok' },
        partial: [{ checkin: '2026-09-01', cheapestPrice: 150, location: 'Phuket' }],
        result: null,
        warnings: [],
        cancelRequested: false,
      })
    })
    expect(screen.getByText(/1 of 2 searches complete/)).toBeInTheDocument()

    act(() => vi.advanceTimersByTime(749))
    expect(api.getSweep).toHaveBeenCalledTimes(1)
    await act(async () => {
      vi.advanceTimersByTime(1)
      await Promise.resolve()
      await Promise.resolve()
    })

    expect(api.getSweep).toHaveBeenCalledTimes(2)
    expect(screen.getByText('Sweep Winner')).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Price by date' })).toBeInTheDocument()
    expect(screen.getByText('One retry was needed')).toBeInTheDocument()

    await act(async () => vi.advanceTimersByTimeAsync(5_000))
    expect(api.getSweep).toHaveBeenCalledTimes(2)
  })

  it('aborts an in-flight poll when the application unmounts', async () => {
    const user = userEvent.setup()
    const pendingPoll = deferred()
    api.createSweep.mockResolvedValue({ jobId: 'sweep-unmount', status: 'queued' })
    api.getSweep.mockImplementation(() => pendingPoll.promise)
    await renderReady()

    await user.click(screen.getByRole('button', { name: /find cheapest dates/i }))
    expect(api.getSweep).toHaveBeenCalledTimes(1)
    const pollSignal = api.getSweep.mock.calls[0][1]

    cleanup()
    expect(pollSignal.aborted).toBe(true)
  })

  it('requests cancellation, polls until the cancelled snapshot, and then stops', async () => {
    api.createSweep.mockResolvedValue({ jobId: 'sweep-cancel', status: 'queued' })
    api.getSweep
      .mockResolvedValueOnce({
        jobId: 'sweep-cancel', status: 'running', progress: { completed: 1, total: 3 },
        partial: [], result: null, warnings: [], cancelRequested: false,
      })
      .mockResolvedValueOnce({
        jobId: 'sweep-cancel', status: 'cancelled', progress: { completed: 1, total: 3 },
        partial: [], result: null, warnings: [], cancelRequested: true,
      })
    api.cancelSweep.mockResolvedValue({
      jobId: 'sweep-cancel', status: 'running', cancelRequested: true,
    })
    await renderReady()
    vi.useFakeTimers()

    fireEvent.click(screen.getByRole('button', { name: /find cheapest dates/i }))
    await act(async () => {})
    expect(screen.getByRole('button', { name: /cancel sweep/i })).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: /cancel sweep/i }))
    await act(async () => {})
    expect(api.cancelSweep).toHaveBeenCalledWith('sweep-cancel', expect.any(AbortSignal))

    await act(async () => vi.advanceTimersByTimeAsync(750))
    expect(screen.getByRole('heading', { name: /date sweep cancelled/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /search all destinations/i })).toBeEnabled()

    await act(async () => vi.advanceTimersByTimeAsync(5_000))
    expect(api.getSweep).toHaveBeenCalledTimes(2)
  })

  it('accepts a terminal DELETE snapshot and aborts a hanging GET poll', async () => {
    const hangingPoll = deferred()
    api.createSweep.mockResolvedValue({ jobId: 'sweep-delete-terminal', status: 'queued' })
    api.getSweep.mockImplementation(() => hangingPoll.promise)
    api.cancelSweep.mockResolvedValue({
      jobId: 'sweep-delete-terminal',
      status: 'cancelled',
      progress: { completed: 0, total: 3 },
      partial: [],
      result: null,
      warnings: [],
      cancelRequested: true,
    })
    await renderReady()
    vi.useFakeTimers()

    fireEvent.click(screen.getByRole('button', { name: /find cheapest dates/i }))
    await act(async () => {})
    const pollSignal = api.getSweep.mock.calls[0][1]

    fireEvent.click(screen.getByRole('button', { name: /cancel sweep/i }))
    await act(async () => {})

    expect(screen.getByRole('heading', { name: /date sweep cancelled/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /search all destinations/i })).toBeEnabled()
    expect(pollSignal.aborted).toBe(true)
    await act(async () => vi.advanceTimersByTimeAsync(5_000))
    expect(api.getSweep).toHaveBeenCalledTimes(1)
  })

  it('applies a completed DELETE snapshot while aborting its hanging poll', async () => {
    const hangingPoll = deferred()
    api.createSweep.mockResolvedValue({ jobId: 'sweep-delete-completed', status: 'queued' })
    api.getSweep.mockImplementation(() => hangingPoll.promise)
    api.cancelSweep.mockResolvedValue({
      jobId: 'sweep-delete-completed',
      status: 'completed',
      progress: { completed: 3, total: 3 },
      partial: [],
      result: {
        cheapestDate: { checkin: '2026-09-01', checkout: '2026-09-02' },
        bestDateResults: {
          beachfront: [hotel('DELETE Winner', 'Phuket', 120, 'delete-winner')],
          non_beachfront: [],
        },
      },
      warnings: [],
      cancelRequested: true,
    })
    await renderReady()

    fireEvent.click(screen.getByRole('button', { name: /find cheapest dates/i }))
    await act(async () => {})
    const pollSignal = api.getSweep.mock.calls[0][1]
    fireEvent.click(screen.getByRole('button', { name: /cancel sweep/i }))
    await act(async () => {})

    expect(screen.getByRole('heading', { name: /date sweep completed/i })).toBeInTheDocument()
    expect(screen.getByText('DELETE Winner')).toBeInTheDocument()
    expect(pollSignal.aborted).toBe(true)
  })

  it('applies canonical failure details from a failed DELETE snapshot', async () => {
    const hangingPoll = deferred()
    api.createSweep.mockResolvedValue({ jobId: 'sweep-delete-failed', status: 'queued' })
    api.getSweep.mockImplementation(() => hangingPoll.promise)
    api.cancelSweep.mockResolvedValue({
      jobId: 'sweep-delete-failed',
      status: 'failed',
      progress: { completed: 0, total: 3 },
      partial: [],
      result: null,
      warnings: [],
      error: { code: 'cancel_failure', message: 'Cancellation could not be finalized' },
      cancelRequested: true,
    })
    await renderReady()

    fireEvent.click(screen.getByRole('button', { name: /find cheapest dates/i }))
    await act(async () => {})
    const pollSignal = api.getSweep.mock.calls[0][1]
    fireEvent.click(screen.getByRole('button', { name: /cancel sweep/i }))
    await act(async () => {})

    expect(screen.getByRole('heading', { name: /date sweep failed/i })).toBeInTheDocument()
    expect(screen.getByRole('alert')).toHaveTextContent(
      'Date sweep failed: Cancellation could not be finalized (cancel_failure)',
    )
    expect(pollSignal.aborted).toBe(true)
  })

  it('disables cancel immediately, sends one DELETE, and unlocks after a real failure', async () => {
    const cancellation = deferred()
    api.createSweep.mockResolvedValue({ jobId: 'sweep-one-delete', status: 'queued' })
    api.getSweep.mockResolvedValue({
      jobId: 'sweep-one-delete', status: 'running', progress: { completed: 0, total: 3 },
      partial: [], result: null, warnings: [], cancelRequested: false,
    })
    api.cancelSweep.mockImplementation(() => cancellation.promise)
    await renderReady()
    vi.useFakeTimers()

    fireEvent.click(screen.getByRole('button', { name: /find cheapest dates/i }))
    await act(async () => {})
    fireEvent.click(screen.getByRole('button', { name: /cancel sweep/i }))

    const lockedCancel = screen.getByRole('button', { name: /cancellation requested/i })
    expect(lockedCancel).toBeDisabled()
    fireEvent.click(lockedCancel)
    expect(api.cancelSweep).toHaveBeenCalledTimes(1)

    await act(async () => cancellation.reject(new Error('DELETE failed')))

    expect(screen.getByRole('alert')).toHaveTextContent('Sweep cancellation failed: DELETE failed')
    expect(screen.getByRole('button', { name: /cancel sweep/i })).toBeEnabled()
  })

  it('keeps a terminal GET snapshot when a slower DELETE response is still running', async () => {
    const cancellation = deferred()
    api.createSweep.mockResolvedValue({ jobId: 'sweep-race', status: 'queued' })
    api.getSweep
      .mockResolvedValueOnce({
        jobId: 'sweep-race', status: 'running', progress: { completed: 1, total: 3 },
        partial: [], result: null, warnings: [], cancelRequested: false,
      })
      .mockResolvedValueOnce({
        jobId: 'sweep-race', status: 'cancelled', progress: { completed: 1, total: 3 },
        partial: [], result: null, warnings: [], cancelRequested: true,
      })
    api.cancelSweep.mockImplementation(() => cancellation.promise)
    await renderReady()
    vi.useFakeTimers()

    fireEvent.click(screen.getByRole('button', { name: /find cheapest dates/i }))
    await act(async () => {})
    fireEvent.click(screen.getByRole('button', { name: /cancel sweep/i }))
    await act(async () => vi.advanceTimersByTimeAsync(750))

    expect(screen.getByRole('heading', { name: /date sweep cancelled/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /search all destinations/i })).toBeEnabled()

    await act(async () => {
      cancellation.resolve({ jobId: 'sweep-race', status: 'running', cancelRequested: true })
    })

    expect(screen.getByRole('heading', { name: /date sweep cancelled/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /search all destinations/i })).toBeEnabled()
    await act(async () => vi.advanceTimersByTimeAsync(5_000))
    expect(api.getSweep).toHaveBeenCalledTimes(2)
  })

  it('does not add a late DELETE rejection after GET has terminalized the sweep', async () => {
    const cancellation = deferred()
    api.createSweep.mockResolvedValue({ jobId: 'sweep-reject-race', status: 'queued' })
    api.getSweep
      .mockResolvedValueOnce({
        jobId: 'sweep-reject-race', status: 'running', progress: { completed: 1, total: 3 },
        partial: [], result: null, warnings: [], cancelRequested: false,
      })
      .mockResolvedValueOnce({
        jobId: 'sweep-reject-race', status: 'cancelled', progress: { completed: 1, total: 3 },
        partial: [], result: null, warnings: [], cancelRequested: true,
      })
    api.cancelSweep.mockImplementation(() => cancellation.promise)
    await renderReady()
    vi.useFakeTimers()

    fireEvent.click(screen.getByRole('button', { name: /find cheapest dates/i }))
    await act(async () => {})
    fireEvent.click(screen.getByRole('button', { name: /cancel sweep/i }))
    const cancellationSignal = api.cancelSweep.mock.calls[0][1]
    await act(async () => vi.advanceTimersByTimeAsync(750))

    expect(screen.getByRole('heading', { name: /date sweep cancelled/i })).toBeInTheDocument()
    expect(cancellationSignal.aborted).toBe(true)
    await act(async () => cancellation.reject(new Error('late DELETE failure')))

    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
    expect(screen.getByRole('heading', { name: /date sweep cancelled/i })).toBeInTheDocument()
  })

  it('clears a cancellation transport error after polling reaches cancelled', async () => {
    api.createSweep.mockResolvedValue({ jobId: 'sweep-cancel-error', status: 'queued' })
    api.getSweep
      .mockResolvedValueOnce({
        jobId: 'sweep-cancel-error', status: 'running', progress: { completed: 1, total: 3 },
        partial: [], result: null, warnings: [], cancelRequested: false,
      })
      .mockResolvedValueOnce({
        jobId: 'sweep-cancel-error', status: 'cancelled', progress: { completed: 1, total: 3 },
        partial: [], result: null, warnings: [], cancelRequested: true,
      })
    api.cancelSweep.mockRejectedValue(new Error('DELETE response was lost'))
    await renderReady()
    vi.useFakeTimers()

    fireEvent.click(screen.getByRole('button', { name: /find cheapest dates/i }))
    await act(async () => {})
    fireEvent.click(screen.getByRole('button', { name: /cancel sweep/i }))
    await act(async () => {})
    expect(screen.getByRole('alert')).toHaveTextContent(
      'Sweep cancellation failed: DELETE response was lost',
    )

    await act(async () => vi.advanceTimersByTimeAsync(750))
    expect(screen.getByRole('heading', { name: /date sweep cancelled/i })).toBeInTheDocument()
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  })

  it('clears sweep-only output as soon as an ordinary search starts', async () => {
    const user = userEvent.setup()
    const pendingSearch = deferred()
    api.createSweep.mockResolvedValue({ jobId: 'sweep-clear', status: 'queued' })
    api.getSweep.mockResolvedValue({
      jobId: 'sweep-clear',
      status: 'completed',
      progress: { completed: 1, total: 1 },
      partial: [{ checkin: '2026-09-01', cheapestPrice: 125, location: 'Bangkok' }],
      result: {
        dates: [{ checkin: '2026-09-01', cheapestPrice: 125, location: 'Bangkok' }],
        cheapestDate: { checkin: '2026-09-01', checkout: '2026-09-06', cheapestPrice: 125, location: 'Bangkok' },
      },
      warnings: [],
      cancelRequested: false,
    })
    api.fetchJson.mockImplementation(path => {
      if (path === '/api/destinations') return Promise.resolve(DESTINATIONS)
      if (path === '/api/search-all') return pendingSearch.promise
      return Promise.reject(new Error(`Unexpected API path: ${path}`))
    })
    await renderReady()

    await user.click(screen.getByRole('button', { name: /find cheapest dates/i }))
    expect(await screen.findByRole('heading', { name: 'Price by date' })).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: /search all destinations/i }))
    expect(screen.queryByRole('heading', { name: 'Price by date' })).not.toBeInTheDocument()
    expect(screen.queryByRole('heading', { name: /date sweep/i })).not.toBeInTheDocument()
  })

  it('shows the requested date range without installing blank query dates when no sweep price exists', async () => {
    const user = userEvent.setup()
    const previousQuery = { checkin: '2026-09-10', checkout: '2026-09-15' }
    api.fetchJson.mockImplementation(path => {
      if (path === '/api/destinations') return Promise.resolve(DESTINATIONS)
      if (path === '/api/search-all') {
        return Promise.resolve(searchResult(
          previousQuery,
          hotel('Previous Search Hotel', 'Phuket', 180, 'previous'),
        ))
      }
      return Promise.reject(new Error(`Unexpected API path: ${path}`))
    })
    api.createSweep.mockResolvedValue({ jobId: 'sweep-empty', status: 'queued' })
    api.getSweep.mockResolvedValue({
      jobId: 'sweep-empty',
      status: 'completed',
      progress: { completed: 2, total: 2 },
      partial: [],
      result: {
        startDate: '2026-11-01',
        endDate: '2026-12-01',
        dates: [],
        cheapestDate: null,
        bestDateResults: { beachfront: [], non_beachfront: [] },
      },
      warnings: [],
      cancelRequested: false,
    })
    await renderReady()

    setSearchDates(previousQuery.checkin, previousQuery.checkout)
    await user.click(screen.getByRole('button', { name: /search all destinations/i }))
    expect(await screen.findByText('Previous Search Hotel')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: /find cheapest dates/i }))

    expect(await screen.findByText(/No priced hotel results were found from 2026-11-01 to 2026-12-01/)).toBeInTheDocument()
    expect(screen.queryByText('Previous Search Hotel')).not.toBeInTheDocument()
    expect(screen.queryByText(/^Stay:/)).not.toBeInTheDocument()
    expect(screen.queryByRole('heading', { name: /hotels and above/i })).not.toBeInTheDocument()
  })

  it('announces a polling failure as a date-sweep error', async () => {
    const user = userEvent.setup()
    api.createSweep.mockResolvedValue({ jobId: 'sweep-error', status: 'queued' })
    api.getSweep.mockRejectedValue(new Error('polling connection lost'))
    await renderReady()

    await user.click(screen.getByRole('button', { name: /find cheapest dates/i }))

    expect(await screen.findByRole('alert')).toHaveTextContent(
      'Date sweep failed: polling connection lost',
    )
  })

  it('announces a terminal failed snapshot as a date-sweep error', async () => {
    const user = userEvent.setup()
    api.createSweep.mockResolvedValue({ jobId: 'sweep-failed', status: 'queued' })
    api.getSweep.mockResolvedValue({
      jobId: 'sweep-failed',
      status: 'failed',
      progress: { completed: 0, total: 2 },
      partial: [],
      result: null,
      warnings: [{ message: 'Generic sweep warning' }],
      error: { code: 'upstream_unavailable', message: 'Hotel providers were unavailable' },
      cancelRequested: false,
    })
    await renderReady()

    await user.click(screen.getByRole('button', { name: /find cheapest dates/i }))

    expect(await screen.findByRole('alert')).toHaveTextContent(
      'Date sweep failed: Hotel providers were unavailable (upstream_unavailable)',
    )
    expect(screen.getByRole('alert')).not.toHaveTextContent('Generic sweep warning')
  })
})
