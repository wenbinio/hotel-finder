import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
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
      warnings: [{ message: 'Hotel providers were unavailable' }],
      cancelRequested: false,
    })
    await renderReady()

    await user.click(screen.getByRole('button', { name: /find cheapest dates/i }))

    expect(await screen.findByRole('alert')).toHaveTextContent(
      'Date sweep failed: Hotel providers were unavailable',
    )
  })
})
