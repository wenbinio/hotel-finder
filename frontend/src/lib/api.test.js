import { afterEach, describe, expect, it, vi } from 'vitest'
import { ApiError, cancelSweep, createSweep, fetchJson, getSweep } from './api'

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('fetchJson', () => {
  it('turns an HTML 500 into a friendly ApiError', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('<h1>error</h1>', {
      status: 500,
      headers: { 'content-type': 'text/html' },
    })))

    await expect(fetchJson('/api/search')).rejects.toMatchObject({
      name: 'ApiError', status: 500, code: 'http_error',
    })
  })

  it('rejects a successful non-JSON response with a typed error', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('okay', {
      status: 200,
      headers: { 'content-type': 'text/plain' },
    })))

    await expect(fetchJson('/api/search')).rejects.toBeInstanceOf(ApiError)
    await expect(fetchJson('/api/search')).rejects.toMatchObject({ code: 'invalid_response' })
  })

  it('preserves a structured backend error', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({
      error: { code: 'invalid_dates', message: 'Choose future dates', fields: { checkin: 'past' } },
    }), { status: 422, headers: { 'content-type': 'application/json' } })))

    await expect(fetchJson('/api/search')).rejects.toMatchObject({
      code: 'invalid_dates', details: { checkin: 'past' },
    })
  })
})

describe('sweep API', () => {
  it('uses only relative API paths and encodes sweep IDs', async () => {
    const fetch = vi.fn().mockImplementation(() => Promise.resolve(new Response('{}', {
      headers: { 'content-type': 'application/json' },
    })))
    vi.stubGlobal('fetch', fetch)

    await createSweep({ locations: ['Bangkok'] })
    await getSweep('job/a b')
    await cancelSweep('job/a b')

    expect(fetch.mock.calls.map(([path]) => path)).toEqual([
      '/api/cheapest-dates', '/api/sweeps/job%2Fa%20b', '/api/sweeps/job%2Fa%20b',
    ])
    expect(fetch.mock.calls[0][1]).toMatchObject({ method: 'POST', body: '{"locations":["Bangkok"]}' })
    expect(fetch.mock.calls[2][1]).toMatchObject({ method: 'DELETE' })
  })
})
