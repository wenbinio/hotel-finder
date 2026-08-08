import { afterEach, describe, expect, it, vi } from 'vitest'
import { ApiError, cancelSweep, createSweep, fetchJson, getSweep } from './api'

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('fetchJson', () => {
  it('keeps a caller content type as the single case-insensitive header', async () => {
    let sentHeaders
    vi.stubGlobal('fetch', vi.fn().mockImplementation((_path, options) => {
      sentHeaders = options.headers
      return Promise.resolve(new Response('{}', {
        headers: { 'content-type': 'application/json' },
      }))
    }))

    await fetchJson('/api/search', {
      method: 'POST',
      body: {},
      headers: { 'Content-Type': 'application/merge-patch+json' },
    })

    expect(sentHeaders).toBeInstanceOf(Headers)
    expect([...sentHeaders.entries()]).toEqual([
      ['content-type', 'application/merge-patch+json'],
    ])
  })

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

  it('recognizes JSON content types without case sensitivity', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('{"ok":true}', {
      headers: { 'content-type': 'Application/JSON; Charset=UTF-8' },
    })))

    await expect(fetchJson('/api/search')).resolves.toEqual({ ok: true })
  })

  it('rethrows an AbortError raised while parsing the response body', async () => {
    const abort = new DOMException('The operation was aborted', 'AbortError')
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      headers: new Headers({ 'content-type': 'application/json' }),
      json: vi.fn().mockRejectedValue(abort),
    }))

    await expect(fetchJson('/api/search')).rejects.toBe(abort)
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
