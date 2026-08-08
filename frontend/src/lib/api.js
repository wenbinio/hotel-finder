export class ApiError extends Error {
  constructor(message, { status = 0, code = 'network_error', details = null } = {}) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.code = code
    this.details = details
  }
}

async function jsonPayload(response) {
  const contentType = response.headers.get('content-type') || ''
  if (!contentType.includes('application/json')) return null
  try {
    return await response.json()
  } catch {
    throw new ApiError('Server returned an invalid JSON response', {
      status: response.status,
      code: 'invalid_response',
    })
  }
}

export async function fetchJson(path, { body, headers, ...options } = {}) {
  let response
  try {
    response = await fetch(path, {
      ...options,
      headers: {
        ...(body === undefined ? {} : { 'content-type': 'application/json' }),
        ...headers,
      },
      body: body === undefined || typeof body === 'string' ? body : JSON.stringify(body),
    })
  } catch (error) {
    if (error?.name === 'AbortError') throw error
    throw new ApiError('Unable to reach the server', { details: error?.message || null })
  }

  const payload = await jsonPayload(response)
  if (!response.ok) {
    const error = payload?.error
    throw new ApiError(error?.message || `Request failed (${response.status})`, {
      status: response.status,
      code: error?.code || 'http_error',
      details: error?.fields || null,
    })
  }
  if (payload === null) {
    throw new ApiError('Server returned a non-JSON response', {
      status: response.status,
      code: 'invalid_response',
    })
  }
  return payload
}

export const createSweep = (payload, signal) => fetchJson('/api/cheapest-dates', {
  method: 'POST', body: payload, signal,
})

export const getSweep = (id, signal) => fetchJson(`/api/sweeps/${encodeURIComponent(id)}`, { signal })

export const cancelSweep = (id, signal) => fetchJson(`/api/sweeps/${encodeURIComponent(id)}`, {
  method: 'DELETE', signal,
})
