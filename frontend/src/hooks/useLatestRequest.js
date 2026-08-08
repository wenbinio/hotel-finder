import { useCallback, useEffect, useRef } from 'react'

export function useLatestRequest() {
  const current = useRef(null)

  const begin = useCallback(() => {
    current.current?.controller.abort()
    const request = { controller: new AbortController() }
    current.current = request

    return {
      signal: request.controller.signal,
      isCurrent: () => current.current === request && !request.controller.signal.aborted,
      finish: () => {
        if (current.current === request) current.current = null
      },
    }
  }, [])

  useEffect(() => () => {
    current.current?.controller.abort()
    current.current = null
  }, [])

  return { begin }
}
