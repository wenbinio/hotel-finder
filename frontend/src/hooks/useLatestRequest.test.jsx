import { act, renderHook } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { useLatestRequest } from './useLatestRequest'

describe('useLatestRequest', () => {
  it('aborts its predecessor and ignores its late completion', async () => {
    const abort = vi.spyOn(AbortController.prototype, 'abort')
    const { result, unmount } = renderHook(() => useLatestRequest())
    let resolveFirst
    const firstCompletion = new Promise(resolve => { resolveFirst = resolve })

    let first
    let second
    let staleOutcome
    act(() => {
      first = result.current.begin()
      staleOutcome = firstCompletion.then(() => first.isCurrent())
      second = result.current.begin()
    })

    expect(abort).toHaveBeenCalledTimes(1)
    expect(first.signal.aborted).toBe(true)
    expect(first.isCurrent()).toBe(false)
    expect(second.isCurrent()).toBe(true)

    resolveFirst()
    await expect(staleOutcome).resolves.toBe(false)

    act(() => first.finish())
    expect(second.isCurrent()).toBe(true)

    unmount()
    expect(second.signal.aborted).toBe(true)
    abort.mockRestore()
  })
})
