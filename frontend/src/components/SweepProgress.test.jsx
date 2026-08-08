import { cleanup, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import SweepProgress from './SweepProgress'

afterEach(cleanup)

describe('SweepProgress', () => {
  it('announces progress and cancels the canonical jobId', async () => {
    const user = userEvent.setup()
    const onCancel = vi.fn()
    render(
      <SweepProgress
        job={{
          jobId: 'sweep-7',
          status: 'running',
          progress: { completed: 2, total: 6, currentLocation: 'Phuket' },
          partial: [{ checkin: '2026-08-09', cheapestPrice: 120, location: 'Bangkok' }],
          warnings: ['Phuket timed out'],
        }}
        onCancel={onCancel}
      />,
    )

    expect(screen.getByRole('progressbar')).toHaveValue(2)
    expect(screen.getByRole('progressbar')).toHaveAttribute('max', '6')
    expect(screen.getByRole('status')).toHaveTextContent('2 of 6')
    expect(screen.getByText('2026-08-09: $120 in Bangkok')).toBeInTheDocument()
    expect(screen.getByText('Phuket timed out')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Cancel sweep' }))
    expect(onCancel).toHaveBeenCalledTimes(1)
    expect(onCancel).toHaveBeenCalledWith('sweep-7')
  })

  it('renders JSON-safe warning objects as useful text', () => {
    render(
      <SweepProgress
        job={{
          jobId: 'sweep-8',
          status: 'running',
          progress: { completed: 0, total: 2 },
          warnings: [
            { message: 'Phuket timed out', code: 'timeout', location: 'Phuket' },
            { code: 'rate_limited', location: 'Bangkok' },
            { details: ['unexpected', 429] },
          ],
        }}
      />,
    )

    expect(screen.getByText('Phuket timed out')).toBeInTheDocument()
    expect(screen.getByText('rate_limited — Bangkok')).toBeInTheDocument()
    expect(screen.getByText('{"details":["unexpected",429]}')).toBeInTheDocument()
  })

  it('disables repeated cancellation after a cancellation request', () => {
    render(
      <SweepProgress
        job={{
          jobId: 'sweep-cancelling',
          status: 'running',
          progress: { completed: 1, total: 3 },
          cancelRequested: true,
        }}
        onCancel={vi.fn()}
      />,
    )

    expect(screen.getByRole('button', { name: 'Cancellation requested' })).toBeDisabled()
  })

  it('renders indeterminate progress until the backend provides a total', () => {
    render(
      <SweepProgress
        job={{ jobId: 'sweep-queued', status: 'queued', progress: {} }}
        onCancel={vi.fn()}
      />,
    )

    expect(screen.getByRole('status')).toHaveTextContent('Waiting for progress details')
    expect(screen.getByRole('progressbar')).not.toHaveAttribute('value')
    expect(screen.queryByText('0 of 1 searches complete')).not.toBeInTheDocument()
  })
})
