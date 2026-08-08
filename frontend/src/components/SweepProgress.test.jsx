import { cleanup, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import SweepProgress from './SweepProgress'

afterEach(cleanup)

describe('SweepProgress', () => {
  it('announces progress, partial results, warnings, and cancellation', async () => {
    const user = userEvent.setup()
    const onCancel = vi.fn()
    render(
      <SweepProgress
        job={{
          id: 'sweep-7',
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
})
