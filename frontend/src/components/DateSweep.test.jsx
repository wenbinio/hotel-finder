import { cleanup, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import DateSweep from './DateSweep'

const DESTINATIONS = {
  beachfront: [{ name: 'Phuket' }],
  non_beachfront: [{ name: 'Bangkok' }],
}

afterEach(cleanup)

describe('DateSweep', () => {
  it('does not submit blank single-location sweep', async () => {
    const user = userEvent.setup()
    const onSubmit = vi.fn()
    render(<DateSweep destinations={DESTINATIONS} onSubmit={onSubmit} />)

    await user.click(screen.getByRole('button', { name: 'Single location' }))

    expect(screen.getByRole('button', { name: 'Find cheapest dates' })).toBeDisabled()
  })

  it('uses a six-sample 90-day default window', () => {
    render(<DateSweep destinations={DESTINATIONS} now={new Date(2026, 7, 8, 12)} onSubmit={vi.fn()} />)

    expect(screen.getByLabelText('From')).toHaveValue('2026-08-09')
    expect(screen.getByLabelText('To')).toHaveValue('2026-11-07')
    expect(screen.getByLabelText('Samples')).toHaveValue(6)
  })
})
