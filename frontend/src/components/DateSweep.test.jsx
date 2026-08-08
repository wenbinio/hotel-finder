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

  it('rejects a single location that is not in the allowed destination list', async () => {
    const user = userEvent.setup()
    render(<DateSweep destinations={DESTINATIONS} onSubmit={vi.fn()} />)

    await user.click(screen.getByRole('button', { name: 'Single location' }))
    await user.type(screen.getByLabelText('Destination'), 'Atlantis')

    expect(screen.getByRole('button', { name: 'Find cheapest dates' })).toBeDisabled()
    expect(screen.getByRole('alert')).toHaveTextContent('listed destination')
  })

  it('canonicalizes a case-insensitive destination match before submission', async () => {
    const user = userEvent.setup()
    const onSubmit = vi.fn()
    render(<DateSweep destinations={DESTINATIONS} onSubmit={onSubmit} />)

    await user.click(screen.getByRole('button', { name: 'Single location' }))
    await user.type(screen.getByLabelText('Destination'), 'pHuKeT')
    await user.click(screen.getByRole('button', { name: 'Find cheapest dates' }))

    expect(onSubmit).toHaveBeenCalledWith(expect.objectContaining({ locations: ['Phuket'] }))
  })

  it.each([0, 31])('disables nights outside 1..30: %s', nights => {
    render(<DateSweep destinations={DESTINATIONS} value={{ nights }} onSubmit={vi.fn()} />)

    expect(screen.getByRole('button', { name: 'Find cheapest dates' })).toBeDisabled()
    expect(screen.getByRole('alert')).toHaveTextContent('Nights must be between 1 and 30')
  })

  it.each([0, 11])('disables sample counts outside 1..10: %s', sampleCount => {
    render(<DateSweep destinations={DESTINATIONS} value={{ sampleCount }} onSubmit={vi.fn()} />)

    expect(screen.getByRole('button', { name: 'Find cheapest dates' })).toBeDisabled()
    expect(screen.getByRole('alert')).toHaveTextContent('Samples must be between 1 and 10')
  })

  it.each([[1, 1], [30, 10]])('accepts bounded nights %s and samples %s', (nights, sampleCount) => {
    render(<DateSweep destinations={DESTINATIONS} value={{ nights, sampleCount }} onSubmit={vi.fn()} />)

    expect(screen.getByRole('button', { name: 'Find cheapest dates' })).toBeEnabled()
  })

  it('allows 200 logical searches and rejects 201 or more', () => {
    const makeDestinations = count => ({
      non_beachfront: Array.from({ length: count }, (_, index) => ({ name: `City ${index + 1}` })),
    })
    const { rerender } = render(
      <DateSweep destinations={makeDestinations(20)} value={{ sampleCount: 10 }} onSubmit={vi.fn()} />,
    )
    expect(screen.getByRole('button', { name: 'Find cheapest dates' })).toBeEnabled()

    rerender(<DateSweep destinations={makeDestinations(21)} value={{ sampleCount: 10 }} onSubmit={vi.fn()} />)
    expect(screen.getByRole('button', { name: 'Find cheapest dates' })).toBeDisabled()
    expect(screen.getByRole('alert')).toHaveTextContent('200 logical hotel searches')
  })
})
