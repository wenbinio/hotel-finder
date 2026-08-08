import { cleanup, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import SearchPanel from './SearchPanel'

afterEach(cleanup)

describe('SearchPanel', () => {
  it('renders future defaults and derives five nights', () => {
    render(<SearchPanel now={new Date(2026, 7, 8, 12)} loading={false} onSubmit={vi.fn()} />)

    expect(screen.getByLabelText('Check-in')).toHaveValue('2026-08-09')
    expect(screen.getByLabelText('Check-out')).toHaveValue('2026-08-14')
    expect(screen.getByText('5 nights')).toBeInTheDocument()
  })

  it('guards an invalid checkout date instead of submitting', async () => {
    const user = userEvent.setup()
    const onSubmit = vi.fn()
    render(<SearchPanel now={new Date(2026, 7, 8, 12)} loading={false} onSubmit={onSubmit} />)

    await user.clear(screen.getByLabelText('Check-out'))
    await user.type(screen.getByLabelText('Check-out'), '2026-08-09')
    await user.click(screen.getByRole('button', { name: 'Search all destinations' }))

    expect(onSubmit).not.toHaveBeenCalled()
    expect(screen.getByRole('alert')).toHaveTextContent('after check-in')
  })
})
