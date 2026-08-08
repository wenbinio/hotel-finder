import { cleanup, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it } from 'vitest'
import HotelResults from './HotelResults'

afterEach(cleanup)

const results = [
  {
    name: 'Grand Hotel', location: 'Bangkok', price: 100, flight_cost: 20, star_class: 4,
    providers: { booking: { rate: 80 } },
    url: 'https://www.google.com/travel/hotels/entity/bangkok',
  },
  {
    name: 'Grand Hotel', location: 'Phuket', price: 90, flight_cost: 30, star_class: 5,
  },
]

describe('HotelResults', () => {
  it('uses the displayed best rate to sort total trip rows', async () => {
    const user = userEvent.setup()
    render(<HotelResults results={results} query={{ minStars: 4 }} nights={2} />)

    expect(screen.getByText('4-star hotels and above')).toBeInTheDocument()
    expect(screen.getByRole('columnheader', { name: 'Estimated flight RT' })).toBeInTheDocument()
    expect(screen.getByText('Hotel results')).toBeInTheDocument()
    expect(screen.queryByText(/null/i)).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Room Quality' })).not.toBeInTheDocument()
    expect(screen.getAllByRole('row')).toHaveLength(3)

    await user.click(screen.getByRole('button', { name: 'Total trip' }))

    const rows = screen.getAllByRole('row')
    expect(rows[1]).toHaveTextContent('Bangkok')
    expect(rows[1]).toHaveTextContent('$80')
    expect(rows[1]).toHaveTextContent('$180')
    expect(screen.getByRole('link', { name: 'Grand Hotel' })).toHaveAttribute('rel', 'noreferrer')
  })
})
