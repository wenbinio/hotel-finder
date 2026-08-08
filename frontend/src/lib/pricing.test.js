import { describe, expect, it } from 'vitest'
import { bestNightlyRate, sortHotels, stableHotelKey } from './pricing'

describe('hotel pricing', () => {
  it('does not merge same-named hotels in different places', () => {
    expect(stableHotelKey({ name: 'Grand Hotel', location: 'Bangkok' }))
      .not.toBe(stableHotelKey({ name: 'Grand Hotel', location: 'Phuket' }))
  })

  it('uses the lowest provider rate when available', () => {
    expect(bestNightlyRate({ price: 100, providers: { x: { rate: 80 }, y: 85 } })).toBe(80)
  })

  it('sorts total trip by the same best rate used for display', () => {
    const sorted = sortHotels([
      { name: 'A', price: 100, providers: { x: { rate: 80 } }, flight_cost: 20 },
      { name: 'B', price: 90, flight_cost: 30 },
    ], 'total', 2)

    expect(sorted.map(hotel => hotel.name)).toEqual(['A', 'B'])
  })
})
