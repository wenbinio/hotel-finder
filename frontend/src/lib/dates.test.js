import { describe, expect, it } from 'vitest'
import { defaultSearchDates, defaultSweepDates, nightsBetween } from './dates'

describe('date defaults', () => {
  it('uses local tomorrow and a five-night stay', () => {
    expect(defaultSearchDates(new Date(2026, 7, 8, 12))).toEqual({
      checkin: '2026-08-09', checkout: '2026-08-14', nights: 5,
    })
  })

  it('uses a 90-day sweep window and six samples', () => {
    expect(defaultSweepDates(new Date(2026, 7, 8, 12))).toEqual({
      startDate: '2026-08-09', endDate: '2026-11-07', nights: 1, sampleCount: 6,
    })
  })

  it('derives nights from authoritative local dates', () => {
    expect(nightsBetween('2026-08-09', '2026-08-14')).toBe(5)
  })
})
