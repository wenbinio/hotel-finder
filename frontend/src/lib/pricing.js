function numericRate(value) {
  const rate = typeof value === 'object' && value !== null ? value.rate : value
  return Number.isFinite(rate) && rate >= 0 ? rate : null
}

function compareText(left, right) {
  return String(left || '').localeCompare(String(right || ''), undefined, { sensitivity: 'base' })
}

function normalizedGoogleEntityUrl(value) {
  if (typeof value !== 'string' || !value.trim()) return null
  try {
    const url = new URL(value)
    if (
      url.protocol !== 'https:'
      || url.hostname !== 'www.google.com'
      || url.port
      || url.username
      || url.password
      || !url.pathname.startsWith('/travel/hotels/entity/')
    ) return null
    url.search = ''
    url.hash = ''
    return url.href
  } catch {
    return null
  }
}

export function stableHotelKey(hotel) {
  const entityUrl = normalizedGoogleEntityUrl(hotel.url)
  if (entityUrl) return JSON.stringify(['url', entityUrl])
  const location = String(hotel.location || '').trim().toLowerCase()
  const name = String(hotel.name || '').trim().toLowerCase()
  return JSON.stringify(['hotel', location, name])
}

export function bestNightlyRate(hotel) {
  const rates = [numericRate(hotel.price)]
  for (const provider of Object.values(hotel.providers || {})) rates.push(numericRate(provider))
  const validRates = rates.filter(rate => rate !== null)
  return validRates.length ? Math.min(...validRates) : null
}

export function tripTotal(hotel, nights) {
  const rate = bestNightlyRate(hotel)
  const flight = numericRate(hotel.flight_cost) || 0
  return rate === null ? null : rate * nights + flight
}

export function sortHotels(hotels, sortKey = 'price', nights = 1) {
  const valueFor = hotel => {
    if (sortKey === 'total') return tripTotal(hotel, nights)
    if (sortKey === 'rating') return numericRate(hotel.rating)
    if (sortKey === 'best') return bestNightlyRate(hotel)
    return numericRate(hotel.price)
  }
  const direction = sortKey === 'rating' ? -1 : 1

  return [...hotels].sort((left, right) => {
    const leftValue = valueFor(left)
    const rightValue = valueFor(right)
    if (leftValue === null) return rightValue === null ? compareText(stableHotelKey(left), stableHotelKey(right)) : 1
    if (rightValue === null) return -1
    if (leftValue !== rightValue) return direction * (leftValue - rightValue)
    return compareText(stableHotelKey(left), stableHotelKey(right))
  })
}
