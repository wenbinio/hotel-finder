function localDate(year, month, day) {
  return new Date(year, month, day)
}

function formatLocalDate(date) {
  const year = date.getFullYear()
  const month = String(date.getMonth() + 1).padStart(2, '0')
  const day = String(date.getDate()).padStart(2, '0')
  return `${year}-${month}-${day}`
}

function parseLocalDate(value) {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(value)
  if (!match) return null

  const [, year, month, day] = match.map(Number)
  const date = localDate(year, month - 1, day)
  return date.getFullYear() === year && date.getMonth() === month - 1 && date.getDate() === day
    ? date
    : null
}

function addLocalDays(date, days) {
  return localDate(date.getFullYear(), date.getMonth(), date.getDate() + days)
}

export function nightsBetween(checkin, checkout) {
  const start = parseLocalDate(checkin)
  const end = parseLocalDate(checkout)
  if (!start || !end) return 0
  return Math.round((end.getTime() - start.getTime()) / 86_400_000)
}

export function defaultSearchDates(now = new Date()) {
  const checkinDate = addLocalDays(now, 1)
  const checkoutDate = addLocalDays(checkinDate, 5)
  const checkin = formatLocalDate(checkinDate)
  const checkout = formatLocalDate(checkoutDate)
  return { checkin, checkout, nights: nightsBetween(checkin, checkout) }
}

export function defaultSweepDates(now = new Date()) {
  const startDate = formatLocalDate(addLocalDays(now, 1))
  const endDate = formatLocalDate(addLocalDays(now, 91))
  return { startDate, endDate, nights: 1, sampleCount: 6 }
}
