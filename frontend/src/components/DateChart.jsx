function priceFor(date) {
  return date.cheapestPrice ?? date.cheapest_price
}

export default function DateChart({ data }) {
  const dates = data?.dates || []
  const pricedDates = dates.filter(date => Number.isFinite(priceFor(date)))
  if (!pricedDates.length) return null

  const cheapest = data.cheapestDate || [...pricedDates].sort((left, right) => priceFor(left) - priceFor(right))[0]
  const maxPrice = Math.max(...pricedDates.map(priceFor))
  const minPrice = Math.min(...pricedDates.map(priceFor))
  const location = data.location || data.locations?.join(', ') || 'selected destinations'

  return (
    <section className="date-chart" aria-labelledby="date-chart-title">
      <h2 id="date-chart-title">Price by date</h2>
      <p className="best-date" role="status">
        Lowest sampled rate is ${priceFor(cheapest)} on {cheapest.checkin} in {cheapest.location || location}.
      </p>
      <p className="chart-summary">
        {pricedDates.map(date => `${date.checkin}: $${priceFor(date)}`).join('; ')}.
      </p>
      <div className="bars" aria-hidden="true">
        {dates.map(date => {
          const price = priceFor(date)
          const percent = Number.isFinite(price) ? ((price - minPrice) / (maxPrice - minPrice || 1)) * 80 + 20 : 0
          const isCheapest = date.checkin === cheapest.checkin
          return (
            <div key={`${date.checkin}-${date.location || ''}`} className={`bar-col ${isCheapest ? 'cheapest' : ''}`}>
              <span className="bar-price">{Number.isFinite(price) ? `$${price}` : '—'}</span>
              <div className="bar" style={{ height: `${percent}%` }} />
              <span className="bar-date">{date.checkin}</span>
            </div>
          )
        })}
      </div>
    </section>
  )
}
