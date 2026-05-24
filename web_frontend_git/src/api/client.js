const BASE = '/api'

export async function sendMessage(message, sender = 'user') {
  const res = await fetch(`${BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sender, message }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || 'Chat service unavailable')
  }
  return res.json()
}

export async function getRestaurants({ aspect, polarity, limit = 20, search, cuisine_type } = {}) {
  const params = new URLSearchParams()
  if (aspect)       params.set('aspect', aspect)
  if (polarity)     params.set('polarity', polarity)
  if (search)       params.set('search', search)
  if (cuisine_type) params.set('cuisine_type', cuisine_type)
  params.set('limit', limit)
  const res = await fetch(`${BASE}/restaurants?${params}`)
  if (!res.ok) throw new Error('Failed to fetch restaurants')
  return res.json()
}

export async function getRestaurant(name) {
  const res = await fetch(`${BASE}/restaurants/${encodeURIComponent(name)}`)
  if (!res.ok) throw new Error('Restaurant not found')
  return res.json()
}

export async function getStats() {
  const res = await fetch(`${BASE}/stats`)
  if (!res.ok) throw new Error('Failed to fetch stats')
  return res.json()
}

export async function getBestByMonth({ year, month, limit = 10, aspect, cuisine_type } = {}) {
  const params = new URLSearchParams()
  if (year)         params.set('year', year)
  if (month)        params.set('month', month)
  if (aspect)       params.set('aspect', aspect)
  if (cuisine_type) params.set('cuisine_type', cuisine_type)
  params.set('limit', limit)
  const res = await fetch(`${BASE}/restaurants/best-by-month?${params}`)
  if (!res.ok) throw new Error('Failed to fetch best by month')
  return res.json()
}

export async function getRestaurantInfo(name) {
  const res = await fetch(`/api/restaurant-info/${encodeURIComponent(name)}`)
  if (!res.ok) return null
  return res.json()
}

// Fetches up to 10 live Google Maps reviews with ABSA inference on each.
export async function getLiveReviewsBulk(restaurantName) {
  const res = await fetch(`${BASE}/live-reviews-bulk`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ restaurant_name: restaurantName }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || 'Failed to fetch live reviews')
  }
  return res.json()
}
