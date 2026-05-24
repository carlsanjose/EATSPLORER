import { useState, useEffect, useCallback } from 'react'
import { getRestaurants, getBestByMonth } from '../api/client'
import RestaurantCard from './RestaurantCard'

const ASPECTS = [
  { key: 'overall', label: 'Overall', icon: '⭐' },
  { key: 'food_quality', label: 'Food Quality', icon: '🍴' },
  { key: 'service', label: 'Service', icon: '🛎️' },
  { key: 'ambiance', label: 'Ambiance', icon: '🌿' },
  { key: 'price_value', label: 'Price / Value', icon: '💰' },
]

const POLARITIES = [
  { key: null, label: 'All' },
  { key: 'Positive', label: '✅ Positive' },
  { key: 'Neutral', label: '🟡 Neutral' },
  { key: 'Negative', label: '❌ Negative' },
]

const CUISINE_TYPES = [
  'Filipino', 'Cafe', 'Coffee Shop', 'Korean', 'Japanese',
  'Chinese', 'Fast Food', 'BBQ', 'Pizza', 'Seafood',
  'Bar & Restaurant', 'Asian', 'Breakfast',
]

const LIMITS = [10, 20, 50, 99]

export default function RestaurantBrowser() {
  const [restaurants, setRestaurants] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  const [aspect, setAspect] = useState(null)
  const [polarity, setPolarity] = useState(null)
  const [cuisine, setCuisine] = useState(null)
  const [search, setSearch] = useState('')
  const [limit, setLimit] = useState(20)
  const [selected, setSelected] = useState(null)

  // Best-by-month feature
  const now = new Date()
  const [showMonthly, setShowMonthly] = useState(false)
  const [year, setYear] = useState(now.getFullYear())
  const [month, setMonth] = useState(now.getMonth() + 1)

  const fetchData = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      let data
      if (showMonthly && year && month) {
        data = await getBestByMonth({
          year,
          month,
          limit,
          aspect,
          cuisine_type: cuisine || undefined,
        })
      } else {
        data = await getRestaurants({
          aspect,
          polarity,
          limit,
          search: search || undefined,
          cuisine_type: cuisine || undefined,
        })
      }
      setRestaurants(data)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [aspect, polarity, limit, search, cuisine, showMonthly, year, month])

  useEffect(() => {
    const t = setTimeout(fetchData, 300)
    return () => clearTimeout(t)
  }, [fetchData])

  return (
    <div className="browser">
      <div className="browser-header">
        <h2 className="browser-title">Restaurant Explorer</h2>
        <p className="browser-sub">
          {loading ? 'Loading...' : `${restaurants.length} restaurants`}
        </p>
      </div>

      {/* Search */}
      <div className="browser-search">
        <span className="search-icon">🔍</span>
        <input
          className="search-input"
          type="text"
          placeholder="Search restaurants..."
          value={search}
          onChange={e => setSearch(e.target.value)}
        />
        {search && (
          <button className="search-clear" onClick={() => setSearch('')}>✕</button>
        )}
      </div>

      {/* Best by Month Toggle */}
      <div className="filter-group">
        <span className="filter-label">View mode</span>
        <div className="filter-chips">
          <button
            className={`chip ${!showMonthly ? 'chip-active' : ''}`}
            onClick={() => setShowMonthly(false)}
          >
            📊 All Time
          </button>
          <button
            className={`chip ${showMonthly ? 'chip-active' : ''}`}
            onClick={() => setShowMonthly(true)}
          >
            📅 By Month
          </button>
        </div>
      </div>

      {/* Month/Year Selectors (only show when in monthly mode) */}
      {showMonthly && (
        <div className="filter-group filter-row">
          <span className="filter-label">Select month</span>
          <div className="filter-chips">
            <select
              className="month-year-select"
              value={month}
              onChange={e => setMonth(parseInt(e.target.value))}
            >
              {[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12].map(m => (
                <option key={m} value={m}>
                  {new Date(2000, m - 1).toLocaleString('default', { month: 'long' })}
                </option>
              ))}
            </select>
            <select
              className="month-year-select"
              value={year}
              onChange={e => setYear(parseInt(e.target.value))}
            >
              {[2024, 2025, 2026].map(y => (
                <option key={y} value={y}>
                  {y}
                </option>
              ))}
            </select>
          </div>
        </div>
      )}

      {/* Aspect Filter */}
      <div className="filter-group">
        <span className="filter-label">Sort by aspect</span>
        <div className="filter-chips">
          {ASPECTS.map(a => (
            <button
              key={a.key ?? 'overall'}
              className={`chip ${aspect === a.key ? 'chip-active' : ''}`}
              onClick={() => setAspect(a.key)}
            >
              {a.icon} {a.label}
            </button>
          ))}
        </div>
      </div>

      {/* Cuisine Filter (hide polarity when showing monthly) */}
      <div className="filter-group">
        <span className="filter-label">Cuisine type</span>
        <div className="filter-chips">
          <button
            className={`chip chip-sm ${cuisine === null ? 'chip-active' : ''}`}
            onClick={() => setCuisine(null)}
          >All</button>
          {CUISINE_TYPES.map(c => (
            <button
              key={c}
              className={`chip chip-sm ${cuisine === c ? 'chip-active' : ''}`}
              onClick={() => setCuisine(cuisine === c ? null : c)}
            >{c}</button>
          ))}
        </div>
      </div>

      {/* Polarity Filter */}
      <div className="filter-group">
        <span className="filter-label">Sentiment</span>
        <div className="filter-chips">
          {POLARITIES.map(p => (
            <button
              key={p.key ?? 'all'}
              className={`chip chip-sm ${polarity === p.key ? 'chip-active' : ''}`}
              onClick={() => setPolarity(p.key)}
            >
              {p.label}
            </button>
          ))}
        </div>
      </div>

      {/* Limit */}
      <div className="filter-group filter-row">
        <span className="filter-label">Show</span>
        <div className="filter-chips">
          {LIMITS.map(l => (
            <button
              key={l}
              className={`chip chip-sm ${limit === l ? 'chip-active' : ''}`}
              onClick={() => setLimit(l)}
            >
              {l === 99 ? 'All' : l}
            </button>
          ))}
        </div>
      </div>

      {/* Results */}
      <div className="browser-results">
        {error && (
          <div className="browser-error">
            ⚠️ {error}
            <button onClick={fetchData}>Retry</button>
          </div>
        )}

        {!error && !loading && restaurants.length === 0 && (
          <div className="browser-empty">
            No restaurants found. Try adjusting your filters.
          </div>
        )}

        {loading && (
          <div className="browser-loading">
            {[1, 2, 3, 4, 5].map(i => <div key={i} className="card-skeleton" />)}
          </div>
        )}

        {!loading && restaurants.map((r, i) => (
          <RestaurantCard
            key={r.restaurant_name}
            restaurant={r}
            rank={i + 1}
            activeAspect={aspect}
            isSelected={selected === r.restaurant_name}
            onSelect={() => setSelected(
              selected === r.restaurant_name ? null : r.restaurant_name
            )}
          />
        ))}
      </div>
    </div>
  )
}
