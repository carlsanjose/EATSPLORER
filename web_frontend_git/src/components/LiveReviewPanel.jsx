import { useState } from 'react'
import { getLiveReviewsBulk } from '../api/client'

const ASPECTS = [
  { key: 'overall', label: 'Overall', icon: '⭐' },
  { key: 'food_quality', label: 'Food', icon: '🍴' },
  { key: 'service', label: 'Service', icon: '🛎️' },
  { key: 'ambiance', label: 'Ambiance', icon: '🌿' },
  { key: 'price_value', label: 'Value', icon: '💰' },
]

const POL_COLOR = {
  Positive: 'var(--positive)',
  Neutral: 'var(--neutral)',
  Negative: 'var(--negative)',
}
const POL_EMOJI = { Positive: '✅', Neutral: '🟡', Negative: '❌' }

function StarRating({ rating }) {
  if (!rating) return null
  return (
    <span className="live-stars">
      {'★'.repeat(rating)}{'☆'.repeat(5 - rating)}
    </span>
  )
}

function AbsaGrid({ absa }) {
  if (!absa) return <p className="absa-na">Eatsplorer analysis unavailable</p>
  return (
    <div className="absa-grid">
      {ASPECTS.map(a => {
        const d = absa[a.key]
        if (!d || d.polarity === 'N/A') {
          return (
            <div key={a.key} className="absa-cell absa-na-cell">
              <span className="absa-icon">{a.icon}</span>
              <span className="absa-label">{a.label}</span>
              <span className="absa-val absa-na-val">N/A</span>
            </div>
          )
        }
        const color = POL_COLOR[d.polarity] || '#888'
        return (
          <div key={a.key} className="absa-cell">
            <span className="absa-icon">{a.icon}</span>
            <span className="absa-label">{a.label}</span>
            <span className="absa-val" style={{ color }}>
              {POL_EMOJI[d.polarity]} {d.score != null ? d.score.toFixed(2) : ''}
            </span>
          </div>
        )
      })}
    </div>
  )
}

function ReviewItem({ review, index }) {
  const [open, setOpen] = useState(false)
  return (
    <div className="live-review-item">
      <div className="live-review-header" onClick={() => setOpen(v => !v)}>
        <div className="live-review-meta">
          <span className="live-review-author">{review.author}</span>
          <StarRating rating={review.rating} />
          <span className="live-review-time">{review.time}</span>
        </div>
        <span className="live-review-toggle">{open ? '▲' : '▼'}</span>
      </div>
      {open && (
        <div className="live-review-body">
          <p className="live-review-text">"{review.text}"</p>
          <AbsaGrid absa={review.absa_inference} />
        </div>
      )}
    </div>
  )
}

export default function LiveReviewPanel({ restaurantName }) {
  const [state, setState] = useState('idle') // idle | loading | done | error
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)

  async function fetchReviews() {
    setState('loading')
    setError(null)
    try {
      const result = await getLiveReviewsBulk(restaurantName)
      if (result.error) {
        setError(result.error)
        setState('error')
      } else {
        setData(result)
        setState('done')
      }
    } catch (e) {
      setError(e.message)
      setState('error')
    }
  }

  if (state === 'idle') {
    return (
      <button className="live-reviews-btn" onClick={fetchReviews}>
        🔴 Load Live Reviews
      </button>
    )
  }

  if (state === 'loading') {
    return (
      <div className="live-panel">
        <div className="live-loading">
          <span className="live-spinner" />
          Fetching live reviews
        </div>
      </div>
    )
  }

  if (state === 'error') {
    return (
      <div className="live-panel">
        <p className="live-error">⚠️ {error}</p>
        <button className="live-reviews-btn" onClick={fetchReviews}>Retry</button>
      </div>
    )
  }

  // done
  return (
    <div className="live-panel">
      <div className="live-panel-header">
        <span className="live-panel-title">
          🔴 Live Reviews — {data.found_name}
        </span>
        {data.google_rating && (
          <span className="live-google-rating">
            ★ {data.google_rating} Google
          </span>
        )}
        <button className="live-panel-close" onClick={() => setState('idle')}>✕</button>
      </div>

      {data.reviews.length === 0 ? (
        <p className="live-error">No recent text reviews found.</p>
      ) : (
        <div className="live-reviews-list">
          {data.reviews.map((rv, i) => (
            <ReviewItem key={i} review={rv} index={i} />
          ))}
        </div>
      )}
    </div>
  )
}
