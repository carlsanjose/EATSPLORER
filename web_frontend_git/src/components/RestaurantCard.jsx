import { useState } from 'react'
import RestaurantInfoPanel from './RestaurantInfoPanel'
import LiveReviewPanel from './LiveReviewPanel'

const POLARITY_COLOR = {
  Positive: 'var(--positive)',
  Neutral: 'var(--neutral)',
  Negative: 'var(--negative)',
}

const POLARITY_EMOJI = {
  Positive: '✅',
  Neutral: '🟡',
  Negative: '❌',
}

const ASPECTS = [
  { key: 'overall', label: 'Overall', icon: '⭐' },
  { key: 'food_quality', label: 'Food Quality', icon: '🍴' },
  { key: 'service', label: 'Service', icon: '🛎️' },
  { key: 'ambiance', label: 'Ambiance', icon: '🌿' },
  { key: 'price_value', label: 'Price / Value', icon: '💰' },
]

export default function RestaurantCard({ restaurant: r, rank, activeAspect, isSelected, onSelect }) {
  const compositeScore = r.composite_score
  const compositePolarity = r.composite_polarity
  const compositeColor = POLARITY_COLOR[compositePolarity] || 'var(--text-muted)'
  const [showLive, setShowLive] = useState(false)

  return (
    <div
      className={`restaurant-card ${isSelected ? 'card-expanded' : ''}`}
      onClick={onSelect}
    >
      {/* Rank + Name */}
      <div className="card-header">
        <div className="card-rank">#{rank}</div>
        <div className="card-info">
          <div className="card-name">{r.restaurant_name}</div>
          <div className="card-meta">
            <span style={{ color: compositeColor }}>
              {POLARITY_EMOJI[compositePolarity] || '⬜'} {compositePolarity || 'N/A'}
            </span>
            <span className="card-dot">·</span>
            <span className="card-reviews">{r.total_reviews} reviews</span>
          </div>
        </div>
        <div className="card-score-badge" style={{ borderColor: compositeColor }}>
          <span className="score-num">
            {compositeScore != null ? compositeScore.toFixed(2) : 'N/A'}
          </span>
          <span className="score-denom">/5</span>
        </div>
      </div>

      {/* Highlighted aspect (if filtering by one) */}
      {(() => {
        const aspectDef = ASPECTS.find(a => a.key === activeAspect)
        const aspectData = activeAspect === 'overall'
          ? r.overall
          : activeAspect && r[activeAspect]

        if (!aspectData || aspectData.avg == null) return null

        return (
          <div className="card-highlight">
            {aspectDef?.icon || '⭐'}{' '}
            <strong>{aspectDef?.label || 'Overall'}:</strong>{' '}
            {aspectData.avg.toFixed(2)}/5.00{' '}
            <span style={{ color: POLARITY_COLOR[aspectData.polarity] }}>
              ({aspectData.polarity})
            </span>
          </div>
        )
      })()}

      {/* Composite score bar */}
      <div className="score-bar-row">
        <ScoreBar value={compositeScore} color={compositeColor} />
      </div>

      {/* Expanded section */}
      {isSelected && (
        <div className="card-aspects" onClick={e => e.stopPropagation()}>
          <div className="aspects-divider" />

          {/* Aspect breakdown */}
          {ASPECTS.map(a => {
            const asp = a.key === 'overall'
              ? r.overall
              : r[a.key]
            const hasData = asp && asp.avg != null
            const color = hasData ? (POLARITY_COLOR[asp.polarity] || 'var(--text-muted)') : 'var(--border)'
            return (
              <div key={a.key} className="aspect-row">
                <span className="aspect-icon">{a.icon}</span>
                <span className="aspect-label">{a.label}</span>
                {hasData ? (
                  <>
                    <div className="aspect-bar-wrap">
                      <div
                        className="aspect-bar-fill"
                        style={{ width: `${(asp.avg / 5) * 100}%`, background: color }}
                      />
                    </div>
                    <span className="aspect-score" style={{ color }}>
                      {asp.avg.toFixed(2)}
                    </span>
                    <span className="aspect-count">({asp.review_count})</span>
                  </>
                ) : (
                  <span className="aspect-na">— N/A</span>
                )}
              </div>
            )
          })}

          {/* Restaurant info: address, signature dish, map */}
          <RestaurantInfoPanel
            restaurantName={r.restaurant_name}
            compositeScore={r.composite_score}
            compositePolarity={r.composite_polarity}
          />

          {/* Live Reviews + ABSA panel */}
          <div className="aspects-divider" />
          {showLive ? (
            <LiveReviewPanel restaurantName={r.restaurant_name} />
          ) : (
            <button
              className="live-reviews-btn"
              onClick={() => setShowLive(true)}
            >
              🔴 Live Reviews
            </button>
          )}

          <div className="aspects-footer">
            {r.aspects_scored}/5 aspects scored · {r.total_reviews} total reviews
          </div>
        </div>
      )}

      <div className="card-expand-hint">
        {isSelected ? '▲ Less' : '▼ Details'}
      </div>
    </div>
  )
}

function ScoreBar({ value, color }) {
  const pct = value != null ? (value / 5) * 100 : 0
  return (
    <div className="score-bar-track">
      <div
        className="score-bar-fill"
        style={{ width: `${pct}%`, background: color }}
      />
    </div>
  )
}
