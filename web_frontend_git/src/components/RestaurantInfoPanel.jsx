import { useState, useEffect } from 'react'
import { getRestaurantInfo } from '../api/client'

const POLARITY_COLOR = {
  Positive: 'var(--positive)',
  Neutral:  'var(--neutral)',
  Negative: 'var(--negative)',
}

export default function RestaurantInfoPanel({ restaurantName, compositeScore, compositePolarity }) {
  const [info, setInfo]       = useState(null)
  const [loading, setLoading] = useState(true)
  const [mapOpen, setMapOpen] = useState(false)

  useEffect(() => {
    setLoading(true)
    setMapOpen(false)
    getRestaurantInfo(restaurantName)
      .then(data => { setInfo(data); setLoading(false) })
      .catch(() => { setInfo(null); setLoading(false) })
  }, [restaurantName])

  if (loading) return <div className="info-loading">Loading details...</div>
  if (!info && !restaurantName) return null

  const hasDish    = info?.best_dish_signature_dish
  const hasAddress = info?.address
  const hasSummary = info?.quick_summary
  const hasCuisine = info?.cuisine_type

  // Embed URL — name + address pinpoints the exact business.
  // MUST use maps.google.com/maps?q=...&output=embed — this is the only
  // format browsers allow in iframes without a paid API key.
  // maps/search/?api=1 and maps/embed/v1 both require an API key and show
  // a blocked icon without one.
  const embedQuery = info?.address
    ? `${restaurantName}, ${info.address}`
    : `${restaurantName}, Legazpi City, Albay, Philippines`
  const embedUrl = `https://maps.google.com/maps?q=${encodeURIComponent(embedQuery)}&output=embed`

  // Directions URL — use full address for accurate routing
  const dest   = info?.address || `${restaurantName} Legazpi City Philippines`
  const navUrl = `https://www.google.com/maps/dir/?api=1&destination=${encodeURIComponent(dest)}`

  if (!hasDish && !hasAddress && !hasSummary && !hasCuisine) {
    // Still show map even without info CSV data
    return (
      <div className="info-panel" onClick={e => e.stopPropagation()}>
        <div className="info-divider" />
        <MapSection
          embedUrl={embedUrl} navUrl={navUrl}
          mapOpen={mapOpen} setMapOpen={setMapOpen}
          restaurantName={restaurantName}
        />
      </div>
    )
  }

  const hasComposite = compositeScore != null
  const compositeColor = POLARITY_COLOR[compositePolarity] || 'var(--text-muted)'

  return (
    <div className="info-panel" onClick={e => e.stopPropagation()}>
      <div className="info-divider" />
      
      {hasCuisine && <div className="cuisine-badge">{info.cuisine_type}</div>}
      {hasSummary && <p className="info-summary">{info.quick_summary}</p>}
      {hasDish && (
        <div className="info-row">
          <span className="info-icon">🍽️</span>
          <div>
            <div className="info-label">Signature dish</div>
            <div className="info-value">{info.best_dish_signature_dish}</div>
          </div>
        </div>
      )}
      {hasAddress && (
        <div className="info-row">
          <span className="info-icon">📍</span>
          <div>
            <div className="info-label">Address</div>
            <div className="info-value">{info.address}</div>
          </div>
        </div>
      )}
      <MapSection
        embedUrl={embedUrl} navUrl={navUrl}
        mapOpen={mapOpen} setMapOpen={setMapOpen}
        restaurantName={restaurantName}
      />
    </div>
  )
}

function MapSection({ embedUrl, navUrl, mapOpen, setMapOpen, restaurantName }) {
  return (
    <>
      <div className="map-actions">
        <button
          className={`map-toggle-btn ${mapOpen ? 'map-toggle-active' : ''}`}
          onClick={() => setMapOpen(v => !v)}
        >
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="13" height="13">
            <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/>
            <circle cx="12" cy="10" r="3"/>
          </svg>
          {mapOpen ? 'Hide map' : 'Show map'}
        </button>
        <a
          className="nav-directions-btn"
          href={navUrl}
          target="_blank"
          rel="noopener noreferrer"
          onClick={e => e.stopPropagation()}
        >
          🧭 Get Directions
        </a>
      </div>
      {mapOpen && (
        <div className="map-embed-wrap">
          <iframe
            className="map-embed"
            src={embedUrl}
            allowFullScreen
            loading="lazy"
            referrerPolicy="no-referrer-when-downgrade"
            title={`Map of ${restaurantName}`}
          />
        </div>
      )}
    </>
  )
}
