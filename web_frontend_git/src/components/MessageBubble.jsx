import { useState } from 'react'
import RestaurantCard from './RestaurantCard'

export default function MessageBubble({ message, onButtonClick }) {
  // Add 'custom' to your destructuring to catch the new map payload from Rasa
  const { role, text, buttons, cards, isError, custom } = message
  const isBot = role === 'bot'
  const [selectedCard, setSelectedCard] = useState(null)

  // Helper to generate the Google Maps Embed URL based on the name and address
  const getMapUrl = (name, address) => {
    const query = encodeURIComponent(`${name} ${address} Legazpi City`);
    return `https://maps.google.com/maps?q=${query}&t=&z=15&ie=UTF8&iwloc=&output=embed`;
  };

  return (
    <div className={`message ${isBot ? 'bot-message' : 'user-message'}`}>
      {isBot && <div className="bot-avatar">🍽️</div>}
      <div className={isBot ? `bot-response${isError ? ' bot-response-error' : ''}` : 'user-bubble'}>

        {/* Conversational header when cards follow; regular prose otherwise */}
        {text && (
          <div
            className={cards && cards.length > 0 ? 'response-header' : 'response-text'}
            dangerouslySetInnerHTML={{ __html: renderMarkdown(text) }}
          />
        )}

        {/* NEW: Inline Google Map (Triggers ONLY when action_location is called) */}
        {custom && custom.payload === 'inline_map' && (
          <div className="inline-map-container" style={{ marginTop: '10px', borderRadius: '8px', overflow: 'hidden' }}>
            <iframe
              title={`Map of ${custom.restaurant_name}`}
              width="100%"
              height="250"
              frameBorder="0"
              scrolling="no"
              marginHeight="0"
              marginWidth="0"
              src={getMapUrl(custom.restaurant_name, custom.address)}
              style={{ border: 0 }}
              allowFullScreen
            ></iframe>
          </div>
        )}

        {/* Restaurant cards */}
        {cards && cards.length > 0 && (
          <div className="chat-cards">
            {cards.map((r, i) => (
              <RestaurantCard
                key={r.restaurant_name}
                restaurant={r}
                rank={i + 1}
                activeAspect={message.aspect || null}
                isSelected={selectedCard === r.restaurant_name}
                onSelect={() => setSelectedCard(
                  selectedCard === r.restaurant_name ? null : r.restaurant_name
                )}
              />
            ))}
          </div>
        )}

        {buttons && buttons.length > 0 && (
          <div className="bubble-buttons">
            {buttons.map((btn, i) => (
              <button key={i} className="bubble-btn" onClick={() => onButtonClick(btn)}>
                {btn.title}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

function renderMarkdown(text) {
  if (!text) return ''
  return text
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/\n/g, '<br/>')
    .replace(/^---$/gm, '<hr class="response-hr"/>')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)/g, '<em>$1</em>')
    .replace(
      /(https:\/\/www\.google\.com\/maps\/dir\/[^\s<]+)/g,
      '<a href="$1" target="_blank" rel="noopener noreferrer" class="nav-btn">🧭 Open Navigation</a>'
    )
    .replace(
      /(https:\/\/maps\.google\.com[^\s<]+)/g,
      '<a href="$1" target="_blank" rel="noopener noreferrer" class="maps-link-inline">🗺️ Open in Maps</a>'
    )
    .replace(
      /(?<!href=")(https?:\/\/[^\s<"]+)(?!")/g,
      '<a href="$1" target="_blank" rel="noopener noreferrer">$1</a>'
    )
}