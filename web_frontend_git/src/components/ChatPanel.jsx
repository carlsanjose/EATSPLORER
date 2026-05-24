import { useState, useRef, useEffect, useCallback } from 'react'
import { sendMessage } from '../api/client'
import MessageBubble from './MessageBubble'

const SUGGESTIONS = [
  'Top 3 restaurants with best food quality',
  'Budget-friendly café with great ambiance',
  'Tell me about 1st Colonial Grill',
  'Best restaurants in Legazpi City overall',
  'Where is Jollibee?',
  'Latest review for Starbucks',
]

export default function ChatPanel() {
  const [messages, setMessages] = useState([{
    id: 'welcome', role: 'bot', ts: Date.now(),
    text: "Hi there! 🍽️ I'm Eatsplorer, your guide to dining in Legazpi City. I use real customer review analysis to help you find the perfect restaurant. What are you looking for today?",
  }])
  const [input, setInput]     = useState('')
  const [loading, setLoading] = useState(false)
  const bottomRef  = useRef(null)
  const inputRef   = useRef(null)
  const sessionId  = useRef(`user_${Date.now()}`)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  const send = useCallback(async (text) => {
    const trimmed = text.trim()
    if (!trimmed || loading) return

    setMessages(prev => [...prev, { id: Date.now(), role: 'user', text: trimmed, ts: Date.now() }])
    setInput('')
    setLoading(true)

    try {
      const responses = await sendMessage(trimmed, sessionId.current)

      const valid = responses.filter(r => r.text || r.image || r.custom)

      const merged  = valid.map(r => r.text || '').filter(Boolean).join('\n\n')
      const buttons = valid.flatMap(r => r.buttons || [])
      const cards   = valid.flatMap(r =>
        r.custom?.type === 'restaurant_cards' ? (r.custom.restaurants ?? []) : []
      )
      const customData = valid.find(r => r.custom)?.custom

      if (!merged && buttons.length === 0 && cards.length === 0) return

      // When cards are present, strip the raw ranked list (#1, #2 ...) —
      // the cards already show that info. Keep anything BEFORE the first rank
      // marker as the conversational header from actions.py.
      // If no rank marker at all (e.g. single restaurant), keep all text as-is.
      let displayText = merged
      if (cards.length > 0) {
        const firstRankIdx = merged.search(/^#\d+\s/m)
        displayText = firstRankIdx > 0
          ? merged.slice(0, firstRankIdx).trim()
          : merged.trim()
      }

      setMessages(prev => [...prev, {
        id:      `bot_${Date.now()}`,
        role:    'bot',
        text:    displayText || undefined,
        buttons: buttons.length > 0 ? buttons : undefined,
        cards:   cards.length   > 0 ? cards   : undefined,
        custom:  customData,
        ts:      Date.now(),
      }])
    } catch (err) {
      setMessages(prev => [...prev, {
        id: `err_${Date.now()}`,
        role: 'bot',
        text: `⚠️ ${err.message}`,
        isError: true,
        ts: Date.now(),
      }])
    } finally {
      setLoading(false)
      inputRef.current?.focus()
    }
  }, [loading])

  const handleKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(input) }
  }

  return (
    <div className="chat-panel">
      <div className="chat-header">
        <div className="chat-avatar">🤖</div>
        <div>
          <div className="chat-title">Eatsplorer AI</div>
          <div className="chat-status">
            <span className={`status-dot ${loading ? 'thinking' : 'online'}`} />
            {loading ? 'Thinking...' : 'Online'}
          </div>
        </div>
        <button className="chat-clear" onClick={() => {
          setMessages([{
            id: 'welcome', role: 'bot', ts: Date.now(),
            text: "Hi! I'm Eatsplorer. What dining experience are you looking for in Legazpi City? 🍽️",
          }])
          sessionId.current = `user_${Date.now()}`
        }} title="Clear conversation">↺</button>
      </div>

      <div className="chat-messages">
        {messages.map(msg => (
          <MessageBubble key={msg.id} message={msg}
            onButtonClick={btn => send(btn.payload || btn.title)} />
        ))}
        {loading && (
          <div className="message bot-message">
            <div className="bot-avatar">🍽️</div>
            <div className="bubble bubble-bot typing-bubble">
              <span className="dot"/><span className="dot"/><span className="dot"/>
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {messages.length <= 2 && (
        <div className="suggestions">
          <p className="suggestions-label">Try asking:</p>
          <div className="suggestions-list">
            {SUGGESTIONS.map(s => (
              <button key={s} className="suggestion-chip" onClick={() => send(s)}>{s}</button>
            ))}
          </div>
        </div>
      )}

      <div className="chat-input-area">
        <textarea ref={inputRef} className="chat-input"
          placeholder="Ask about restaurants in Legazpi City..."
          value={input} onChange={e => setInput(e.target.value)}
          onKeyDown={handleKey} rows={1} disabled={loading}
        />
        <button className="send-btn" onClick={() => send(input)}
          disabled={!input.trim() || loading} aria-label="Send">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
            <line x1="22" y1="2" x2="11" y2="13"/>
            <polygon points="22 2 15 22 11 13 2 9 22 2"/>
          </svg>
        </button>
      </div>
    </div>
  )
}
