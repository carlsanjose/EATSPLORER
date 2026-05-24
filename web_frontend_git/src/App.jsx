import { useState } from 'react'
import ChatPanel from './components/ChatPanel'
import RestaurantBrowser from './components/RestaurantBrowser'
import './App.css'

export default function App() {
  const [browserOpen, setBrowserOpen] = useState(true)
  const [activeTab, setActiveTab] = useState('chat')

  return (
     <div className="app">
      {/* ── Header ── */}
      <header className="app-header">
        <div className="header-brand">
            <img className="header-icon" src="/eatsplorer_icon_logo.png" alt="Eatsplorer Logo" />
          <div>
            <img className="header-logo" src="/eatsplorer_long_logo.png" alt="Eatsplorer Long Logo" />
            <p className="header-sub">Legazpi City Dining Discovery</p>
          </div>
        </div>


        {/* Desktop toggle button */}
        <button
          className="browser-toggle-btn"
          onClick={() => setBrowserOpen(v => !v)}
          title={browserOpen ? 'Hide restaurant browser' : 'Show restaurant browser'}
        >
          {browserOpen ? '◀ Hide browser' : '▶ Browse restaurants'}
        </button>

        {/* Mobile tabs */}
        <div className="mobile-tabs">
          <button
            className={`mobile-tab ${activeTab === 'chat' ? 'active' : ''}`}
            onClick={() => setActiveTab('chat')}
          >💬 Chat</button>
          <button
            className={`mobile-tab ${activeTab === 'browse' ? 'active' : ''}`}
            onClick={() => setActiveTab('browse')}
          >🔍 Browse</button>
        </div>
      </header>

      <main className="app-main">
        <div className={`panel panel-browser ${browserOpen ? 'browser-open' : 'browser-closed'} ${activeTab === 'browse' ? 'mobile-visible' : 'mobile-hidden'}`}>
          <RestaurantBrowser />
        </div>
        <div className={`panel panel-chat ${activeTab === 'chat' ? 'mobile-visible' : 'mobile-hidden'}`}>
          <ChatPanel />
        </div>
      </main>
    </div>
  )
}
