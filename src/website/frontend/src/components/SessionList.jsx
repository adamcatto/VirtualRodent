import React from 'react'

function SessionList({ sessions, onSelectSession, selectedSession }) {
  if (!sessions || sessions.length === 0) {
    return <div className="placeholder">No sessions available</div>
  }

  // Sort sessions by validation loss (best first)
  const sortedSessions = [...sessions].sort((a, b) => {
    const aLoss = a.best_val_loss ?? Infinity
    const bLoss = b.best_val_loss ?? Infinity
    return aLoss - bLoss
  })

  return (
    <div className="session-list">
      <table className="session-table">
        <thead>
          <tr>
            <th>Session ID</th>
            <th>Animal</th>
            <th>Brain Region</th>
            <th>Neurons</th>
            <th>Val Loss</th>
            <th>Test Loss</th>
            <th>R2</th>
          </tr>
        </thead>
        <tbody>
          {sortedSessions.map(session => {
            const testMetrics = session.test_metrics || {}
            const metadata = session.session_metadata || {}

            return (
              <tr
                key={session.session_id}
                className={`session-row ${selectedSession === session.session_id ? 'selected' : ''}`}
                onClick={() => onSelectSession(session.session_id)}
              >
                <td className="session-id">{session.session_id}</td>
                <td>{metadata.animal || '-'}</td>
                <td>{metadata.brain_region || '-'}</td>
                <td>{metadata.num_neurons || '-'}</td>
                <td>{session.best_val_loss?.toFixed(4) || '-'}</td>
                <td>{testMetrics['test/loss']?.toFixed(4) || '-'}</td>
                <td>{testMetrics['test/r2']?.toFixed(4) || '-'}</td>
              </tr>
            )
          })}
        </tbody>
      </table>

      <div className="session-stats">
        <p>Total sessions: {sessions.length}</p>
        <p>
          Avg Val Loss: {
            (sessions.reduce((sum, s) => sum + (s.best_val_loss || 0), 0) / sessions.length).toFixed(4)
          }
        </p>
      </div>
    </div>
  )
}

export default SessionList
