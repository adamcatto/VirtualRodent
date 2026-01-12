import React, { useState, useEffect, useCallback, useRef } from 'react'
import axios from 'axios'
import PoseViewer3D from './PoseViewer3D'
import MouseMesh from './MouseMesh'

const API_BASE = ''

/**
 * RawPoseViewer - View raw pose data directly from HDF5 files
 * Simple 3D pose playback without neural data or predictions
 */
function RawPoseViewer() {
  const [sessions, setSessions] = useState([])
  const [selectedSession, setSelectedSession] = useState(null)
  const [sessionInfo, setSessionInfo] = useState(null)
  const [poseData, setPoseData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [loadingProgress, setLoadingProgress] = useState(0)
  const [error, setError] = useState(null)
  
  // Playback state
  const [currentFrame, setCurrentFrame] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const [playbackSpeed, setPlaybackSpeed] = useState(1)
  
  // Render mode: 'skeleton' | 'mouse'
  const [renderMode, setRenderMode] = useState('mouse')
  
  // Load settings
  const [startFrame, setStartFrame] = useState(0)
  const [numFrames, setNumFrames] = useState(3000)  // 1 minute at 50Hz
  const [subsample, setSubsample] = useState(1)
  
  const abortControllerRef = useRef(null)

  // Load available sessions on mount
  useEffect(() => {
    axios.get(`${API_BASE}/api/sessions`)
      .then(response => setSessions(response.data))
      .catch(err => {
        console.error('Error loading sessions:', err)
        setError('Failed to load sessions')
      })
  }, [])

  // Select a session and load its info
  const selectSession = (session) => {
    setSelectedSession(session)
    setPoseData(null)
    setCurrentFrame(0)
    setError(null)
    
    // Load session info
    axios.get(`${API_BASE}/api/session/${session.brain_region}/${session.animal}/${session.session_id}/info`)
      .then(response => {
        setSessionInfo(response.data)
        // Default to first minute or total duration if less
        const defaultFrames = Math.min(3000, response.data.num_frames)
        setNumFrames(defaultFrames)
        setStartFrame(0)
      })
      .catch(err => {
        console.error('Error loading session info:', err)
        setError('Failed to load session info')
      })
  }

  // Load pose data
  const loadPoseData = () => {
    if (!selectedSession || !sessionInfo) return
    
    // Cancel any previous request
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
    }
    abortControllerRef.current = new AbortController()
    
    setLoading(true)
    setLoadingProgress(0)
    setError(null)
    setPoseData(null)
    
    const { brain_region, animal, session_id } = selectedSession
    
    axios.get(`${API_BASE}/api/session/${brain_region}/${animal}/${session_id}/pose`, {
      params: {
        start: startFrame,
        limit: numFrames,
        subsample: subsample,
      },
      signal: abortControllerRef.current.signal,
      onDownloadProgress: (progressEvent) => {
        if (progressEvent.total) {
          setLoadingProgress(Math.round((progressEvent.loaded / progressEvent.total) * 100))
        } else {
          // Estimate based on expected size (~100 bytes per frame)
          const estimatedTotal = numFrames * 100
          setLoadingProgress(Math.min(95, Math.round((progressEvent.loaded / estimatedTotal) * 100)))
        }
      },
    })
      .then(response => {
        setLoadingProgress(100)
        setPoseData(response.data)
        setCurrentFrame(0)
        setLoading(false)
      })
      .catch(err => {
        if (axios.isCancel(err)) {
          console.log('Request cancelled')
          return
        }
        console.error('Error loading pose data:', err)
        setError(err.response?.data?.error || 'Failed to load pose data')
        setLoading(false)
      })
  }

  // Cancel loading
  const cancelLoading = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
    }
    setLoading(false)
    setLoadingProgress(0)
  }

  // Playback loop
  useEffect(() => {
    if (!isPlaying || !poseData) return

    const effectiveRate = poseData.effective_sampling_rate_hz || 50
    const interval = setInterval(() => {
      setCurrentFrame(prev => {
        const next = prev + 1
        if (next >= poseData.pose.length) {
          setIsPlaying(false)
          return 0
        }
        return next
      })
    }, 1000 / (effectiveRate * playbackSpeed))

    return () => clearInterval(interval)
  }, [isPlaying, poseData, playbackSpeed])

  // Keyboard controls
  useEffect(() => {
    const handleKeyPress = (e) => {
      if (!poseData) return
      
      switch (e.key) {
        case ' ':
          e.preventDefault()
          setIsPlaying(prev => !prev)
          break
        case 'ArrowLeft':
          setCurrentFrame(prev => Math.max(0, prev - 1))
          break
        case 'ArrowRight':
          setCurrentFrame(prev => Math.min(poseData.pose.length - 1, prev + 1))
          break
        case 'ArrowUp':
          setCurrentFrame(prev => Math.min(poseData.pose.length - 1, prev + 50))
          break
        case 'ArrowDown':
          setCurrentFrame(prev => Math.max(0, prev - 50))
          break
      }
    }

    window.addEventListener('keydown', handleKeyPress)
    return () => window.removeEventListener('keydown', handleKeyPress)
  }, [poseData])

  // Get current pose for rendering
  const currentPose = poseData?.pose?.[currentFrame] || null

  // Format time display
  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60)
    const secs = (seconds % 60).toFixed(1)
    return mins > 0 ? `${mins}m ${secs}s` : `${secs}s`
  }

  return (
    <div className="raw-pose-viewer">
      <div className="viewer-layout">
        {/* Session List Sidebar */}
        <div className="session-sidebar">
          <h3>Available Sessions</h3>
          <div className="session-list-scroll">
            {sessions.map(session => (
              <div
                key={`${session.brain_region}/${session.animal}/${session.session_id}`}
                className={`session-item ${selectedSession?.session_id === session.session_id ? 'selected' : ''}`}
                onClick={() => selectSession(session)}
              >
                <div className="session-name">{session.session_id}</div>
                <div className="session-meta">
                  {session.animal} • {session.brain_region}
                </div>
                <div className="session-duration">
                  {formatTime(session.duration_seconds)}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Main Content */}
        <div className="viewer-main">
          {!selectedSession ? (
            <div className="placeholder-message">
              <h2>Select a session to view pose data</h2>
              <p>Choose a session from the list on the left</p>
            </div>
          ) : !poseData && !loading ? (
            /* Load Configuration */
            <div className="load-config-panel">
              <h3>Session: {selectedSession.session_id}</h3>
              <p className="session-details">
                {selectedSession.animal} • {selectedSession.brain_region} • 
                Total: {formatTime(sessionInfo?.duration_seconds || 0)} ({sessionInfo?.num_frames?.toLocaleString()} frames)
              </p>
              
              <div className="config-form">
                <div className="config-row">
                  <label>
                    Start Frame:
                    <input
                      type="number"
                      min={0}
                      max={sessionInfo?.num_frames - 1 || 0}
                      value={startFrame}
                      onChange={(e) => setStartFrame(parseInt(e.target.value) || 0)}
                    />
                    <span className="hint">= {formatTime(startFrame / 50)}</span>
                  </label>
                </div>
                
                <div className="config-row">
                  <label>
                    Frames to Load:
                    <input
                      type="number"
                      min={1}
                      max={sessionInfo?.num_frames || 10000}
                      value={numFrames}
                      onChange={(e) => setNumFrames(parseInt(e.target.value) || 1000)}
                    />
                    <span className="hint">= {formatTime(numFrames / 50)}</span>
                  </label>
                </div>
                
                <div className="config-row">
                  <label>
                    Subsample (1 = full rate):
                    <input
                      type="number"
                      min={1}
                      max={10}
                      value={subsample}
                      onChange={(e) => setSubsample(parseInt(e.target.value) || 1)}
                    />
                    <span className="hint">= {(50 / subsample).toFixed(0)} Hz</span>
                  </label>
                </div>
                
                <button className="load-button" onClick={loadPoseData}>
                  Load Pose Data
                </button>
              </div>
            </div>
          ) : loading ? (
            /* Loading Progress */
            <div className="loading-panel">
              <h3>Loading pose data...</h3>
              <div className="progress-bar">
                <div className="progress-fill" style={{ width: `${loadingProgress}%` }} />
              </div>
              <p>{loadingProgress}% complete</p>
              <button className="cancel-button" onClick={cancelLoading}>Cancel</button>
            </div>
          ) : error ? (
            /* Error */
            <div className="error-panel">
              <h3>Error</h3>
              <p>{error}</p>
              <button onClick={() => setError(null)}>Try Again</button>
            </div>
          ) : (
            /* Pose Viewer */
            <div className="pose-playback">
              {/* Render Mode Toggle */}
              <div className="render-mode-toggle">
                <button 
                  className={renderMode === 'skeleton' ? 'active' : ''}
                  onClick={() => setRenderMode('skeleton')}
                >
                  🦴 Skeleton
                </button>
                <button 
                  className={renderMode === 'mouse' ? 'active' : ''}
                  onClick={() => setRenderMode('mouse')}
                >
                  🐭 Mouse Mesh
                </button>
              </div>
              
              <div className="pose-3d-container">
                {renderMode === 'skeleton' ? (
                  <PoseViewer3D
                    poseData={currentPose}
                    title={`${selectedSession.session_id} - Frame ${currentFrame + 1}`}
                    color={0x4CAF50}
                  />
                ) : (
                  <MouseMesh
                    poseData={currentPose}
                    title={`${selectedSession.session_id} - Frame ${currentFrame + 1}`}
                  />
                )}
              </div>
              
              {/* Playback Controls */}
              <div className="playback-controls-panel">
                <div className="playback-buttons">
                  <button onClick={() => setCurrentFrame(0)} title="Start">⏮</button>
                  <button onClick={() => setCurrentFrame(prev => Math.max(0, prev - 50))} title="Back 1s">⏪</button>
                  <button onClick={() => setCurrentFrame(prev => Math.max(0, prev - 1))} title="Previous">◀</button>
                  <button onClick={() => setIsPlaying(!isPlaying)} title={isPlaying ? 'Pause' : 'Play'}>
                    {isPlaying ? '⏸' : '▶'}
                  </button>
                  <button onClick={() => setCurrentFrame(prev => Math.min(poseData.pose.length - 1, prev + 1))} title="Next">▶</button>
                  <button onClick={() => setCurrentFrame(prev => Math.min(poseData.pose.length - 1, prev + 50))} title="Forward 1s">⏩</button>
                  <button onClick={() => setCurrentFrame(poseData.pose.length - 1)} title="End">⏭</button>
                </div>
                
                <div className="frame-display">
                  <span>Frame: {currentFrame + 1} / {poseData.pose.length}</span>
                  <span>Time: {formatTime(currentFrame / poseData.effective_sampling_rate_hz)}</span>
                </div>
                
                <div className="timeline-slider">
                  <input
                    type="range"
                    min={0}
                    max={poseData.pose.length - 1}
                    value={currentFrame}
                    onChange={(e) => setCurrentFrame(parseInt(e.target.value))}
                  />
                </div>
                
                <div className="speed-control">
                  <label>
                    Speed:
                    <select value={playbackSpeed} onChange={(e) => setPlaybackSpeed(parseFloat(e.target.value))}>
                      <option value={0.1}>0.1x</option>
                      <option value={0.25}>0.25x</option>
                      <option value={0.5}>0.5x</option>
                      <option value={1}>1x</option>
                      <option value={2}>2x</option>
                      <option value={4}>4x</option>
                    </select>
                  </label>
                  <button 
                    className="reload-button"
                    onClick={() => {
                      setPoseData(null)
                      setCurrentFrame(0)
                    }}
                  >
                    Load Different Range
                  </button>
                </div>
              </div>
              
              {/* Data Info */}
              <div className="data-info-bar">
                <span>Loaded: {poseData.num_frames_loaded} frames ({formatTime(poseData.duration_loaded_seconds)})</span>
                <span>Rate: {poseData.effective_sampling_rate_hz} Hz</span>
                <span>Session Total: {formatTime(poseData.total_duration_seconds)}</span>
              </div>
              
              <div className="keyboard-help">
                <small>Keyboard: Space (play/pause), ←/→ (±1 frame), ↑/↓ (±1 second)</small>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default RawPoseViewer

