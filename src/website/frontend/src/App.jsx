import React, { useState, useEffect, useRef } from 'react'
import axios from 'axios'
import MetricsDashboard from './components/MetricsDashboard'
import CombinedViewer from './components/CombinedViewer'
import DataExplorer from './components/DataExplorer'
import SessionList from './components/SessionList'
import RawPoseViewer from './components/RawPoseViewer'
import './styles/App.css'

// Use relative URLs for API calls (Vite proxy handles /api -> localhost:5000)
const API_BASE = ''

// Sampling rate
const SAMPLING_RATE_HZ = 50

function App() {
  const [viewMode, setViewMode] = useState('pose')  // 'pose' | 'neural'
  const [experiments, setExperiments] = useState([])
  const [selectedExperiment, setSelectedExperiment] = useState(null)
  const [selectedSession, setSelectedSession] = useState(null)
  const [metrics, setMetrics] = useState(null)
  const [predictions, setPredictions] = useState(null)
  const [loading, setLoading] = useState(false)
  const [predictionsLoading, setPredictionsLoading] = useState(false)
  const [loadingProgress, setLoadingProgress] = useState(0)
  const [loadingError, setLoadingError] = useState(null)  // Error message for failed loads
  const [activeTab, setActiveTab] = useState('comparison')  // 'comparison' | 'explorer'
  const [sessionInfo, setSessionInfo] = useState(null)  // Info about available data
  const [requestedSamples, setRequestedSamples] = useState('')  // User-specified sample count
  const [loadedBytes, setLoadedBytes] = useState(0)  // Track bytes loaded for progress
  
  // Ref to store abort controller for cancelling requests
  const abortControllerRef = useRef(null)
  const progressIntervalRef = useRef(null)

  useEffect(() => {
    // Load experiments on mount
    axios.get(`${API_BASE}/api/experiments`)
      .then(response => setExperiments(response.data))
      .catch(error => console.error('Error loading experiments:', error))
  }, [])

  const loadExperiment = (exp) => {
    setSelectedExperiment(exp)
    setSelectedSession(null)
    setPredictions(null)
    setLoading(true)

    // Load metrics
    axios.get(`${API_BASE}/api/experiment/${exp.name}/metrics`)
      .then(response => {
        setMetrics(response.data)
        setLoading(false)
      })
      .catch(error => {
        console.error('Error loading metrics:', error)
        setLoading(false)
      })

    // Load predictions (only for single-experiment or if not multi-session)
    if (!exp.is_multi_session) {
      axios.get(`${API_BASE}/api/experiment/${exp.name}/predictions`)
        .then(response => setPredictions(response.data))
        .catch(error => console.error('Error loading predictions:', error))
    }
  }

  // Cancel any ongoing request
  const cancelLoading = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
      abortControllerRef.current = null
    }
    if (progressIntervalRef.current) {
      clearInterval(progressIntervalRef.current)
      progressIntervalRef.current = null
    }
    setPredictionsLoading(false)
    setLoadingProgress(0)
  }

  // Load session info (without loading predictions yet)
  const loadSession = (experimentName, sessionId) => {
    // Cancel any previous request
    cancelLoading()
    
    setSelectedSession(sessionId)
    setLoading(true)
    setPredictionsLoading(false)
    setLoadingProgress(0)
    setLoadingError(null)
    setPredictions(null)
    setSessionInfo(null)
    setRequestedSamples('')
    setLoadedBytes(0)

    // Load session metrics
    axios.get(`${API_BASE}/api/experiment/${experimentName}/session/${sessionId}/metrics`)
      .then(response => {
        setMetrics(response.data)
      })
      .catch(error => {
        console.error('Error loading session metrics:', error)
      })

    // Load session info (total samples available)
    axios.get(`${API_BASE}/api/experiment/${experimentName}/session/${sessionId}/info`)
      .then(response => {
        setSessionInfo(response.data)
        // Default to 200 samples or total if less
        setRequestedSamples(Math.min(200, response.data.num_samples).toString())
        setLoading(false)
      })
      .catch(error => {
        console.error('Error loading session info:', error)
        setLoading(false)
        setLoadingError('Failed to load session info')
      })
  }

  // Start loading predictions with user-specified sample count
  const startLoadingPredictions = () => {
    if (!selectedExperiment || !selectedSession || !sessionInfo) return
    
    const samples = parseInt(requestedSamples) || 200
    if (samples < 1) return
    
    setPredictionsLoading(true)
    setLoadingProgress(0)
    setLoadingError(null)
    setLoadedBytes(0)

    // Estimate total bytes based on sample count
    // ~120KB per sample (based on our testing: 24MB for 200 samples)
    const estimatedTotalBytes = samples * 120 * 1024

    // Create abort controller for this request
    abortControllerRef.current = new AbortController()

    // Load session predictions with progress tracking
    axios.get(`${API_BASE}/api/experiment/${selectedExperiment.name}/session/${selectedSession}/predictions`, {
      params: { limit: samples },
      timeout: 300000,  // 5 minute timeout for large requests
      signal: abortControllerRef.current.signal,
      onDownloadProgress: (progressEvent) => {
        const loaded = progressEvent.loaded
        setLoadedBytes(loaded)
        if (progressEvent.total) {
          setLoadingProgress(Math.round((loaded / progressEvent.total) * 100))
        } else {
          // Estimate progress based on expected size
          const estimatedProgress = Math.min(95, Math.round((loaded / estimatedTotalBytes) * 100))
          setLoadingProgress(estimatedProgress)
        }
      },
    })
      .then(response => {
        setLoadingProgress(100)
        console.log('Predictions API response:', {
          status: response.status,
          hasPredictions: !!response.data?.predictions,
          predictionsLength: response.data?.predictions?.length,
          hasGroundTruth: !!response.data?.ground_truth,
          groundTruthLength: response.data?.ground_truth?.length,
          hasNeural: !!response.data?.neural,
          neuralLength: response.data?.neural?.length,
          numTotalSamples: response.data?.num_total_samples,
          limitApplied: response.data?.limit_applied,
        })
        setPredictions(response.data)
        setPredictionsLoading(false)
      })
      .catch(error => {
        // Don't show error for cancelled requests
        if (axios.isCancel(error) || error.name === 'CanceledError') {
          console.log('Request cancelled')
          setPredictionsLoading(false)
          return
        }
        
        setLoadingProgress(0)
        setPredictionsLoading(false)
        
        // Set user-friendly error message
        let errorMsg = 'Failed to load predictions'
        if (error.code === 'ECONNABORTED') {
          errorMsg = 'Request timed out - try fewer samples'
        } else if (error.response?.status === 404) {
          errorMsg = 'Predictions file not found - run test evaluation first'
        } else if (error.response?.data?.error) {
          errorMsg = error.response.data.error
        } else if (error.message) {
          errorMsg = error.message
        }
        setLoadingError(errorMsg)
        
        console.error('Error loading session predictions:', error)
        console.error('Error details:', error.response?.data)
      })
  }

  // Format bytes for display
  const formatBytes = (bytes) => {
    if (bytes < 1024) return `${bytes} B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
    return `${(bytes / 1024 / 1024).toFixed(1)} MB`
  }

  return (
    <div className="app">
      <header>
        <h1>VirtualRodent</h1>
        <nav className="main-nav">
          <button 
            className={`nav-button ${viewMode === 'pose' ? 'active' : ''}`}
            onClick={() => setViewMode('pose')}
          >
            🐭 Pose Viewer
          </button>
          <button 
            className={`nav-button ${viewMode === 'neural' ? 'active' : ''}`}
            onClick={() => setViewMode('neural')}
          >
            🧠 Neural Decoding
          </button>
        </nav>
      </header>

      {/* Raw Pose Viewer Mode */}
      {viewMode === 'pose' && <RawPoseViewer />}

      {/* Neural Decoding Mode */}
      {viewMode === 'neural' && (
      <div className="main-content">
        <aside className="sidebar">
          <h2>Experiments</h2>
          {experiments.map(exp => (
            <div
              key={exp.name}
              className={`experiment-item ${selectedExperiment?.name === exp.name ? 'active' : ''}`}
              onClick={() => loadExperiment(exp)}
            >
              <h3>{exp.name}</h3>
              <p>Strategy: {exp.strategy}</p>
              <p>Model: {exp.model_type}</p>
              {exp.is_multi_session ? (
                <>
                  <p>Sessions: {exp.successful_sessions}/{exp.total_sessions}</p>
                  <p>Avg Val Loss: {exp.avg_val_loss?.toFixed(4)}</p>
                </>
              ) : (
                <p>Val Loss: {exp.best_val_loss?.toFixed(4)}</p>
              )}
              {exp.session_id && <p className="session-tag">Session: {exp.session_id}</p>}
            </div>
          ))}
        </aside>

        <main>
          {loading ? (
            <div className="placeholder">
              <h2>Loading...</h2>
            </div>
          ) : selectedExperiment ? (
            <>
              {/* For multi-session experiments, show session list */}
              {selectedExperiment.is_multi_session && !selectedSession && (
                <section>
                  <h2>Per-Session Results</h2>
                  <p>Select a session to view details:</p>
                  <SessionList
                    sessions={selectedExperiment.sessions}
                    onSelectSession={(sessionId) => loadSession(selectedExperiment.name, sessionId)}
                    selectedSession={selectedSession}
                  />
                </section>
              )}

              {/* Session or experiment details */}
              {(selectedSession || !selectedExperiment.is_multi_session) && (
                <>
                  {selectedSession && (
                    <div className="session-header">
                      <button onClick={() => {
                        setSelectedSession(null)
                        setPredictions(null)
                        loadExperiment(selectedExperiment)
                      }}>
                        &larr; Back to Sessions
                      </button>
                      <h2>Session: {selectedSession}</h2>
                    </div>
                  )}

                  <section>
                    <h2>Metrics Dashboard</h2>
                    {metrics && <MetricsDashboard metrics={metrics} />}
                  </section>

                  <section>
                    <h2>Neural Decoding Visualization</h2>

                    {/* Session Info & Load Controls */}
                    {sessionInfo && !predictions && !predictionsLoading && (
                      <div className="preload-controls">
                        <div className="session-info-panel">
                          <h4>Session Data Available</h4>
                          <div className="info-row">
                            <span className="info-label">Total Samples:</span>
                            <span className="info-value">{sessionInfo.num_samples.toLocaleString()}</span>
                          </div>
                          <div className="info-row">
                            <span className="info-label">Total Duration:</span>
                            <span className="info-value">{sessionInfo.total_time_seconds.toFixed(1)}s ({(sessionInfo.total_time_seconds / 60).toFixed(1)} min)</span>
                          </div>
                          <div className="info-row">
                            <span className="info-label">Pose Horizon:</span>
                            <span className="info-value">{sessionInfo.pose_horizon} timesteps ({(sessionInfo.pose_horizon / SAMPLING_RATE_HZ).toFixed(1)}s)</span>
                          </div>
                          {sessionInfo.num_neurons && (
                            <div className="info-row">
                              <span className="info-label">Neurons:</span>
                              <span className="info-value">{sessionInfo.num_neurons}</span>
                            </div>
                          )}
                        </div>
                        
                        <div className="load-config">
                          <label>
                            <span>Samples to load:</span>
                            <input
                              type="number"
                              min="1"
                              max={sessionInfo.num_samples}
                              value={requestedSamples}
                              onChange={(e) => setRequestedSamples(e.target.value)}
                              placeholder="e.g., 200"
                            />
                            <span className="input-hint">
                              = {((parseInt(requestedSamples) || 0) / SAMPLING_RATE_HZ).toFixed(1)}s
                              {parseInt(requestedSamples) > 500 && (
                                <span className="warning"> (large request, may be slow)</span>
                              )}
                            </span>
                          </label>
                          <button 
                            className="load-button"
                            onClick={startLoadingPredictions}
                            disabled={!requestedSamples || parseInt(requestedSamples) < 1}
                          >
                            Load Data
                          </button>
                        </div>
                      </div>
                    )}

                    {/* Data Controls - shown when data is loaded */}
                    {predictions && !predictionsLoading && (
                      <div className="data-controls">
                        <span className="data-info">
                          Loaded: {predictions.predictions?.length || 0} / {predictions.num_total_samples} samples
                          ({((predictions.predictions?.length || 0) / SAMPLING_RATE_HZ).toFixed(1)}s of {(predictions.num_total_samples / SAMPLING_RATE_HZ).toFixed(1)}s)
                        </span>
                        <button 
                          className="reload-button"
                          onClick={() => {
                            setPredictions(null)
                            setLoadingError(null)
                          }}
                        >
                          Load Different Amount
                        </button>
                      </div>
                    )}

                    {/* Tab Navigation */}
                    <div className="tab-navigation">
                      <button
                        className={`tab-button ${activeTab === 'comparison' ? 'active' : ''}`}
                        onClick={() => setActiveTab('comparison')}
                      >
                        Prediction vs Ground Truth
                      </button>
                      <button
                        className={`tab-button ${activeTab === 'explorer' ? 'active' : ''}`}
                        onClick={() => setActiveTab('explorer')}
                      >
                        Data Explorer
                      </button>
                    </div>

                    {/* Loading Progress */}
                    {predictionsLoading && (
                      <div className="loading-container">
                        <div className="loading-content">
                          <h3>Loading visualization data...</h3>
                          <div className="progress-bar">
                            <div
                              className="progress-fill"
                              style={{ width: `${loadingProgress}%` }}
                            />
                          </div>
                          <p className="loading-text">
                            {loadingProgress}% complete
                          </p>
                          <p className="loading-details">
                            Downloaded: {formatBytes(loadedBytes)}
                            {requestedSamples && ` • Requested: ${requestedSamples} samples (${(parseInt(requestedSamples) / SAMPLING_RATE_HZ).toFixed(1)}s)`}
                          </p>
                          <button 
                            className="cancel-button"
                            onClick={cancelLoading}
                          >
                            Cancel
                          </button>
                        </div>
                      </div>
                    )}

                    {/* Error Message */}
                    {loadingError && !predictionsLoading && (
                      <div className="error-container">
                        <div className="error-content">
                          <h3>⚠️ Error Loading Data</h3>
                          <p>{loadingError}</p>
                          <button 
                            className="retry-button"
                            onClick={() => loadSession(selectedExperiment.name, selectedSession, dataDuration)}
                          >
                            Retry
                          </button>
                        </div>
                      </div>
                    )}

                    {/* Tab Content */}
                    {!predictionsLoading && !loadingError && (
                      <div className="tab-content">
                        {activeTab === 'comparison' ? (
                          <CombinedViewer
                            predictions={predictions?.predictions}
                            groundTruth={predictions?.ground_truth}
                            neural={predictions?.neural}
                            numTotalSamples={predictions?.num_total_samples}
                            neuralSubsampleFactor={predictions?.neural_time_subsample_factor || 10}
                            poseSubsampleFactor={predictions?.pose_subsample_factor || 1}
                          />
                        ) : (
                          <DataExplorer
                            groundTruth={predictions?.ground_truth}
                            neural={predictions?.neural}
                            numTotalSamples={predictions?.num_total_samples}
                            neuralSubsampleFactor={predictions?.neural_time_subsample_factor || 10}
                            poseSubsampleFactor={predictions?.pose_subsample_factor || 1}
                          />
                        )}
                      </div>
                    )}
                  </section>
                </>
              )}

              {/* Multi-session summary when no specific session selected */}
              {selectedExperiment.is_multi_session && !selectedSession && metrics?.is_multi_session && (
                <section>
                  <h2>Experiment Summary</h2>
                  <div className="metrics-grid">
                    <div className="metric-card">
                      <h3>Total Sessions</h3>
                      <p className="metric-value">{metrics.summary?.total_sessions}</p>
                    </div>
                    <div className="metric-card">
                      <h3>Successful</h3>
                      <p className="metric-value">{metrics.summary?.successful}</p>
                    </div>
                    <div className="metric-card">
                      <h3>Failed</h3>
                      <p className="metric-value">{metrics.summary?.failed}</p>
                    </div>
                  </div>
                </section>
              )}
            </>
          ) : (
            <div className="placeholder">
              <h2>Select an experiment to view results</h2>
              <p>Train models using:</p>
              <code>python scripts/train_per_session.py</code>
            </div>
          )}
        </main>
      </div>
      )}
    </div>
  )
}

export default App
