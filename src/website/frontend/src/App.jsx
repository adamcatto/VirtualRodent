import React, { useState, useEffect, useRef } from 'react'
import axios from 'axios'
import MetricsDashboard from './components/MetricsDashboard'
import CombinedViewer from './components/CombinedViewer'
import DataExplorer from './components/DataExplorer'
import SessionList from './components/SessionList'
import './styles/App.css'

// Use relative URLs for API calls (Vite proxy handles /api -> localhost:5000)
const API_BASE = ''

// Duration options - samples at 50Hz
// Backend caps at 200 samples to keep JSON payload under 50MB
// Pose horizon is subsampled 5x (250->50 timesteps) on backend
const DURATION_OPTIONS = [
  { label: '2 seconds', value: 2, samples: 100 },
  { label: '4 seconds (max)', value: 4, samples: 200 },
]

function App() {
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
  const [dataDuration, setDataDuration] = useState(4)  // Default 4 seconds (max supported)
  
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

  const loadSession = (experimentName, sessionId, duration = dataDuration) => {
    // Cancel any previous request
    cancelLoading()
    
    setSelectedSession(sessionId)
    setLoading(true)
    setPredictionsLoading(true)
    setLoadingProgress(0)
    setLoadingError(null)
    setPredictions(null)

    // Load session metrics
    axios.get(`${API_BASE}/api/experiment/${experimentName}/session/${sessionId}/metrics`)
      .then(response => {
        setMetrics(response.data)
        setLoading(false)
      })
      .catch(error => {
        console.error('Error loading session metrics:', error)
        setLoading(false)
      })

    // Calculate samples from duration (50Hz) - backend caps at 1000
    const samples = Math.min(duration * 50, 1000)

    // Estimate loading time based on data size (more realistic progress)
    const estimatedLoadTimeMs = Math.max(3000, samples * 5)  // ~5ms per sample
    const progressStep = 95 / (estimatedLoadTimeMs / 200)  // Update every 200ms, cap at 95%

    // Simulate loading progress
    progressIntervalRef.current = setInterval(() => {
      setLoadingProgress(prev => {
        if (prev >= 95) {
          return 95
        }
        return Math.min(95, prev + progressStep)
      })
    }, 200)

    // Create abort controller for this request
    abortControllerRef.current = new AbortController()

    // Load session predictions with duration and timeout
    axios.get(`${API_BASE}/api/experiment/${experimentName}/session/${sessionId}/predictions`, {
      params: { limit: samples },
      timeout: 120000,  // 2 minute timeout for large requests
      signal: abortControllerRef.current.signal,
    })
      .then(response => {
        if (progressIntervalRef.current) {
          clearInterval(progressIntervalRef.current)
          progressIntervalRef.current = null
        }
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
        if (progressIntervalRef.current) {
          clearInterval(progressIntervalRef.current)
          progressIntervalRef.current = null
        }
        
        // Don't show error for cancelled requests
        if (axios.isCancel(error) || error.name === 'CanceledError') {
          console.log('Request cancelled')
          return
        }
        
        setLoadingProgress(0)
        setPredictionsLoading(false)
        
        // Set user-friendly error message
        let errorMsg = 'Failed to load predictions'
        if (error.code === 'ECONNABORTED') {
          errorMsg = 'Request timed out - try a shorter duration'
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

  // Reload predictions with new duration
  const reloadPredictions = (newDuration) => {
    setDataDuration(newDuration)
    setLoadingError(null)  // Clear any previous error
    if (selectedExperiment && selectedSession) {
      loadSession(selectedExperiment.name, selectedSession, newDuration)
    }
  }

  return (
    <div className="app">
      <header>
        <h1>VirtualRodent - Neural Decoding Visualization</h1>
        <p>Pose prediction from neural signals</p>
      </header>

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

                    {/* Data Controls */}
                    <div className="data-controls">
                      <label>
                        Data Duration:
                        <select
                          value={dataDuration}
                          onChange={(e) => reloadPredictions(parseInt(e.target.value))}
                        >
                          {DURATION_OPTIONS.map(opt => (
                            <option key={opt.value} value={opt.value}>
                              {opt.label} (~{opt.samples} samples)
                            </option>
                          ))}
                        </select>
                      </label>
                      {predictionsLoading && (
                        <button 
                          className="cancel-button"
                          onClick={cancelLoading}
                        >
                          Cancel
                        </button>
                      )}
                      {predictions && !predictionsLoading && (
                        <span className="data-info">
                          Loaded: {predictions.predictions?.length || 0} / {predictions.num_total_samples} samples
                        </span>
                      )}
                    </div>

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
                          <p className="loading-text">{Math.round(loadingProgress)}% - Fetching {dataDuration}s of data</p>
                          <p className="loading-hint">Large datasets may take up to 2 minutes to load</p>
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
    </div>
  )
}

export default App
