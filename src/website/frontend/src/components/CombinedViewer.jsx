import React, { useState, useEffect, useCallback } from 'react'
import PoseViewer3D from './PoseViewer3D'
import NeuralSignalViewer from './NeuralSignalViewer'

function CombinedViewer({ predictions, groundTruth, neural, numTotalSamples, neuralSubsampleFactor = 10, poseSubsampleFactor = 1 }) {
  const [currentFrame, setCurrentFrame] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const [playbackSpeed, setPlaybackSpeed] = useState(1)
  const [showPredictions, setShowPredictions] = useState(true)
  const [selectedTimeStep, setSelectedTimeStep] = useState(0)  // Within prediction horizon
  const [showDiagnostics, setShowDiagnostics] = useState(false)  // Toggle diagnostic info

  const numFrames = predictions?.length || groundTruth?.length || 0
  const poseHorizon = predictions?.[0]?.length || groundTruth?.[0]?.length || 250  // Default 5 seconds

  // Debug logging
  useEffect(() => {
    console.log('CombinedViewer received data:', {
      hasPredictions: !!predictions,
      predictionsLength: predictions?.length,
      predictionsShape: predictions ? `[${predictions.length}, ${predictions[0]?.length}, ${predictions[0]?.[0]?.length}]` : null,
      hasGroundTruth: !!groundTruth,
      groundTruthLength: groundTruth?.length,
      groundTruthShape: groundTruth ? `[${groundTruth.length}, ${groundTruth[0]?.length}, ${groundTruth[0]?.[0]?.length}]` : null,
      hasNeural: !!neural,
      neuralLength: neural?.length,
      neuralShape: neural ? `[${neural.length}, ${neural[0]?.length}, ${neural[0]?.[0]?.length}]` : null,
      numTotalSamples,
      numFrames,
      poseHorizon
    })
  }, [predictions, groundTruth, neural, numTotalSamples, numFrames, poseHorizon])

  // Playback loop
  useEffect(() => {
    if (!isPlaying || numFrames === 0) return

    const interval = setInterval(() => {
      setCurrentFrame(prev => {
        const next = prev + 1
        if (next >= numFrames) {
          setIsPlaying(false)
          return 0
        }
        return next
      })
    }, 1000 / (50 * playbackSpeed))  // 50Hz base rate

    return () => clearInterval(interval)
  }, [isPlaying, numFrames, playbackSpeed])

  // Get current pose data
  const getCurrentPose = useCallback((data, frame, timeStep) => {
    if (!data || !data[frame]) return null
    // data[frame] is shape [horizon, 69] - get specific time step
    const horizon = data[frame]
    if (!horizon || !horizon[timeStep]) return null
    return horizon[timeStep]
  }, [])

  const currentGroundTruth = getCurrentPose(groundTruth, currentFrame, selectedTimeStep)
  const currentPrediction = getCurrentPose(predictions, currentFrame, selectedTimeStep)
  const currentNeural = neural?.[currentFrame] || null

  // Debug current frame data - track pose changes
  useEffect(() => {
    // Check if pose is changing by comparing first few coordinates
    const poseStats = currentGroundTruth ? {
      x0: currentGroundTruth[0]?.toFixed(2),
      y0: currentGroundTruth[23]?.toFixed(2),
      z0: currentGroundTruth[46]?.toFixed(2),
      sum: currentGroundTruth.reduce((a, b) => a + b, 0).toFixed(2)
    } : null

    console.log('Current frame data:', {
      currentFrame,
      selectedTimeStep,
      currentGroundTruth: currentGroundTruth ? `[${currentGroundTruth.length}]` : null,
      currentPrediction: currentPrediction ? `[${currentPrediction.length}]` : null,
      currentNeural: currentNeural ? `[${currentNeural.length}, ${currentNeural[0]?.length}]` : null,
      poseStats,  // This helps verify poses are changing between frames
    })
  }, [currentFrame, selectedTimeStep, currentGroundTruth, currentPrediction, currentNeural])

  // Keyboard controls
  useEffect(() => {
    const handleKeyPress = (e) => {
      switch (e.key) {
        case ' ':
          e.preventDefault()
          setIsPlaying(prev => !prev)
          break
        case 'ArrowLeft':
          setCurrentFrame(prev => Math.max(0, prev - 1))
          break
        case 'ArrowRight':
          setCurrentFrame(prev => Math.min(numFrames - 1, prev + 1))
          break
        case 'ArrowUp':
          setSelectedTimeStep(prev => Math.min(poseHorizon - 1, prev + 10))
          break
        case 'ArrowDown':
          setSelectedTimeStep(prev => Math.max(0, prev - 10))
          break
      }
    }

    window.addEventListener('keydown', handleKeyPress)
    return () => window.removeEventListener('keydown', handleKeyPress)
  }, [numFrames, poseHorizon])

  if (numFrames === 0) {
    return (
      <div className="combined-viewer placeholder">
        <h3>No prediction data available</h3>
        <p>Run training with test evaluation to generate predictions.</p>
      </div>
    )
  }

  return (
    <div className="combined-viewer">
      {/* Neural Signal Panel */}
      {currentNeural && (
        <div className="neural-panel">
          <NeuralSignalViewer
            neuralData={currentNeural}
            currentFrame={selectedTimeStep}
            numNeurons={50}
            subsampleFactor={neuralSubsampleFactor}
          />
        </div>
      )}

      {/* Pose Visualization Panels */}
      <div className="pose-panels">
        <div className="pose-panel ground-truth">
          <PoseViewer3D
            poseData={currentGroundTruth}
            title="Ground Truth"
            color={0x4CAF50}
          />
        </div>

        {showPredictions && (
          <div className="pose-panel prediction">
            <PoseViewer3D
              poseData={currentPrediction}
              title="Prediction"
              color={0x2196F3}
            />
          </div>
        )}
      </div>

      {/* Controls */}
      <div className="viewer-controls">
        <div className="playback-controls">
          <button onClick={() => setCurrentFrame(0)} title="Reset">
            ⏮
          </button>
          <button onClick={() => setCurrentFrame(prev => Math.max(0, prev - 1))} title="Previous">
            ⏪
          </button>
          <button onClick={() => setIsPlaying(!isPlaying)} title={isPlaying ? 'Pause' : 'Play'}>
            {isPlaying ? '⏸' : '▶'}
          </button>
          <button onClick={() => setCurrentFrame(prev => Math.min(numFrames - 1, prev + 1))} title="Next">
            ⏩
          </button>
          <button onClick={() => setCurrentFrame(numFrames - 1)} title="End">
            ⏭
          </button>
        </div>

        <div className="frame-info">
          <span>Sample: {currentFrame + 1} / {numFrames}</span>
          <span>Time Step: {selectedTimeStep + 1} / {poseHorizon}</span>
          {numTotalSamples && numTotalSamples > numFrames && (
            <span className="total-info">(Total: {numTotalSamples})</span>
          )}
        </div>

        <div className="slider-controls">
          <label>
            Sample:
            <input
              type="range"
              min={0}
              max={numFrames - 1}
              value={currentFrame}
              onChange={(e) => setCurrentFrame(parseInt(e.target.value))}
            />
          </label>

          <label>
            Time Step (within horizon):
            <input
              type="range"
              min={0}
              max={poseHorizon - 1}
              value={selectedTimeStep}
              onChange={(e) => setSelectedTimeStep(parseInt(e.target.value))}
            />
          </label>

          <label>
            Speed:
            <select value={playbackSpeed} onChange={(e) => setPlaybackSpeed(parseFloat(e.target.value))}>
              <option value={0.25}>0.25x</option>
              <option value={0.5}>0.5x</option>
              <option value={1}>1x</option>
              <option value={2}>2x</option>
              <option value={4}>4x</option>
            </select>
          </label>
        </div>

        <div className="toggle-controls">
          <label>
            <input
              type="checkbox"
              checked={showPredictions}
              onChange={(e) => setShowPredictions(e.target.checked)}
            />
            Show Predictions
          </label>
          <label>
            <input
              type="checkbox"
              checked={showDiagnostics}
              onChange={(e) => setShowDiagnostics(e.target.checked)}
            />
            Show Diagnostics
          </label>
        </div>
      </div>

      {/* Diagnostics Panel - helps verify data is changing between frames */}
      {showDiagnostics && currentGroundTruth && (
        <div className="diagnostics-panel">
          <h4>Pose Diagnostics (Ground Truth)</h4>
          <div className="diagnostics-grid">
            <div className="diag-item">
              <span className="diag-label">Nose X:</span>
              <span className="diag-value">{currentGroundTruth[0]?.toFixed(3)}</span>
            </div>
            <div className="diag-item">
              <span className="diag-label">Nose Y:</span>
              <span className="diag-value">{currentGroundTruth[23]?.toFixed(3)}</span>
            </div>
            <div className="diag-item">
              <span className="diag-label">Nose Z:</span>
              <span className="diag-value">{currentGroundTruth[46]?.toFixed(3)}</span>
            </div>
            <div className="diag-item">
              <span className="diag-label">Spine X:</span>
              <span className="diag-value">{currentGroundTruth[3]?.toFixed(3)}</span>
            </div>
            <div className="diag-item">
              <span className="diag-label">Pose Sum:</span>
              <span className="diag-value">{currentGroundTruth.reduce((a, b) => a + b, 0).toFixed(2)}</span>
            </div>
            <div className="diag-item">
              <span className="diag-label">Time (s):</span>
              <span className="diag-value">{((currentFrame + selectedTimeStep) / 50).toFixed(2)}</span>
            </div>
          </div>
          <p className="diagnostics-help">
            <small>If these values change as frames advance, pose data is synced correctly.</small>
          </p>
        </div>
      )}

      {/* Keyboard shortcuts help */}
      <div className="keyboard-help">
        <small>
          Keyboard: Space (play/pause), ←/→ (prev/next sample), ↑/↓ (change time step)
        </small>
      </div>
    </div>
  )
}

export default CombinedViewer
