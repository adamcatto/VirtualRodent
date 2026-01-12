import React, { useState, useEffect, useCallback } from 'react'
import PoseViewer3D from './PoseViewer3D'
import NeuralSignalViewer from './NeuralSignalViewer'

/**
 * DataExplorer - Ground truth only visualization
 * Shows neural activity synchronized with ground truth pose playback
 * No predictions, just raw data exploration
 */
function DataExplorer({ groundTruth, neural, numTotalSamples, neuralSubsampleFactor = 10, poseSubsampleFactor = 1 }) {
  const [currentFrame, setCurrentFrame] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const [playbackSpeed, setPlaybackSpeed] = useState(1)
  const [selectedTimeStep, setSelectedTimeStep] = useState(0)  // Within pose horizon

  const numFrames = groundTruth?.length || 0
  const poseHorizonDisplayed = groundTruth?.[0]?.length || 50  // Subsampled timesteps shown
  const poseHorizonOriginal = poseHorizonDisplayed * poseSubsampleFactor  // Original 250 timesteps = 5s
  const neuralHistoryDisplayed = neural?.[0]?.length || 75  // Subsampled timesteps shown
  const neuralHistoryOriginal = neuralHistoryDisplayed * neuralSubsampleFactor  // Original 750 timesteps = 15s
  const numNeurons = neural?.[0]?.[0]?.length || 0

  // Debug logging
  useEffect(() => {
    console.log('DataExplorer received data:', {
      hasGroundTruth: !!groundTruth,
      groundTruthLength: groundTruth?.length,
      groundTruthShape: groundTruth ? `[${groundTruth.length}, ${groundTruth[0]?.length}, ${groundTruth[0]?.[0]?.length}]` : null,
      hasNeural: !!neural,
      neuralLength: neural?.length,
      neuralShape: neural ? `[${neural.length}, ${neural[0]?.length}, ${neural[0]?.[0]?.length}]` : null,
      numFrames,
      poseHorizonDisplayed,
      poseHorizonOriginal,
      neuralHistoryDisplayed,
      neuralHistoryOriginal,
      numNeurons,
    })
  }, [groundTruth, neural, numFrames, poseHorizonDisplayed, poseHorizonOriginal, neuralHistoryDisplayed, neuralHistoryOriginal, numNeurons])

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
    const horizon = data[frame]
    if (!horizon || !horizon[timeStep]) return null
    return horizon[timeStep]
  }, [])

  const currentGroundTruth = getCurrentPose(groundTruth, currentFrame, selectedTimeStep)
  const currentNeural = neural?.[currentFrame] || null

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
          setSelectedTimeStep(prev => Math.min(poseHorizonDisplayed - 1, prev + 10))
          break
        case 'ArrowDown':
          setSelectedTimeStep(prev => Math.max(0, prev - 10))
          break
      }
    }

    window.addEventListener('keydown', handleKeyPress)
    return () => window.removeEventListener('keydown', handleKeyPress)
  }, [numFrames, poseHorizonDisplayed])

  if (numFrames === 0) {
    return (
      <div className="data-explorer placeholder">
        <h3>No ground truth data available</h3>
        <p>Run training with test evaluation to generate data.</p>
      </div>
    )
  }

  return (
    <div className="data-explorer">
      {/* Neural Signal Panel - Full Width */}
      {currentNeural && (
        <div className="neural-panel full-width">
          <NeuralSignalViewer
            neuralData={currentNeural}
            currentFrame={selectedTimeStep}
            numNeurons={100}
            subsampleFactor={neuralSubsampleFactor}
          />
          <div className="neural-info">
            <span>Neural: {neuralHistoryDisplayed} displayed ({neuralHistoryOriginal} original, {(neuralHistoryOriginal / 50).toFixed(1)}s)</span>
            <span>Neurons: {numNeurons}</span>
          </div>
        </div>
      )}

      {/* Ground Truth Pose - Single Panel */}
      <div className="pose-panels single">
        <div className="pose-panel ground-truth-only">
          <PoseViewer3D
            poseData={currentGroundTruth}
            title="Ground Truth Pose"
            color={0x4CAF50}
          />
        </div>
      </div>

      {/* Controls */}
      <div className="viewer-controls">
        <div className="playback-controls">
          <button onClick={() => setCurrentFrame(0)} title="Reset">
            ⏮
          </button>
          <button onClick={() => setCurrentFrame(prev => Math.max(0, prev - 10))} title="Back 10">
            ⏪
          </button>
          <button onClick={() => setCurrentFrame(prev => Math.max(0, prev - 1))} title="Previous">
            ◀
          </button>
          <button onClick={() => setIsPlaying(!isPlaying)} title={isPlaying ? 'Pause' : 'Play'}>
            {isPlaying ? '⏸' : '▶'}
          </button>
          <button onClick={() => setCurrentFrame(prev => Math.min(numFrames - 1, prev + 1))} title="Next">
            ▶
          </button>
          <button onClick={() => setCurrentFrame(prev => Math.min(numFrames - 1, prev + 10))} title="Forward 10">
            ⏩
          </button>
          <button onClick={() => setCurrentFrame(numFrames - 1)} title="End">
            ⏭
          </button>
        </div>

        <div className="frame-info">
          <span>Sample: {currentFrame + 1} / {numFrames}</span>
          <span>Time Step: {selectedTimeStep + 1} / {poseHorizonDisplayed} ({((selectedTimeStep + 1) * poseSubsampleFactor / 50).toFixed(2)}s into future)</span>
          {numTotalSamples && numTotalSamples > numFrames && (
            <span className="total-info">(Showing {numFrames} of {numTotalSamples} total samples)</span>
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
            Future Time Step:
            <input
              type="range"
              min={0}
              max={poseHorizonDisplayed - 1}
              value={selectedTimeStep}
              onChange={(e) => setSelectedTimeStep(parseInt(e.target.value))}
            />
          </label>

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
        </div>
      </div>

      {/* Info Panel */}
      <div className="data-info-panel">
        <h4>Data Information</h4>
        <div className="info-grid">
          <div className="info-item">
            <span className="info-label">Samples Loaded:</span>
            <span className="info-value">{numFrames} of {numTotalSamples || numFrames}</span>
          </div>
          <div className="info-item">
            <span className="info-label">Pose (per sample):</span>
            <span className="info-value">{poseHorizonDisplayed} frames ({(poseHorizonOriginal / 50).toFixed(1)}s future)</span>
          </div>
          <div className="info-item">
            <span className="info-label">Neural (per sample):</span>
            <span className="info-value">{neuralHistoryDisplayed} frames ({(neuralHistoryOriginal / 50).toFixed(1)}s history)</span>
          </div>
          <div className="info-item">
            <span className="info-label">Neurons:</span>
            <span className="info-value">{numNeurons}</span>
          </div>
        </div>
      </div>

      {/* Keyboard shortcuts help */}
      <div className="keyboard-help">
        <small>
          Keyboard: Space (play/pause), ←/→ (prev/next sample), ↑/↓ (change future time step)
        </small>
      </div>
    </div>
  )
}

export default DataExplorer
