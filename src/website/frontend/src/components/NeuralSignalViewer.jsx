import React, { useRef, useEffect } from 'react'

// Inferno colormap - from matplotlib
// Interpolates between control points for smooth gradient
function infernoColor(t) {
  // Clamp t to [0, 1]
  t = Math.max(0, Math.min(1, t))

  // Inferno control points (from matplotlib)
  const colors = [
    [0.001462, 0.000466, 0.013866],  // 0.0 - almost black
    [0.087411, 0.044556, 0.224813],  // 0.125 - dark purple
    [0.258234, 0.038571, 0.406152],  // 0.25 - purple
    [0.416331, 0.090834, 0.432943],  // 0.375 - magenta
    [0.578304, 0.148039, 0.404411],  // 0.5 - pink-red
    [0.735683, 0.215906, 0.330245],  // 0.625 - red
    [0.865006, 0.316822, 0.226055],  // 0.75 - orange
    [0.954506, 0.468744, 0.099874],  // 0.875 - yellow-orange
    [0.988362, 0.998364, 0.644924],  // 1.0 - bright yellow
  ]

  const numColors = colors.length
  const scaledT = t * (numColors - 1)
  const idx = Math.floor(scaledT)
  const frac = scaledT - idx

  if (idx >= numColors - 1) {
    const c = colors[numColors - 1]
    return [Math.round(c[0] * 255), Math.round(c[1] * 255), Math.round(c[2] * 255)]
  }

  const c1 = colors[idx]
  const c2 = colors[idx + 1]

  return [
    Math.round((c1[0] + frac * (c2[0] - c1[0])) * 255),
    Math.round((c1[1] + frac * (c2[1] - c1[1])) * 255),
    Math.round((c1[2] + frac * (c2[2] - c1[2])) * 255),
  ]
}

function NeuralSignalViewer({ neuralData, currentFrame, numNeurons = 50, subsampleFactor = 10 }) {
  const canvasRef = useRef(null)

  useEffect(() => {
    if (!canvasRef.current || !neuralData || neuralData.length === 0) return

    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    const width = canvas.width
    const height = canvas.height

    // Clear canvas
    ctx.fillStyle = '#1a1a2e'
    ctx.fillRect(0, 0, width, height)

    // Get neural data (expected shape: [time_steps, num_neurons])
    // Check if data is 3D [batch, time, neurons] or 2D [time, neurons]
    const is3D = Array.isArray(neuralData) && Array.isArray(neuralData[0]) && Array.isArray(neuralData[0][0])
    const frameData = is3D ? neuralData[0] : neuralData

    if (!frameData || frameData.length === 0) return
    if (!Array.isArray(frameData[0])) {
      console.warn('NeuralSignalViewer: Expected 2D array [time, neurons], got 1D')
      return
    }

    const timeSteps = frameData.length
    const neurons = frameData[0]?.length || 0

    console.log('NeuralSignalViewer rendering:', { timeSteps, neurons, is3D })

    // Subsample neurons if too many
    const displayNeurons = Math.min(neurons, numNeurons)
    const neuronStep = Math.max(1, Math.floor(neurons / displayNeurons))

    // Calculate dimensions
    const cellWidth = width / timeSteps
    const cellHeight = height / displayNeurons

    // Find min/max for normalization
    let minVal = Infinity, maxVal = -Infinity
    frameData.forEach(row => {
      row.forEach(val => {
        minVal = Math.min(minVal, val)
        maxVal = Math.max(maxVal, val)
      })
    })
    const range = maxVal - minVal || 1

    // Draw heatmap using inferno colormap
    for (let t = 0; t < timeSteps; t++) {
      for (let n = 0; n < displayNeurons; n++) {
        const neuronIdx = n * neuronStep
        if (neuronIdx >= neurons) continue

        const val = frameData[t][neuronIdx]
        const normalized = (val - minVal) / range

        // Use inferno colormap
        const [r, g, b] = infernoColor(normalized)

        ctx.fillStyle = `rgb(${r}, ${g}, ${b})`
        ctx.fillRect(
          t * cellWidth,
          n * cellHeight,
          cellWidth + 1,
          cellHeight + 1
        )
      }
    }

    // Draw current time indicator (accounting for subsampling)
    // currentFrame is in original time scale, convert to subsampled scale
    const scaledFrame = Math.floor(currentFrame / subsampleFactor)
    if (scaledFrame >= 0 && scaledFrame < timeSteps) {
      ctx.strokeStyle = '#ffffff'
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.moveTo(scaledFrame * cellWidth, 0)
      ctx.lineTo(scaledFrame * cellWidth, height)
      ctx.stroke()
    }

    // Draw axis labels
    ctx.fillStyle = '#ffffff'
    ctx.font = '10px monospace'
    ctx.fillText('Time', width - 30, height - 5)
    ctx.save()
    ctx.translate(10, height / 2)
    ctx.rotate(-Math.PI / 2)
    ctx.fillText('Neurons', 0, 0)
    ctx.restore()

  }, [neuralData, currentFrame, numNeurons, subsampleFactor])

  return (
    <div className="neural-signal-viewer">
      <h4>Neural Activity</h4>
      <canvas
        ref={canvasRef}
        width={600}
        height={150}
        className="neural-canvas"
      />
      <div className="neural-legend">
        <span className="legend-low">Low</span>
        <div className="legend-gradient" />
        <span className="legend-high">High</span>
      </div>
    </div>
  )
}

export default NeuralSignalViewer
