import React, { useState, useRef } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'

// Skeleton connections
const SKELETON_CONNECTIONS = [
  [0, 3], [3, 4], [3, 5],
  [4, 6], [6, 8], [8, 10],
  [5, 7], [7, 9], [9, 11],
  [3, 12], [12, 13], [13, 14],
  [14, 15], [14, 16],
  [15, 17], [17, 19], [19, 21],
  [16, 18], [18, 20], [20, 22],
]

function Skeleton({ keypoints, color }) {
  const kp = keypoints.reshape ? keypoints : reshapeKeypoints(keypoints)

  return (
    <group>
      {/* Draw bones */}
      {SKELETON_CONNECTIONS.map(([i, j], idx) => {
        const start = [kp[i][0], kp[i][1], kp[i][2]]
        const end = [kp[j][0], kp[j][1], kp[j][2]]
        const mid = [
          (start[0] + end[0]) / 2,
          (start[1] + end[1]) / 2,
          (start[2] + end[2]) / 2
        ]
        return (
          <line key={idx}>
            <bufferGeometry attach="geometry">
              <bufferAttribute
                attach="attributes-position"
                count={2}
                array={new Float32Array([...start, ...end])}
                itemSize={3}
              />
            </bufferGeometry>
            <lineBasicMaterial color={color} linewidth={2} />
          </line>
        )
      })}

      {/* Draw keypoints */}
      {kp.map((pos, idx) => (
        <mesh key={idx} position={[pos[0], pos[1], pos[2]]}>
          <sphereGeometry args={[0.02, 16, 16]} />
          <meshStandardMaterial color={color} />
        </mesh>
      ))}
    </group>
  )
}

function reshapeKeypoints(flat) {
  // Reshape from (69,) to (23, 3)
  const kp = []
  for (let i = 0; i < 23; i++) {
    kp.push([flat[i * 3], flat[i * 3 + 1], flat[i * 3 + 2]])
  }
  return kp
}

function PosePlayer({ predictions, groundTruth }) {
  const [frame, setFrame] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const intervalRef = useRef(null)

  const numFrames = predictions.length

  const play = () => {
    setIsPlaying(true)
    intervalRef.current = setInterval(() => {
      setFrame(f => (f + 1) % numFrames)
    }, 40) // 25 FPS
  }

  const pause = () => {
    setIsPlaying(false)
    if (intervalRef.current) {
      clearInterval(intervalRef.current)
    }
  }

  const currentPred = predictions[frame] || predictions[0]
  const currentGT = groundTruth[frame] || groundTruth[0]

  return (
    <div className="pose-player">
      <div className="player-controls">
        <button onClick={isPlaying ? pause : play}>
          {isPlaying ? '⏸ Pause' : '▶ Play'}
        </button>
        <input
          type="range"
          min="0"
          max={numFrames - 1}
          value={frame}
          onChange={(e) => setFrame(parseInt(e.target.value))}
        />
        <span>Frame: {frame} / {numFrames}</span>
      </div>

      <div className="canvas-container">
        <div className="canvas-view">
          <h4>Ground Truth</h4>
          <Canvas camera={{ position: [2, 2, 2] }}>
            <ambientLight intensity={0.5} />
            <pointLight position={[10, 10, 10]} />
            <Skeleton keypoints={reshapeKeypoints(currentGT)} color="green" />
            <OrbitControls />
            <gridHelper />
          </Canvas>
        </div>

        <div className="canvas-view">
          <h4>Prediction</h4>
          <Canvas camera={{ position: [2, 2, 2] }}>
            <ambientLight intensity={0.5} />
            <pointLight position={[10, 10, 10]} />
            <Skeleton keypoints={reshapeKeypoints(currentPred)} color="red" />
            <OrbitControls />
            <gridHelper />
          </Canvas>
        </div>
      </div>
    </div>
  )
}

export default PosePlayer
