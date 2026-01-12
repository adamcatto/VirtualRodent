import React, { useRef, useEffect, useState, useCallback } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls'

// Rodent skeleton connectivity (23 keypoints)
// Based on typical rodent motion capture keypoint layout
// Keypoints: 0-nose, 1-head, 2-neck, 3-spine_mid, 4-spine_end, 5-tail_base,
//           6-tail_mid, 7-tail_tip, 8-left_shoulder, 9-left_elbow, 10-left_wrist,
//           11-right_shoulder, 12-right_elbow, 13-right_wrist, 14-left_hip,
//           15-left_knee, 16-left_ankle, 17-right_hip, 18-right_knee, 19-right_ankle,
//           20-left_ear, 21-right_ear, 22-snout
const SKELETON_CONNECTIONS = [
  // Spine/body axis
  [0, 1], [1, 2], [2, 3], [3, 4], [4, 5],  // nose -> head -> neck -> spine_mid -> spine_end -> tail_base
  // Tail
  [5, 6], [6, 7],  // tail_base -> tail_mid -> tail_tip
  // Front left leg
  [2, 8], [8, 9], [9, 10],  // neck -> shoulder_L -> elbow_L -> wrist_L
  // Front right leg
  [2, 11], [11, 12], [12, 13],  // neck -> shoulder_R -> elbow_R -> wrist_R
  // Back left leg
  [4, 14], [14, 15], [15, 16],  // spine_end -> hip_L -> knee_L -> ankle_L
  // Back right leg
  [4, 17], [17, 18], [18, 19],  // spine_end -> hip_R -> knee_R -> ankle_R
  // Ears
  [1, 20], [1, 21],  // head -> ear_L, head -> ear_R
  // Snout
  [0, 22],  // nose -> snout
]

const KEYPOINT_COLORS = {
  spine: 0x4CAF50,
  limbs: 0x2196F3,
  tail: 0xFF9800,
  head: 0xE91E63,
}

// Number of keypoints
const NUM_KEYPOINTS = 23

// Default camera distance for 100% zoom
const DEFAULT_CAMERA_DISTANCE = 600

function PoseViewer3D({ poseData, title, color = 0x4CAF50 }) {
  const containerRef = useRef(null)
  const sceneRef = useRef(null)
  const rendererRef = useRef(null)
  const cameraRef = useRef(null)
  const controlsRef = useRef(null)
  const skeletonRef = useRef(null)
  const [zoomPercentage, setZoomPercentage] = useState(100)

  // Initialize Three.js scene
  useEffect(() => {
    if (!containerRef.current) return

    const container = containerRef.current
    const width = container.clientWidth
    const height = container.clientHeight

    // Scene
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x1a1a2e)
    sceneRef.current = scene

    // Camera - positioned to view rodent-scale coordinates (range roughly -300 to +500)
    const camera = new THREE.PerspectiveCamera(60, width / height, 1, 5000)
    camera.position.set(300, 300, 600)
    cameraRef.current = camera

    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.setSize(width, height)
    renderer.setPixelRatio(window.devicePixelRatio)
    container.appendChild(renderer.domElement)
    rendererRef.current = renderer

    // Controls
    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.dampingFactor = 0.05
    // Slow down scroll-to-zoom (default is 1.0, lower = slower)
    controls.zoomSpeed = 0.3
    controlsRef.current = controls

    // Update zoom percentage when camera moves
    const updateZoomFromCamera = () => {
      const distance = camera.position.distanceTo(controls.target)
      const zoom = Math.round((DEFAULT_CAMERA_DISTANCE / distance) * 100)
      setZoomPercentage(Math.max(10, Math.min(500, zoom)))
    }
    controls.addEventListener('change', updateZoomFromCamera)

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6)
    scene.add(ambientLight)
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8)
    directionalLight.position.set(50, 100, 50)
    scene.add(directionalLight)

    // Grid - larger to match coordinate scale
    const gridHelper = new THREE.GridHelper(800, 20, 0x444466, 0x333355)
    gridHelper.position.set(200, 0, 100)  // Center on typical pose location
    scene.add(gridHelper)

    // Axes - larger
    const axesHelper = new THREE.AxesHelper(200)
    scene.add(axesHelper)

    // Animation loop
    const animate = () => {
      requestAnimationFrame(animate)
      controls.update()
      renderer.render(scene, camera)
    }
    animate()

    // Resize handler
    const handleResize = () => {
      const newWidth = container.clientWidth
      const newHeight = container.clientHeight
      camera.aspect = newWidth / newHeight
      camera.updateProjectionMatrix()
      renderer.setSize(newWidth, newHeight)
    }
    window.addEventListener('resize', handleResize)

    return () => {
      window.removeEventListener('resize', handleResize)
      controls.removeEventListener('change', updateZoomFromCamera)
      renderer.dispose()
      container.removeChild(renderer.domElement)
    }
  }, [])

  // Zoom control handlers
  const handleZoomIn = useCallback(() => {
    if (!cameraRef.current || !controlsRef.current) return
    const camera = cameraRef.current
    const controls = controlsRef.current
    const direction = new THREE.Vector3()
    camera.getWorldDirection(direction)
    camera.position.addScaledVector(direction, 50)
    controls.update()
  }, [])

  const handleZoomOut = useCallback(() => {
    if (!cameraRef.current || !controlsRef.current) return
    const camera = cameraRef.current
    const controls = controlsRef.current
    const direction = new THREE.Vector3()
    camera.getWorldDirection(direction)
    camera.position.addScaledVector(direction, -50)
    controls.update()
  }, [])

  const handleZoomReset = useCallback(() => {
    if (!cameraRef.current || !controlsRef.current) return
    const camera = cameraRef.current
    const controls = controlsRef.current
    camera.position.set(300, 300, DEFAULT_CAMERA_DISTANCE)
    controls.update()
    setZoomPercentage(100)
  }, [])

  const handleZoomSlider = useCallback((e) => {
    if (!cameraRef.current || !controlsRef.current) return
    const newZoom = parseInt(e.target.value)
    const camera = cameraRef.current
    const controls = controlsRef.current
    
    // Calculate new distance based on zoom percentage
    const newDistance = DEFAULT_CAMERA_DISTANCE / (newZoom / 100)
    
    // Move camera along the line from target to current position
    const direction = new THREE.Vector3()
    direction.subVectors(camera.position, controls.target).normalize()
    camera.position.copy(controls.target).addScaledVector(direction, newDistance)
    controls.update()
    setZoomPercentage(newZoom)
  }, [])

  // Update skeleton when pose data changes
  useEffect(() => {
    console.log('PoseViewer3D received poseData:', {
      hasPoseData: !!poseData,
      poseDataLength: poseData?.length,
      poseDataType: typeof poseData,
      isArray: Array.isArray(poseData),
      firstValue: poseData?.[0],
      title
    })

    if (!sceneRef.current || !poseData || poseData.length === 0) return

    const scene = sceneRef.current

    // Remove old skeleton
    if (skeletonRef.current) {
      scene.remove(skeletonRef.current)
      skeletonRef.current.traverse((child) => {
        if (child.geometry) child.geometry.dispose()
        if (child.material) child.material.dispose()
      })
    }

    // Create skeleton group
    const skeletonGroup = new THREE.Group()

    // Parse pose data (69 values = 23 keypoints x 3 coordinates)
    // Data format: [x0, x1, ..., x22, y0, y1, ..., y22, z0, z1, ..., z22]
    // So: x[i] = poseData[i], y[i] = poseData[23+i], z[i] = poseData[46+i]
    const keypoints = []
    for (let i = 0; i < NUM_KEYPOINTS; i++) {
      const x = poseData[i]
      const y = poseData[NUM_KEYPOINTS + i]
      const z = poseData[2 * NUM_KEYPOINTS + i]
      // Use y as vertical (THREE.js Y is up), x as lateral, z as depth
      keypoints.push(new THREE.Vector3(x, z, y))
    }

    // Compute centroid for camera targeting
    const centroid = new THREE.Vector3()
    keypoints.forEach(kp => centroid.add(kp))
    centroid.divideScalar(keypoints.length)

    // Create spheres for keypoints (smaller size)
    const sphereGeometry = new THREE.SphereGeometry(3, 12, 12)  // Smaller spheres
    const sphereMaterial = new THREE.MeshPhongMaterial({ color: color })

    keypoints.forEach((pos, idx) => {
      const sphere = new THREE.Mesh(sphereGeometry, sphereMaterial)
      sphere.position.copy(pos)
      skeletonGroup.add(sphere)
    })

    // Create lines for bones
    const lineMaterial = new THREE.LineBasicMaterial({ color: color, linewidth: 2 })

    SKELETON_CONNECTIONS.forEach(([i, j]) => {
      if (i < keypoints.length && j < keypoints.length) {
        const geometry = new THREE.BufferGeometry().setFromPoints([
          keypoints[i],
          keypoints[j]
        ])
        const line = new THREE.Line(geometry, lineMaterial)
        skeletonGroup.add(line)
      }
    })

    scene.add(skeletonGroup)
    skeletonRef.current = skeletonGroup

    // Center camera on skeleton centroid
    if (controlsRef.current) {
      controlsRef.current.target.copy(centroid)
      controlsRef.current.update()
    }

  }, [poseData, color])

  return (
    <div className="pose-viewer-3d">
      <h4>{title}</h4>
      <div style={{ position: 'relative' }}>
        <div ref={containerRef} className="three-container" />
        <div className="zoom-controls">
          <button onClick={handleZoomOut} title="Zoom Out">−</button>
          <input
            type="range"
            min={10}
            max={500}
            value={zoomPercentage}
            onChange={handleZoomSlider}
            title="Zoom"
          />
          <button onClick={handleZoomIn} title="Zoom In">+</button>
          <span className="zoom-percentage">{zoomPercentage}%</span>
          <button onClick={handleZoomReset} title="Reset Zoom">⟲</button>
        </div>
      </div>
    </div>
  )
}

export default PoseViewer3D
