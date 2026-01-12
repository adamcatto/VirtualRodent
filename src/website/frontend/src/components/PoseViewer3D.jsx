import React, { useRef, useEffect, useState, useCallback } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls'
import { ConvexGeometry } from 'three/examples/jsm/geometries/ConvexGeometry'

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

// Body part groups for convex hull rendering
const BODY_PARTS = {
  // Head region: nose, head, ears, snout
  head: [0, 1, 20, 21, 22],
  // Torso region: neck, spine, shoulders, hips
  torso: [2, 3, 4, 5, 8, 11, 14, 17],
  // Front left leg
  frontLeftLeg: [8, 9, 10],
  // Front right leg
  frontRightLeg: [11, 12, 13],
  // Back left leg
  backLeftLeg: [14, 15, 16],
  // Back right leg
  backRightLeg: [17, 18, 19],
  // Tail
  tail: [5, 6, 7],
}

// Colors for body parts (mouse-like gray/pink)
const BODY_PART_COLORS = {
  head: 0x8B7355,      // Light brown/gray
  torso: 0x696969,     // Gray
  frontLeftLeg: 0xDEB887,  // Light tan
  frontRightLeg: 0xDEB887,
  backLeftLeg: 0xDEB887,
  backRightLeg: 0xDEB887,
  tail: 0xFFB6C1,      // Pink
}

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
  const [renderMode, setRenderMode] = useState('mesh')  // 'skeleton' | 'mesh' | 'both'

  // Initialize Three.js scene
  useEffect(() => {
    if (!containerRef.current) return

    const container = containerRef.current
    
    // Use fallback dimensions if container isn't sized yet
    let width = container.clientWidth || 800
    let height = container.clientHeight || 600

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

    // Resize handler using ResizeObserver for better responsiveness
    const handleResize = (entries) => {
      for (const entry of entries) {
        const { width: newWidth, height: newHeight } = entry.contentRect
        if (newWidth > 0 && newHeight > 0) {
          camera.aspect = newWidth / newHeight
          camera.updateProjectionMatrix()
          renderer.setSize(newWidth, newHeight)
        }
      }
    }
    
    const resizeObserver = new ResizeObserver(handleResize)
    resizeObserver.observe(container)
    
    // Force initial resize after a brief delay to ensure CSS is applied
    setTimeout(() => {
      const actualWidth = container.clientWidth
      const actualHeight = container.clientHeight
      if (actualWidth > 0 && actualHeight > 0) {
        camera.aspect = actualWidth / actualHeight
        camera.updateProjectionMatrix()
        renderer.setSize(actualWidth, actualHeight)
      }
    }, 100)

    return () => {
      resizeObserver.disconnect()
      controls.removeEventListener('change', updateZoomFromCamera)
      renderer.dispose()
      if (container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement)
      }
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

  // Helper function to create a tube between two points (for limbs)
  const createLimbTube = (p1, p2, radius, color) => {
    const direction = new THREE.Vector3().subVectors(p2, p1)
    const length = direction.length()
    const midpoint = new THREE.Vector3().addVectors(p1, p2).multiplyScalar(0.5)
    
    // Create capsule-like shape using cylinder + spheres at ends
    const geometry = new THREE.CylinderGeometry(radius, radius, length, 8)
    const material = new THREE.MeshPhongMaterial({ 
      color: color, 
      transparent: true, 
      opacity: 0.85,
      shininess: 30
    })
    
    const cylinder = new THREE.Mesh(geometry, material)
    cylinder.position.copy(midpoint)
    
    // Orient cylinder to point from p1 to p2
    const axis = new THREE.Vector3(0, 1, 0)
    cylinder.quaternion.setFromUnitVectors(axis, direction.normalize())
    
    return cylinder
  }

  // Update skeleton when pose data changes
  useEffect(() => {
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

    // === RENDER SKELETON (points + lines) ===
    if (renderMode === 'skeleton' || renderMode === 'both') {
      // Create spheres for keypoints
      const sphereGeometry = new THREE.SphereGeometry(3, 12, 12)
      const sphereMaterial = new THREE.MeshPhongMaterial({ color: color })

      keypoints.forEach((pos) => {
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
    }

    // === RENDER MESH (convex hulls + tubes) ===
    if (renderMode === 'mesh' || renderMode === 'both') {
      // Create convex hull for head
      const headPoints = BODY_PARTS.head.map(i => keypoints[i])
      if (headPoints.length >= 4) {
        try {
          const headGeometry = new ConvexGeometry(headPoints)
          const headMaterial = new THREE.MeshPhongMaterial({
            color: BODY_PART_COLORS.head,
            transparent: true,
            opacity: 0.9,
            shininess: 50,
            side: THREE.DoubleSide
          })
          const headMesh = new THREE.Mesh(headGeometry, headMaterial)
          skeletonGroup.add(headMesh)
        } catch (e) {
          // Fallback if convex hull fails
        }
      }

      // Create convex hull for torso
      const torsoPoints = BODY_PARTS.torso.map(i => keypoints[i])
      if (torsoPoints.length >= 4) {
        try {
          const torsoGeometry = new ConvexGeometry(torsoPoints)
          const torsoMaterial = new THREE.MeshPhongMaterial({
            color: BODY_PART_COLORS.torso,
            transparent: true,
            opacity: 0.85,
            shininess: 30,
            side: THREE.DoubleSide
          })
          const torsoMesh = new THREE.Mesh(torsoGeometry, torsoMaterial)
          skeletonGroup.add(torsoMesh)
        } catch (e) {
          // Fallback if convex hull fails
        }
      }

      // Create tube for tail
      const tailIndices = BODY_PARTS.tail
      for (let i = 0; i < tailIndices.length - 1; i++) {
        const radius = 4 - i * 1  // Tapers from 4 to 2
        const tube = createLimbTube(
          keypoints[tailIndices[i]], 
          keypoints[tailIndices[i + 1]], 
          Math.max(1, radius),
          BODY_PART_COLORS.tail
        )
        skeletonGroup.add(tube)
      }

      // Create tubes for limbs
      const limbParts = ['frontLeftLeg', 'frontRightLeg', 'backLeftLeg', 'backRightLeg']
      limbParts.forEach(limbName => {
        const indices = BODY_PARTS[limbName]
        for (let i = 0; i < indices.length - 1; i++) {
          const radius = 5 - i * 1.5  // Tapers from shoulder to paw
          const tube = createLimbTube(
            keypoints[indices[i]], 
            keypoints[indices[i + 1]], 
            Math.max(2, radius),
            BODY_PART_COLORS[limbName]
          )
          skeletonGroup.add(tube)
        }
        
        // Add a small sphere at the paw
        const pawIndex = indices[indices.length - 1]
        const pawGeometry = new THREE.SphereGeometry(3, 8, 8)
        const pawMaterial = new THREE.MeshPhongMaterial({ 
          color: 0xFFCCCC,  // Pink paw
          shininess: 50
        })
        const paw = new THREE.Mesh(pawGeometry, pawMaterial)
        paw.position.copy(keypoints[pawIndex])
        skeletonGroup.add(paw)
      })

      // Add eyes (small black spheres near the head)
      const headPos = keypoints[1]  // head keypoint
      const nosePos = keypoints[0]  // nose keypoint
      const leftEarPos = keypoints[20]
      const rightEarPos = keypoints[21]
      
      // Calculate eye positions (between head and nose, offset laterally)
      const headToNose = new THREE.Vector3().subVectors(nosePos, headPos).normalize()
      const eyeOffset = headToNose.clone().multiplyScalar(10)
      const lateralDir = new THREE.Vector3().subVectors(leftEarPos, rightEarPos).normalize()
      
      const eyeGeometry = new THREE.SphereGeometry(2.5, 8, 8)
      const eyeMaterial = new THREE.MeshPhongMaterial({ color: 0x111111, shininess: 100 })
      
      const leftEye = new THREE.Mesh(eyeGeometry, eyeMaterial)
      leftEye.position.copy(headPos).add(eyeOffset).addScaledVector(lateralDir, 6)
      skeletonGroup.add(leftEye)
      
      const rightEye = new THREE.Mesh(eyeGeometry, eyeMaterial)
      rightEye.position.copy(headPos).add(eyeOffset).addScaledVector(lateralDir, -6)
      skeletonGroup.add(rightEye)

      // Add nose (pink sphere at snout)
      const noseGeometry = new THREE.SphereGeometry(3, 8, 8)
      const noseMaterial = new THREE.MeshPhongMaterial({ color: 0xFFAAAA, shininess: 50 })
      const nose = new THREE.Mesh(noseGeometry, noseMaterial)
      nose.position.copy(keypoints[22])  // snout
      skeletonGroup.add(nose)

      // Add ears (larger spheres at ear positions)
      const earGeometry = new THREE.SphereGeometry(8, 12, 12)
      const earMaterial = new THREE.MeshPhongMaterial({ 
        color: 0xDEB887,
        transparent: true,
        opacity: 0.9,
        shininess: 20
      })
      
      const leftEar = new THREE.Mesh(earGeometry, earMaterial)
      leftEar.position.copy(keypoints[20])
      leftEar.scale.set(1, 0.3, 0.8)  // Flatten the ear
      skeletonGroup.add(leftEar)
      
      const rightEar = new THREE.Mesh(earGeometry, earMaterial)
      rightEar.position.copy(keypoints[21])
      rightEar.scale.set(1, 0.3, 0.8)
      skeletonGroup.add(rightEar)
    }

    scene.add(skeletonGroup)
    skeletonRef.current = skeletonGroup

    // Center camera on skeleton centroid
    if (controlsRef.current) {
      controlsRef.current.target.copy(centroid)
      controlsRef.current.update()
    }

  }, [poseData, color, renderMode])

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
        <div className="render-mode-controls">
          <button 
            className={renderMode === 'skeleton' ? 'active' : ''} 
            onClick={() => setRenderMode('skeleton')}
            title="Skeleton only"
          >
            🦴
          </button>
          <button 
            className={renderMode === 'mesh' ? 'active' : ''} 
            onClick={() => setRenderMode('mesh')}
            title="Mesh"
          >
            🐭
          </button>
          <button 
            className={renderMode === 'both' ? 'active' : ''} 
            onClick={() => setRenderMode('both')}
            title="Both"
          >
            ⚡
          </button>
        </div>
      </div>
    </div>
  )
}

export default PoseViewer3D
