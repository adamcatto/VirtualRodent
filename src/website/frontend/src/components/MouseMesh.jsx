import React, { useRef, useEffect, useState, useCallback } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls'

// Keypoint indices
const KP = {
  NOSE: 0,
  HEAD: 1,
  NECK: 2,
  SPINE_MID: 3,
  SPINE_END: 4,
  TAIL_BASE: 5,
  TAIL_MID: 6,
  TAIL_TIP: 7,
  LEFT_SHOULDER: 8,
  LEFT_ELBOW: 9,
  LEFT_WRIST: 10,
  RIGHT_SHOULDER: 11,
  RIGHT_ELBOW: 12,
  RIGHT_WRIST: 13,
  LEFT_HIP: 14,
  LEFT_KNEE: 15,
  LEFT_ANKLE: 16,
  RIGHT_HIP: 17,
  RIGHT_KNEE: 18,
  RIGHT_ANKLE: 19,
  LEFT_EAR: 20,
  RIGHT_EAR: 21,
  SNOUT: 22,
}

const NUM_KEYPOINTS = 23
const DEFAULT_CAMERA_DISTANCE = 600

// Mouse colors
const COLORS = {
  body: 0x8B7355,      // Brown/gray fur
  belly: 0xD2B48C,     // Tan belly
  ears: 0xFFB6C1,      // Pink ears
  nose: 0xFFAAAA,      // Pink nose
  eyes: 0x111111,      // Black eyes
  tail: 0xFFB6C1,      // Pink tail
  paws: 0xFFCCCC,      // Light pink paws
}

/**
 * Create a smooth tube/capsule between two points
 */
function createCapsule(scene, p1, p2, radius, color, segments = 8) {
  const direction = new THREE.Vector3().subVectors(p2, p1)
  const length = direction.length()
  
  if (length < 0.1) return null
  
  const midpoint = new THREE.Vector3().addVectors(p1, p2).multiplyScalar(0.5)
  
  // Create capsule geometry (cylinder + hemispheres)
  const geometry = new THREE.CapsuleGeometry(radius, length, 4, segments)
  const material = new THREE.MeshPhongMaterial({ 
    color,
    shininess: 30,
  })
  
  const mesh = new THREE.Mesh(geometry, material)
  mesh.position.copy(midpoint)
  
  // Orient to point from p1 to p2
  const axis = new THREE.Vector3(0, 1, 0)
  const quaternion = new THREE.Quaternion()
  quaternion.setFromUnitVectors(axis, direction.clone().normalize())
  mesh.quaternion.copy(quaternion)
  
  return mesh
}

/**
 * Create an ellipsoid (scaled sphere) at a position
 */
function createEllipsoid(position, radiusX, radiusY, radiusZ, color) {
  const geometry = new THREE.SphereGeometry(1, 16, 12)
  const material = new THREE.MeshPhongMaterial({ 
    color,
    shininess: 30,
  })
  const mesh = new THREE.Mesh(geometry, material)
  mesh.position.copy(position)
  mesh.scale.set(radiusX, radiusY, radiusZ)
  return mesh
}

/**
 * Create a smooth spline-based body using TubeGeometry
 */
function createSplineBody(keypoints, indices, radius, taperFn, color) {
  const points = indices.map(i => keypoints[i])
  
  // Create a smooth curve through the points
  const curve = new THREE.CatmullRomCurve3(points)
  
  // Create tube geometry with varying radius
  const segments = 32
  const radialSegments = 12
  
  // Custom tube with tapering
  const geometry = new THREE.TubeGeometry(curve, segments, radius, radialSegments, false)
  
  // Apply tapering by modifying vertices
  const positions = geometry.attributes.position
  for (let i = 0; i < positions.count; i++) {
    const vertex = new THREE.Vector3(
      positions.getX(i),
      positions.getY(i),
      positions.getZ(i)
    )
    
    // Find closest point on curve to determine taper
    // This is approximate - we use the segment index
    const segmentIndex = Math.floor(i / radialSegments) / segments
    const taper = taperFn(segmentIndex)
    
    // Get the center point at this segment
    const centerPoint = curve.getPoint(Math.min(1, segmentIndex))
    
    // Scale distance from center
    const offset = vertex.clone().sub(centerPoint)
    const currentRadius = offset.length()
    if (currentRadius > 0.01) {
      offset.normalize().multiplyScalar(currentRadius * taper)
      positions.setXYZ(i, centerPoint.x + offset.x, centerPoint.y + offset.y, centerPoint.z + offset.z)
    }
  }
  
  geometry.attributes.position.needsUpdate = true
  geometry.computeVertexNormals()
  
  const material = new THREE.MeshPhongMaterial({
    color,
    shininess: 30,
    side: THREE.DoubleSide,
  })
  
  return new THREE.Mesh(geometry, material)
}

/**
 * MouseMesh - A procedurally generated 3D mouse driven by keypoints
 */
function MouseMesh({ poseData, title }) {
  const containerRef = useRef(null)
  const sceneRef = useRef(null)
  const rendererRef = useRef(null)
  const cameraRef = useRef(null)
  const controlsRef = useRef(null)
  const mouseGroupRef = useRef(null)
  const [zoomPercentage, setZoomPercentage] = useState(100)

  // Parse keypoints from pose data
  const parseKeypoints = useCallback((data) => {
    if (!data || data.length < 69) return null
    
    const keypoints = []
    for (let i = 0; i < NUM_KEYPOINTS; i++) {
      const x = data[i]
      const y = data[NUM_KEYPOINTS + i]
      const z = data[2 * NUM_KEYPOINTS + i]
      // Convert to Three.js coords: y as vertical (up), x as lateral, z as depth
      keypoints.push(new THREE.Vector3(x, z, y))
    }
    return keypoints
  }, [])

  // Build the mouse mesh from keypoints
  const buildMouseMesh = useCallback((keypoints) => {
    const group = new THREE.Group()
    
    // === BODY (main torso) ===
    // Create smooth body from neck to tail base
    const bodyPoints = [
      keypoints[KP.NECK],
      keypoints[KP.SPINE_MID],
      keypoints[KP.SPINE_END],
      keypoints[KP.TAIL_BASE],
    ]
    const bodyCurve = new THREE.CatmullRomCurve3(bodyPoints)
    const bodyGeometry = new THREE.TubeGeometry(bodyCurve, 24, 18, 12, false)
    const bodyMaterial = new THREE.MeshPhongMaterial({ 
      color: COLORS.body,
      shininess: 30,
    })
    const bodyMesh = new THREE.Mesh(bodyGeometry, bodyMaterial)
    group.add(bodyMesh)
    
    // === HEAD ===
    // Create head as ellipsoid from neck to nose
    const headCenter = new THREE.Vector3().lerpVectors(
      keypoints[KP.HEAD],
      keypoints[KP.NOSE],
      0.3
    )
    const headDir = new THREE.Vector3().subVectors(keypoints[KP.NOSE], keypoints[KP.NECK])
    const headLength = headDir.length() * 0.6
    
    const headMesh = createEllipsoid(headCenter, 15, 12, headLength * 0.4, COLORS.body)
    // Orient head towards nose
    headMesh.lookAt(keypoints[KP.NOSE])
    group.add(headMesh)
    
    // === SNOUT ===
    const snoutMesh = createEllipsoid(keypoints[KP.SNOUT], 6, 5, 8, COLORS.body)
    group.add(snoutMesh)
    
    // Nose tip (pink)
    const noseGeometry = new THREE.SphereGeometry(4, 8, 8)
    const noseMaterial = new THREE.MeshPhongMaterial({ color: COLORS.nose, shininess: 60 })
    const noseMesh = new THREE.Mesh(noseGeometry, noseMaterial)
    noseMesh.position.copy(keypoints[KP.NOSE])
    group.add(noseMesh)
    
    // === EYES ===
    const headToNose = new THREE.Vector3().subVectors(keypoints[KP.NOSE], keypoints[KP.HEAD]).normalize()
    const eyeOffset = headToNose.clone().multiplyScalar(8)
    const leftEarDir = new THREE.Vector3().subVectors(keypoints[KP.LEFT_EAR], keypoints[KP.HEAD]).normalize()
    const rightEarDir = new THREE.Vector3().subVectors(keypoints[KP.RIGHT_EAR], keypoints[KP.HEAD]).normalize()
    
    const eyeGeometry = new THREE.SphereGeometry(3, 8, 8)
    const eyeMaterial = new THREE.MeshPhongMaterial({ color: COLORS.eyes, shininess: 100 })
    
    const leftEye = new THREE.Mesh(eyeGeometry, eyeMaterial)
    leftEye.position.copy(keypoints[KP.HEAD]).add(eyeOffset).addScaledVector(leftEarDir, 8)
    group.add(leftEye)
    
    const rightEye = new THREE.Mesh(eyeGeometry, eyeMaterial)
    rightEye.position.copy(keypoints[KP.HEAD]).add(eyeOffset).addScaledVector(rightEarDir, 8)
    group.add(rightEye)
    
    // === EARS ===
    const earGeometry = new THREE.SphereGeometry(10, 12, 12)
    const earMaterial = new THREE.MeshPhongMaterial({ 
      color: COLORS.ears,
      shininess: 20,
      transparent: true,
      opacity: 0.95,
    })
    
    const leftEar = new THREE.Mesh(earGeometry, earMaterial)
    leftEar.position.copy(keypoints[KP.LEFT_EAR])
    leftEar.scale.set(1, 0.2, 0.8)
    group.add(leftEar)
    
    const rightEar = new THREE.Mesh(earGeometry, earMaterial)
    rightEar.position.copy(keypoints[KP.RIGHT_EAR])
    rightEar.scale.set(1, 0.2, 0.8)
    group.add(rightEar)
    
    // === TAIL ===
    const tailPoints = [
      keypoints[KP.TAIL_BASE],
      keypoints[KP.TAIL_MID],
      keypoints[KP.TAIL_TIP],
    ]
    const tailCurve = new THREE.CatmullRomCurve3(tailPoints)
    const tailGeometry = new THREE.TubeGeometry(tailCurve, 16, 3, 8, false)
    const tailMaterial = new THREE.MeshPhongMaterial({ 
      color: COLORS.tail,
      shininess: 40,
    })
    const tailMesh = new THREE.Mesh(tailGeometry, tailMaterial)
    group.add(tailMesh)
    
    // === FRONT LEGS ===
    const legRadius = 4
    const pawRadius = 3
    
    // Left front leg
    const leftUpperArm = createCapsule(group, keypoints[KP.LEFT_SHOULDER], keypoints[KP.LEFT_ELBOW], legRadius, COLORS.body)
    if (leftUpperArm) group.add(leftUpperArm)
    
    const leftForearm = createCapsule(group, keypoints[KP.LEFT_ELBOW], keypoints[KP.LEFT_WRIST], legRadius * 0.8, COLORS.body)
    if (leftForearm) group.add(leftForearm)
    
    const leftPawGeom = new THREE.SphereGeometry(pawRadius, 8, 8)
    const pawMaterial = new THREE.MeshPhongMaterial({ color: COLORS.paws, shininess: 30 })
    const leftPaw = new THREE.Mesh(leftPawGeom, pawMaterial)
    leftPaw.position.copy(keypoints[KP.LEFT_WRIST])
    group.add(leftPaw)
    
    // Right front leg
    const rightUpperArm = createCapsule(group, keypoints[KP.RIGHT_SHOULDER], keypoints[KP.RIGHT_ELBOW], legRadius, COLORS.body)
    if (rightUpperArm) group.add(rightUpperArm)
    
    const rightForearm = createCapsule(group, keypoints[KP.RIGHT_ELBOW], keypoints[KP.RIGHT_WRIST], legRadius * 0.8, COLORS.body)
    if (rightForearm) group.add(rightForearm)
    
    const rightPaw = new THREE.Mesh(leftPawGeom.clone(), pawMaterial)
    rightPaw.position.copy(keypoints[KP.RIGHT_WRIST])
    group.add(rightPaw)
    
    // === BACK LEGS ===
    const backLegRadius = 5
    
    // Left back leg
    const leftThigh = createCapsule(group, keypoints[KP.LEFT_HIP], keypoints[KP.LEFT_KNEE], backLegRadius, COLORS.body)
    if (leftThigh) group.add(leftThigh)
    
    const leftShin = createCapsule(group, keypoints[KP.LEFT_KNEE], keypoints[KP.LEFT_ANKLE], backLegRadius * 0.7, COLORS.body)
    if (leftShin) group.add(leftShin)
    
    const leftFootGeom = new THREE.SphereGeometry(pawRadius * 1.2, 8, 8)
    const leftFoot = new THREE.Mesh(leftFootGeom, pawMaterial)
    leftFoot.position.copy(keypoints[KP.LEFT_ANKLE])
    leftFoot.scale.set(1.2, 0.6, 1.5)  // Flatten and elongate foot
    group.add(leftFoot)
    
    // Right back leg
    const rightThigh = createCapsule(group, keypoints[KP.RIGHT_HIP], keypoints[KP.RIGHT_KNEE], backLegRadius, COLORS.body)
    if (rightThigh) group.add(rightThigh)
    
    const rightShin = createCapsule(group, keypoints[KP.RIGHT_KNEE], keypoints[KP.RIGHT_ANKLE], backLegRadius * 0.7, COLORS.body)
    if (rightShin) group.add(rightShin)
    
    const rightFoot = new THREE.Mesh(leftFootGeom.clone(), pawMaterial)
    rightFoot.position.copy(keypoints[KP.RIGHT_ANKLE])
    rightFoot.scale.set(1.2, 0.6, 1.5)
    group.add(rightFoot)
    
    return group
  }, [])

  // Initialize Three.js scene
  useEffect(() => {
    if (!containerRef.current) return

    const container = containerRef.current
    let width = container.clientWidth || 800
    let height = container.clientHeight || 600

    // Scene
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x1a1a2e)
    sceneRef.current = scene

    // Camera
    const camera = new THREE.PerspectiveCamera(60, width / height, 1, 5000)
    camera.position.set(300, 300, 600)
    cameraRef.current = camera

    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.setSize(width, height)
    renderer.setPixelRatio(window.devicePixelRatio)
    renderer.shadowMap.enabled = true
    container.appendChild(renderer.domElement)
    rendererRef.current = renderer

    // Controls
    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.dampingFactor = 0.05
    controls.zoomSpeed = 0.3
    controlsRef.current = controls

    // Update zoom display
    const updateZoomFromCamera = () => {
      const distance = camera.position.distanceTo(controls.target)
      const zoom = Math.round((DEFAULT_CAMERA_DISTANCE / distance) * 100)
      setZoomPercentage(Math.max(10, Math.min(500, zoom)))
    }
    controls.addEventListener('change', updateZoomFromCamera)

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5)
    scene.add(ambientLight)
    
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8)
    directionalLight.position.set(100, 200, 100)
    directionalLight.castShadow = true
    scene.add(directionalLight)
    
    const fillLight = new THREE.DirectionalLight(0xffffff, 0.3)
    fillLight.position.set(-100, 50, -100)
    scene.add(fillLight)

    // Grid
    const gridHelper = new THREE.GridHelper(800, 20, 0x444466, 0x333355)
    gridHelper.position.set(200, 0, 100)
    scene.add(gridHelper)

    // Axes
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
    
    // Initial resize
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

  // Update mouse mesh when pose data changes
  useEffect(() => {
    if (!sceneRef.current || !poseData || poseData.length === 0) return

    const scene = sceneRef.current
    const keypoints = parseKeypoints(poseData)
    
    if (!keypoints) return

    // Remove old mouse mesh
    if (mouseGroupRef.current) {
      scene.remove(mouseGroupRef.current)
      mouseGroupRef.current.traverse((child) => {
        if (child.geometry) child.geometry.dispose()
        if (child.material) child.material.dispose()
      })
    }

    // Build new mouse mesh
    const mouseGroup = buildMouseMesh(keypoints)
    scene.add(mouseGroup)
    mouseGroupRef.current = mouseGroup

    // Center camera on mouse
    const centroid = new THREE.Vector3()
    keypoints.forEach(kp => centroid.add(kp))
    centroid.divideScalar(keypoints.length)
    
    if (controlsRef.current) {
      controlsRef.current.target.copy(centroid)
      controlsRef.current.update()
    }

  }, [poseData, parseKeypoints, buildMouseMesh])

  // Zoom handlers
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
    
    const newDistance = DEFAULT_CAMERA_DISTANCE / (newZoom / 100)
    const direction = new THREE.Vector3()
    direction.subVectors(camera.position, controls.target).normalize()
    camera.position.copy(controls.target).addScaledVector(direction, newDistance)
    controls.update()
    setZoomPercentage(newZoom)
  }, [])

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

export default MouseMesh

