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

// Mouse colors - more realistic
const COLORS = {
  body: 0x6B5344,      // Darker brown fur
  bodyLight: 0x8B7355, // Lighter brown highlights
  belly: 0xC4A77D,     // Tan belly
  ears: 0xE8B4B8,      // Pink ears (inner)
  earOuter: 0x7A6655,  // Darker ear outer
  nose: 0xFFAAAA,      // Pink nose
  eyes: 0x111111,      // Black eyes
  eyeHighlight: 0xFFFFFF, // Eye highlight
  tail: 0xDDA0A0,      // Pinkish tail
  paws: 0xE8C4C4,      // Light pink paws
  whiskers: 0x333333,  // Dark whiskers
}

// Spring physics constants
const SPRING_STIFFNESS = 0.15
const SPRING_DAMPING = 0.7

/**
 * Simple 2-bone IK solver (for limbs)
 * Given shoulder, target (paw), and bone lengths, returns elbow position
 */
function solveIK2Bone(shoulder, target, upperLength, lowerLength, bendDirection) {
  const shoulderToTarget = new THREE.Vector3().subVectors(target, shoulder)
  const distance = shoulderToTarget.length()
  
  // Clamp distance to reachable range
  const maxReach = upperLength + lowerLength - 0.1
  const minReach = Math.abs(upperLength - lowerLength) + 0.1
  
  if (distance > maxReach) {
    // Target too far - stretch towards it
    const dir = shoulderToTarget.normalize()
    return shoulder.clone().addScaledVector(dir, upperLength)
  }
  
  if (distance < minReach) {
    // Target too close - bend maximally
    const dir = shoulderToTarget.normalize()
    const perpendicular = new THREE.Vector3().crossVectors(dir, bendDirection).normalize()
    return shoulder.clone().addScaledVector(dir, upperLength * 0.5).addScaledVector(perpendicular, upperLength * 0.8)
  }
  
  // Law of cosines to find elbow angle
  const a = upperLength
  const b = lowerLength
  const c = distance
  
  const cosAngle = (a * a + c * c - b * b) / (2 * a * c)
  const angle = Math.acos(Math.max(-1, Math.min(1, cosAngle)))
  
  // Calculate elbow position
  const dirToTarget = shoulderToTarget.normalize()
  const perpendicular = new THREE.Vector3().crossVectors(dirToTarget, bendDirection).normalize()
  
  const elbow = shoulder.clone()
    .addScaledVector(dirToTarget, Math.cos(angle) * upperLength)
    .addScaledVector(perpendicular, Math.sin(angle) * upperLength)
  
  return elbow
}

/**
 * Spring class for secondary motion
 */
class Spring {
  constructor(target) {
    this.position = target.clone()
    this.velocity = new THREE.Vector3()
    this.target = target.clone()
  }
  
  update(newTarget, dt = 1/50) {
    this.target.copy(newTarget)
    
    // Spring force
    const displacement = new THREE.Vector3().subVectors(this.target, this.position)
    const springForce = displacement.multiplyScalar(SPRING_STIFFNESS)
    
    // Apply force to velocity
    this.velocity.add(springForce)
    
    // Damping
    this.velocity.multiplyScalar(SPRING_DAMPING)
    
    // Update position
    this.position.add(this.velocity)
    
    return this.position.clone()
  }
}

/**
 * Create a smooth tube/capsule between two points
 */
function createCapsule(p1, p2, radius, material) {
  const direction = new THREE.Vector3().subVectors(p2, p1)
  const length = direction.length()
  
  if (length < 0.1) return null
  
  const midpoint = new THREE.Vector3().addVectors(p1, p2).multiplyScalar(0.5)
  
  const geometry = new THREE.CapsuleGeometry(radius, length, 4, 8)
  const mesh = new THREE.Mesh(geometry, material)
  mesh.position.copy(midpoint)
  
  const axis = new THREE.Vector3(0, 1, 0)
  const quaternion = new THREE.Quaternion()
  quaternion.setFromUnitVectors(axis, direction.clone().normalize())
  mesh.quaternion.copy(quaternion)
  
  return mesh
}

/**
 * Create whiskers
 */
function createWhiskers(snoutPos, headDir, upDir, material) {
  const whiskers = new THREE.Group()
  const rightDir = new THREE.Vector3().crossVectors(headDir, upDir).normalize()
  
  const whiskerAngles = [-30, -10, 10, 30]
  const whiskerLength = 25
  
  whiskerAngles.forEach((angleDeg, i) => {
    const angle = (angleDeg * Math.PI) / 180
    const side = i < 2 ? 1 : -1
    
    // Left whisker
    const leftStart = snoutPos.clone().addScaledVector(rightDir, -4)
    const leftEnd = leftStart.clone()
      .addScaledVector(rightDir, -whiskerLength * Math.cos(angle))
      .addScaledVector(headDir, whiskerLength * Math.sin(angle) * 0.3)
      .addScaledVector(upDir, side * 3)
    
    const leftGeom = new THREE.BufferGeometry().setFromPoints([leftStart, leftEnd])
    const leftWhisker = new THREE.Line(leftGeom, material)
    whiskers.add(leftWhisker)
    
    // Right whisker
    const rightStart = snoutPos.clone().addScaledVector(rightDir, 4)
    const rightEnd = rightStart.clone()
      .addScaledVector(rightDir, whiskerLength * Math.cos(angle))
      .addScaledVector(headDir, whiskerLength * Math.sin(angle) * 0.3)
      .addScaledVector(upDir, side * 3)
    
    const rightGeom = new THREE.BufferGeometry().setFromPoints([rightStart, rightEnd])
    const rightWhisker = new THREE.Line(rightGeom, material)
    whiskers.add(rightWhisker)
  })
  
  return whiskers
}

/**
 * MouseMesh - A procedurally generated 3D mouse with IK and spring physics
 */
function MouseMesh({ poseData, title }) {
  const containerRef = useRef(null)
  const sceneRef = useRef(null)
  const rendererRef = useRef(null)
  const cameraRef = useRef(null)
  const controlsRef = useRef(null)
  const mouseGroupRef = useRef(null)
  const [zoomPercentage, setZoomPercentage] = useState(100)
  
  // Spring physics state for secondary motion
  const springsRef = useRef({
    leftEar: null,
    rightEar: null,
    tailMid: null,
    tailTip: null,
  })
  
  // Previous keypoints for interpolation
  const prevKeypointsRef = useRef(null)
  
  // Materials (reusable)
  const materialsRef = useRef(null)

  // Initialize materials
  const getMaterials = useCallback(() => {
    if (materialsRef.current) return materialsRef.current
    
    materialsRef.current = {
      body: new THREE.MeshStandardMaterial({ 
        color: COLORS.body,
        roughness: 0.8,
        metalness: 0.0,
      }),
      bodyLight: new THREE.MeshStandardMaterial({ 
        color: COLORS.bodyLight,
        roughness: 0.7,
        metalness: 0.0,
      }),
      belly: new THREE.MeshStandardMaterial({ 
        color: COLORS.belly,
        roughness: 0.8,
        metalness: 0.0,
      }),
      ear: new THREE.MeshStandardMaterial({ 
        color: COLORS.ears,
        roughness: 0.6,
        metalness: 0.0,
        side: THREE.DoubleSide,
      }),
      nose: new THREE.MeshStandardMaterial({ 
        color: COLORS.nose, 
        roughness: 0.4,
        metalness: 0.1,
      }),
      eye: new THREE.MeshStandardMaterial({ 
        color: COLORS.eyes, 
        roughness: 0.1,
        metalness: 0.3,
      }),
      tail: new THREE.MeshStandardMaterial({ 
        color: COLORS.tail,
        roughness: 0.5,
        metalness: 0.0,
      }),
      paw: new THREE.MeshStandardMaterial({ 
        color: COLORS.paws,
        roughness: 0.6,
        metalness: 0.0,
      }),
      whisker: new THREE.LineBasicMaterial({ 
        color: COLORS.whiskers,
        linewidth: 1,
      }),
    }
    
    return materialsRef.current
  }, [])

  // Parse keypoints from pose data
  const parseKeypoints = useCallback((data) => {
    if (!data || data.length < 69) return null
    
    const keypoints = []
    for (let i = 0; i < NUM_KEYPOINTS; i++) {
      const x = data[i]
      const y = data[NUM_KEYPOINTS + i]
      const z = data[2 * NUM_KEYPOINTS + i]
      keypoints.push(new THREE.Vector3(x, z, y))
    }
    return keypoints
  }, [])

  // Interpolate between previous and current keypoints
  const interpolateKeypoints = useCallback((current, previous, t = 0.3) => {
    if (!previous) return current
    
    return current.map((kp, i) => {
      return new THREE.Vector3().lerpVectors(previous[i], kp, t)
    })
  }, [])

  // Apply spring physics to secondary elements
  const applySpringPhysics = useCallback((keypoints) => {
    const springs = springsRef.current
    
    // Initialize springs if needed
    if (!springs.leftEar) {
      springs.leftEar = new Spring(keypoints[KP.LEFT_EAR])
      springs.rightEar = new Spring(keypoints[KP.RIGHT_EAR])
      springs.tailMid = new Spring(keypoints[KP.TAIL_MID])
      springs.tailTip = new Spring(keypoints[KP.TAIL_TIP])
    }
    
    // Update springs and get smoothed positions
    const smoothedKeypoints = [...keypoints]
    smoothedKeypoints[KP.LEFT_EAR] = springs.leftEar.update(keypoints[KP.LEFT_EAR])
    smoothedKeypoints[KP.RIGHT_EAR] = springs.rightEar.update(keypoints[KP.RIGHT_EAR])
    smoothedKeypoints[KP.TAIL_MID] = springs.tailMid.update(keypoints[KP.TAIL_MID])
    smoothedKeypoints[KP.TAIL_TIP] = springs.tailTip.update(keypoints[KP.TAIL_TIP])
    
    return smoothedKeypoints
  }, [])

  // Build the mouse mesh from keypoints with IK
  const buildMouseMesh = useCallback((rawKeypoints) => {
    const group = new THREE.Group()
    const materials = getMaterials()
    
    // Apply spring physics for secondary motion
    const keypoints = applySpringPhysics(rawKeypoints)
    
    // Calculate body directions for reference
    const spineDir = new THREE.Vector3().subVectors(keypoints[KP.NECK], keypoints[KP.TAIL_BASE]).normalize()
    const upDir = new THREE.Vector3(0, 1, 0)
    const rightDir = new THREE.Vector3().crossVectors(spineDir, upDir).normalize()
    
    // === BODY (main torso) - using smooth spline ===
    const bodyPoints = [
      keypoints[KP.NECK],
      keypoints[KP.SPINE_MID],
      keypoints[KP.SPINE_END],
      keypoints[KP.TAIL_BASE],
    ]
    const bodyCurve = new THREE.CatmullRomCurve3(bodyPoints)
    const bodyGeometry = new THREE.TubeGeometry(bodyCurve, 32, 18, 16, false)
    const bodyMesh = new THREE.Mesh(bodyGeometry, materials.body)
    bodyMesh.castShadow = true
    group.add(bodyMesh)
    
    // === HEAD ===
    const headCenter = new THREE.Vector3().lerpVectors(keypoints[KP.HEAD], keypoints[KP.NOSE], 0.25)
    const headGeometry = new THREE.SphereGeometry(14, 24, 16)
    const headMesh = new THREE.Mesh(headGeometry, materials.body)
    headMesh.position.copy(headCenter)
    headMesh.scale.set(1.1, 0.9, 1.3)
    headMesh.castShadow = true
    group.add(headMesh)
    
    // === SNOUT ===
    const snoutGeometry = new THREE.SphereGeometry(7, 16, 12)
    const snoutMesh = new THREE.Mesh(snoutGeometry, materials.body)
    snoutMesh.position.copy(keypoints[KP.SNOUT])
    snoutMesh.scale.set(0.8, 0.7, 1.2)
    snoutMesh.castShadow = true
    group.add(snoutMesh)
    
    // Nose tip (pink, shiny)
    const noseGeometry = new THREE.SphereGeometry(3.5, 12, 12)
    const noseMesh = new THREE.Mesh(noseGeometry, materials.nose)
    noseMesh.position.copy(keypoints[KP.NOSE])
    noseMesh.castShadow = true
    group.add(noseMesh)
    
    // === EYES ===
    const headToNose = new THREE.Vector3().subVectors(keypoints[KP.NOSE], keypoints[KP.HEAD]).normalize()
    const eyeForwardOffset = headToNose.clone().multiplyScalar(6)
    
    const eyeGeometry = new THREE.SphereGeometry(4, 16, 16)
    
    // Left eye
    const leftEyePos = keypoints[KP.HEAD].clone().add(eyeForwardOffset).addScaledVector(rightDir, -7).addScaledVector(upDir, 3)
    const leftEye = new THREE.Mesh(eyeGeometry, materials.eye)
    leftEye.position.copy(leftEyePos)
    leftEye.scale.set(1, 1.1, 0.9)
    group.add(leftEye)
    
    // Eye highlight
    const highlightGeom = new THREE.SphereGeometry(1.5, 8, 8)
    const highlightMat = new THREE.MeshBasicMaterial({ color: 0xFFFFFF })
    const leftHighlight = new THREE.Mesh(highlightGeom, highlightMat)
    leftHighlight.position.copy(leftEyePos).addScaledVector(headToNose, 2).addScaledVector(upDir, 1.5)
    group.add(leftHighlight)
    
    // Right eye
    const rightEyePos = keypoints[KP.HEAD].clone().add(eyeForwardOffset).addScaledVector(rightDir, 7).addScaledVector(upDir, 3)
    const rightEye = new THREE.Mesh(eyeGeometry, materials.eye)
    rightEye.position.copy(rightEyePos)
    rightEye.scale.set(1, 1.1, 0.9)
    group.add(rightEye)
    
    const rightHighlight = new THREE.Mesh(highlightGeom, highlightMat)
    rightHighlight.position.copy(rightEyePos).addScaledVector(headToNose, 2).addScaledVector(upDir, 1.5)
    group.add(rightHighlight)
    
    // === EARS (with spring physics applied) ===
    const earGeometry = new THREE.CircleGeometry(12, 16)
    
    // Left ear
    const leftEar = new THREE.Mesh(earGeometry, materials.ear)
    leftEar.position.copy(keypoints[KP.LEFT_EAR])
    leftEar.lookAt(keypoints[KP.LEFT_EAR].clone().add(rightDir).addScaledVector(upDir, 0.5))
    leftEar.castShadow = true
    group.add(leftEar)
    
    // Right ear
    const rightEar = new THREE.Mesh(earGeometry, materials.ear)
    rightEar.position.copy(keypoints[KP.RIGHT_EAR])
    rightEar.lookAt(keypoints[KP.RIGHT_EAR].clone().sub(rightDir).addScaledVector(upDir, 0.5))
    rightEar.castShadow = true
    group.add(rightEar)
    
    // === WHISKERS ===
    const whiskers = createWhiskers(keypoints[KP.SNOUT], headToNose, upDir, materials.whisker)
    group.add(whiskers)
    
    // === TAIL (with spring physics applied) ===
    const tailPoints = [
      keypoints[KP.TAIL_BASE],
      keypoints[KP.TAIL_MID],
      keypoints[KP.TAIL_TIP],
    ]
    const tailCurve = new THREE.CatmullRomCurve3(tailPoints)
    
    // Tapered tail using custom geometry
    const tailSegments = 20
    const tailRadialSegments = 8
    const tailGeometry = new THREE.TubeGeometry(tailCurve, tailSegments, 4, tailRadialSegments, false)
    
    // Taper the tail
    const positions = tailGeometry.attributes.position
    for (let i = 0; i < positions.count; i++) {
      const segmentRatio = Math.floor(i / tailRadialSegments) / tailSegments
      const taper = 1 - segmentRatio * 0.8 // Taper from 100% to 20%
      
      const x = positions.getX(i)
      const y = positions.getY(i)
      const z = positions.getZ(i)
      
      const center = tailCurve.getPoint(Math.min(1, segmentRatio))
      const offset = new THREE.Vector3(x - center.x, y - center.y, z - center.z)
      offset.multiplyScalar(taper)
      
      positions.setXYZ(i, center.x + offset.x, center.y + offset.y, center.z + offset.z)
    }
    tailGeometry.attributes.position.needsUpdate = true
    tailGeometry.computeVertexNormals()
    
    const tailMesh = new THREE.Mesh(tailGeometry, materials.tail)
    tailMesh.castShadow = true
    group.add(tailMesh)
    
    // === LIMBS WITH IK ===
    const upperLegLength = 25
    const lowerLegLength = 20
    const upperArmLength = 18
    const lowerArmLength = 15
    
    // Front left leg with IK
    const leftElbowIK = solveIK2Bone(
      keypoints[KP.LEFT_SHOULDER],
      keypoints[KP.LEFT_WRIST],
      upperArmLength,
      lowerArmLength,
      new THREE.Vector3(0, -1, 1).normalize()
    )
    
    const leftUpperArm = createCapsule(keypoints[KP.LEFT_SHOULDER], leftElbowIK, 4.5, materials.body)
    if (leftUpperArm) { leftUpperArm.castShadow = true; group.add(leftUpperArm) }
    
    const leftForearm = createCapsule(leftElbowIK, keypoints[KP.LEFT_WRIST], 3.5, materials.body)
    if (leftForearm) { leftForearm.castShadow = true; group.add(leftForearm) }
    
    const leftPawGeom = new THREE.SphereGeometry(4, 12, 12)
    const leftPaw = new THREE.Mesh(leftPawGeom, materials.paw)
    leftPaw.position.copy(keypoints[KP.LEFT_WRIST])
    leftPaw.scale.set(1, 0.6, 1.2)
    leftPaw.castShadow = true
    group.add(leftPaw)
    
    // Front right leg with IK
    const rightElbowIK = solveIK2Bone(
      keypoints[KP.RIGHT_SHOULDER],
      keypoints[KP.RIGHT_WRIST],
      upperArmLength,
      lowerArmLength,
      new THREE.Vector3(0, -1, 1).normalize()
    )
    
    const rightUpperArm = createCapsule(keypoints[KP.RIGHT_SHOULDER], rightElbowIK, 4.5, materials.body)
    if (rightUpperArm) { rightUpperArm.castShadow = true; group.add(rightUpperArm) }
    
    const rightForearm = createCapsule(rightElbowIK, keypoints[KP.RIGHT_WRIST], 3.5, materials.body)
    if (rightForearm) { rightForearm.castShadow = true; group.add(rightForearm) }
    
    const rightPaw = new THREE.Mesh(leftPawGeom.clone(), materials.paw)
    rightPaw.position.copy(keypoints[KP.RIGHT_WRIST])
    rightPaw.scale.set(1, 0.6, 1.2)
    rightPaw.castShadow = true
    group.add(rightPaw)
    
    // Back left leg with IK
    const leftKneeIK = solveIK2Bone(
      keypoints[KP.LEFT_HIP],
      keypoints[KP.LEFT_ANKLE],
      upperLegLength,
      lowerLegLength,
      new THREE.Vector3(0, 1, 1).normalize()
    )
    
    const leftThigh = createCapsule(keypoints[KP.LEFT_HIP], leftKneeIK, 6, materials.body)
    if (leftThigh) { leftThigh.castShadow = true; group.add(leftThigh) }
    
    const leftShin = createCapsule(leftKneeIK, keypoints[KP.LEFT_ANKLE], 4.5, materials.body)
    if (leftShin) { leftShin.castShadow = true; group.add(leftShin) }
    
    const leftFootGeom = new THREE.SphereGeometry(5, 12, 12)
    const leftFoot = new THREE.Mesh(leftFootGeom, materials.paw)
    leftFoot.position.copy(keypoints[KP.LEFT_ANKLE])
    leftFoot.scale.set(1.3, 0.5, 1.8)
    leftFoot.castShadow = true
    group.add(leftFoot)
    
    // Back right leg with IK
    const rightKneeIK = solveIK2Bone(
      keypoints[KP.RIGHT_HIP],
      keypoints[KP.RIGHT_ANKLE],
      upperLegLength,
      lowerLegLength,
      new THREE.Vector3(0, 1, 1).normalize()
    )
    
    const rightThigh = createCapsule(keypoints[KP.RIGHT_HIP], rightKneeIK, 6, materials.body)
    if (rightThigh) { rightThigh.castShadow = true; group.add(rightThigh) }
    
    const rightShin = createCapsule(rightKneeIK, keypoints[KP.RIGHT_ANKLE], 4.5, materials.body)
    if (rightShin) { rightShin.castShadow = true; group.add(rightShin) }
    
    const rightFoot = new THREE.Mesh(leftFootGeom.clone(), materials.paw)
    rightFoot.position.copy(keypoints[KP.RIGHT_ANKLE])
    rightFoot.scale.set(1.3, 0.5, 1.8)
    rightFoot.castShadow = true
    group.add(rightFoot)
    
    return group
  }, [getMaterials, applySpringPhysics])

  // Initialize Three.js scene
  useEffect(() => {
    if (!containerRef.current) return

    const container = containerRef.current
    let width = container.clientWidth || 800
    let height = container.clientHeight || 600

    // Scene with gradient background
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x1a1a2e)
    sceneRef.current = scene

    // Camera
    const camera = new THREE.PerspectiveCamera(60, width / height, 1, 5000)
    camera.position.set(300, 300, 600)
    cameraRef.current = camera

    // Renderer with better quality
    const renderer = new THREE.WebGLRenderer({ 
      antialias: true,
      alpha: true,
    })
    renderer.setSize(width, height)
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.shadowMap.enabled = true
    renderer.shadowMap.type = THREE.PCFSoftShadowMap
    renderer.toneMapping = THREE.ACESFilmicToneMapping
    renderer.toneMappingExposure = 1.2
    container.appendChild(renderer.domElement)
    rendererRef.current = renderer

    // Controls
    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.dampingFactor = 0.05
    controls.zoomSpeed = 0.3
    controlsRef.current = controls

    const updateZoomFromCamera = () => {
      const distance = camera.position.distanceTo(controls.target)
      const zoom = Math.round((DEFAULT_CAMERA_DISTANCE / distance) * 100)
      setZoomPercentage(Math.max(10, Math.min(500, zoom)))
    }
    controls.addEventListener('change', updateZoomFromCamera)

    // Enhanced lighting for better visuals
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.4)
    scene.add(ambientLight)
    
    // Key light
    const keyLight = new THREE.DirectionalLight(0xffffff, 1.0)
    keyLight.position.set(150, 300, 200)
    keyLight.castShadow = true
    keyLight.shadow.mapSize.width = 2048
    keyLight.shadow.mapSize.height = 2048
    keyLight.shadow.camera.near = 100
    keyLight.shadow.camera.far = 1000
    keyLight.shadow.camera.left = -200
    keyLight.shadow.camera.right = 200
    keyLight.shadow.camera.top = 200
    keyLight.shadow.camera.bottom = -200
    scene.add(keyLight)
    
    // Fill light
    const fillLight = new THREE.DirectionalLight(0x8888ff, 0.3)
    fillLight.position.set(-100, 100, -100)
    scene.add(fillLight)
    
    // Rim light
    const rimLight = new THREE.DirectionalLight(0xffffaa, 0.4)
    rimLight.position.set(-50, 50, -150)
    scene.add(rimLight)

    // Ground plane with shadow
    const groundGeometry = new THREE.PlaneGeometry(1000, 1000)
    const groundMaterial = new THREE.ShadowMaterial({ opacity: 0.3 })
    const ground = new THREE.Mesh(groundGeometry, groundMaterial)
    ground.rotation.x = -Math.PI / 2
    ground.position.y = 0
    ground.receiveShadow = true
    scene.add(ground)

    // Grid
    const gridHelper = new THREE.GridHelper(800, 20, 0x444466, 0x333355)
    gridHelper.position.set(200, 0.1, 100)
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

    // Interpolate with previous frame for smoothness
    const interpolated = interpolateKeypoints(keypoints, prevKeypointsRef.current, 0.5)
    prevKeypointsRef.current = keypoints

    // Remove old mouse mesh
    if (mouseGroupRef.current) {
      scene.remove(mouseGroupRef.current)
      mouseGroupRef.current.traverse((child) => {
        if (child.geometry) child.geometry.dispose()
      })
    }

    // Build new mouse mesh
    const mouseGroup = buildMouseMesh(interpolated)
    scene.add(mouseGroup)
    mouseGroupRef.current = mouseGroup

    // Center camera on mouse
    const centroid = new THREE.Vector3()
    keypoints.forEach(kp => centroid.add(kp))
    centroid.divideScalar(keypoints.length)
    
    if (controlsRef.current) {
      controlsRef.current.target.lerp(centroid, 0.1)
      controlsRef.current.update()
    }

  }, [poseData, parseKeypoints, interpolateKeypoints, buildMouseMesh])

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
