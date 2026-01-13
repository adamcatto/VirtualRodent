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
  body: 0x6B5344,
  bodyLight: 0x8B7355,
  belly: 0xC4A77D,
  ears: 0xE8B4B8,
  earOuter: 0x7A6655,
  nose: 0xFFAAAA,
  eyes: 0x111111,
  eyeHighlight: 0xFFFFFF,
  tail: 0xDDA0A0,
  paws: 0xE8C4C4,
  whiskers: 0x444444,
  fur: 0x7A6655,
  furTip: 0x9A8675,
  eyelid: 0x5A4534,
}

// Physics constants
const SPRING_STIFFNESS = 0.15
const SPRING_DAMPING = 0.7
const WHISKER_STIFFNESS = 0.08
const WHISKER_DAMPING = 0.85

// Fur shell parameters
const FUR_LAYERS = 5
const FUR_LENGTH = 3
const FUR_DENSITY = 0.7

// Ground plane height (for foot IK)
const GROUND_Y = 0

/**
 * 2-bone IK solver
 */
function solveIK2Bone(shoulder, target, upperLength, lowerLength, bendDirection) {
  const shoulderToTarget = new THREE.Vector3().subVectors(target, shoulder)
  const distance = shoulderToTarget.length()
  
  const maxReach = upperLength + lowerLength - 0.1
  const minReach = Math.abs(upperLength - lowerLength) + 0.1
  
  if (distance > maxReach) {
    const dir = shoulderToTarget.normalize()
    return shoulder.clone().addScaledVector(dir, upperLength)
  }
  
  if (distance < minReach) {
    const dir = shoulderToTarget.normalize()
    const perpendicular = new THREE.Vector3().crossVectors(dir, bendDirection).normalize()
    return shoulder.clone().addScaledVector(dir, upperLength * 0.5).addScaledVector(perpendicular, upperLength * 0.8)
  }
  
  const a = upperLength
  const b = lowerLength
  const c = distance
  
  const cosAngle = (a * a + c * c - b * b) / (2 * a * c)
  const angle = Math.acos(Math.max(-1, Math.min(1, cosAngle)))
  
  const dirToTarget = shoulderToTarget.normalize()
  const perpendicular = new THREE.Vector3().crossVectors(dirToTarget, bendDirection).normalize()
  
  const elbow = shoulder.clone()
    .addScaledVector(dirToTarget, Math.cos(angle) * upperLength)
    .addScaledVector(perpendicular, Math.sin(angle) * upperLength)
  
  return elbow
}

/**
 * Foot IK - adjusts foot position to stay on ground
 */
function applyFootIK(footPos, groundY = GROUND_Y) {
  const adjusted = footPos.clone()
  if (adjusted.y < groundY) {
    adjusted.y = groundY
  }
  return adjusted
}

/**
 * Spring class for secondary motion
 */
class Spring {
  constructor(target, stiffness = SPRING_STIFFNESS, damping = SPRING_DAMPING) {
    this.position = target.clone()
    this.velocity = new THREE.Vector3()
    this.target = target.clone()
    this.stiffness = stiffness
    this.damping = damping
  }
  
  update(newTarget) {
    this.target.copy(newTarget)
    const displacement = new THREE.Vector3().subVectors(this.target, this.position)
    const springForce = displacement.multiplyScalar(this.stiffness)
    this.velocity.add(springForce)
    this.velocity.multiplyScalar(this.damping)
    this.position.add(this.velocity)
    return this.position.clone()
  }
}

/**
 * Whisker with spring physics
 */
class WhiskerSpring {
  constructor(basePos, direction, length) {
    this.basePos = basePos.clone()
    this.direction = direction.clone().normalize()
    this.length = length
    this.tipOffset = new THREE.Vector3()
    this.velocity = new THREE.Vector3()
  }
  
  update(newBasePos, newDirection, movementVelocity) {
    this.basePos.copy(newBasePos)
    this.direction.copy(newDirection).normalize()
    
    // Apply movement-based force to whisker tip
    const force = movementVelocity.clone().multiplyScalar(-0.3)
    this.velocity.add(force)
    
    // Spring back to rest position
    const restoring = this.tipOffset.clone().multiplyScalar(-WHISKER_STIFFNESS)
    this.velocity.add(restoring)
    
    // Damping
    this.velocity.multiplyScalar(WHISKER_DAMPING)
    
    // Update offset
    this.tipOffset.add(this.velocity)
    
    // Clamp offset
    const maxOffset = this.length * 0.4
    if (this.tipOffset.length() > maxOffset) {
      this.tipOffset.normalize().multiplyScalar(maxOffset)
    }
    
    // Calculate tip position
    const tip = this.basePos.clone()
      .addScaledVector(this.direction, this.length)
      .add(this.tipOffset)
    
    return { base: this.basePos.clone(), tip }
  }
}

/**
 * Create fur shell layers on a mesh
 */
function createFurShells(baseMesh, layers = FUR_LAYERS, maxLength = FUR_LENGTH) {
  const group = new THREE.Group()
  
  const baseGeometry = baseMesh.geometry.clone()
  const positions = baseGeometry.attributes.position
  const normals = baseGeometry.attributes.normal
  
  if (!normals) {
    baseGeometry.computeVertexNormals()
  }
  
  for (let layer = 0; layer < layers; layer++) {
    const t = (layer + 1) / layers
    const shellGeometry = baseGeometry.clone()
    const shellPositions = shellGeometry.attributes.position
    const shellNormals = shellGeometry.attributes.normal
    
    // Offset vertices along normals
    for (let i = 0; i < shellPositions.count; i++) {
      const nx = shellNormals.getX(i)
      const ny = shellNormals.getY(i)
      const nz = shellNormals.getZ(i)
      
      const offset = t * maxLength
      
      shellPositions.setXYZ(
        i,
        shellPositions.getX(i) + nx * offset,
        shellPositions.getY(i) + ny * offset,
        shellPositions.getZ(i) + nz * offset
      )
    }
    
    shellPositions.needsUpdate = true
    
    // Create material with decreasing opacity for outer shells
    const opacity = 1 - t * FUR_DENSITY
    const color = new THREE.Color(COLORS.fur).lerp(new THREE.Color(COLORS.furTip), t)
    
    const shellMaterial = new THREE.MeshStandardMaterial({
      color: color,
      transparent: true,
      opacity: Math.max(0.1, opacity),
      roughness: 0.9,
      metalness: 0,
      side: THREE.DoubleSide,
      depthWrite: layer === 0,
      alphaTest: 0.1,
    })
    
    const shellMesh = new THREE.Mesh(shellGeometry, shellMaterial)
    shellMesh.castShadow = layer === 0
    group.add(shellMesh)
  }
  
  return group
}

/**
 * Create eyelids that can blink
 */
function createEyelid(eyePos, eyeRadius, direction, upDir) {
  const lidGeometry = new THREE.SphereGeometry(eyeRadius * 1.15, 16, 8, 0, Math.PI * 2, 0, Math.PI * 0.5)
  const lidMaterial = new THREE.MeshStandardMaterial({
    color: COLORS.eyelid,
    roughness: 0.8,
  })
  const lid = new THREE.Mesh(lidGeometry, lidMaterial)
  lid.position.copy(eyePos)
  
  // Orient lid to face outward
  const lookTarget = eyePos.clone().add(direction)
  lid.lookAt(lookTarget)
  lid.rotateX(Math.PI)
  
  return lid
}

/**
 * Create capsule between two points
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
 * MouseMesh - Full-featured 3D mouse with fur, whisker physics, blinking, and foot IK
 */
function MouseMesh({ poseData, title }) {
  const containerRef = useRef(null)
  const sceneRef = useRef(null)
  const rendererRef = useRef(null)
  const cameraRef = useRef(null)
  const controlsRef = useRef(null)
  const mouseGroupRef = useRef(null)
  const [zoomPercentage, setZoomPercentage] = useState(100)
  
  // Animation state
  const animationStateRef = useRef({
    time: 0,
    blinkTimer: 0,
    nextBlinkTime: 2 + Math.random() * 3,
    isBlinking: false,
    blinkProgress: 0,
    leftEyelid: null,
    rightEyelid: null,
  })
  
  // Physics state
  const springsRef = useRef({
    leftEar: null,
    rightEar: null,
    tailMid: null,
    tailTip: null,
    whiskers: [],
  })
  
  const prevKeypointsRef = useRef(null)
  const prevCentroidRef = useRef(null)
  const materialsRef = useRef(null)

  // Initialize materials
  const getMaterials = useCallback(() => {
    if (materialsRef.current) return materialsRef.current
    
    materialsRef.current = {
      body: new THREE.MeshStandardMaterial({ 
        color: COLORS.body,
        roughness: 0.85,
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
        roughness: 0.3,
        metalness: 0.1,
      }),
      eye: new THREE.MeshStandardMaterial({ 
        color: COLORS.eyes, 
        roughness: 0.05,
        metalness: 0.4,
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
      eyelid: new THREE.MeshStandardMaterial({
        color: COLORS.eyelid,
        roughness: 0.8,
      }),
    }
    
    return materialsRef.current
  }, [])

  // Parse keypoints
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

  // Interpolate keypoints
  const interpolateKeypoints = useCallback((current, previous, t = 0.4) => {
    if (!previous) return current
    return current.map((kp, i) => new THREE.Vector3().lerpVectors(previous[i], kp, t))
  }, [])

  // Apply spring physics
  const applySpringPhysics = useCallback((keypoints, movementVelocity) => {
    const springs = springsRef.current
    
    if (!springs.leftEar) {
      springs.leftEar = new Spring(keypoints[KP.LEFT_EAR])
      springs.rightEar = new Spring(keypoints[KP.RIGHT_EAR])
      springs.tailMid = new Spring(keypoints[KP.TAIL_MID], 0.1, 0.75)
      springs.tailTip = new Spring(keypoints[KP.TAIL_TIP], 0.08, 0.8)
      
      // Initialize whiskers (6 per side)
      const headToNose = new THREE.Vector3().subVectors(keypoints[KP.NOSE], keypoints[KP.HEAD]).normalize()
      const upDir = new THREE.Vector3(0, 1, 0)
      const rightDir = new THREE.Vector3().crossVectors(headToNose, upDir).normalize()
      
      for (let i = 0; i < 6; i++) {
        const angle = ((i - 2.5) / 5) * Math.PI * 0.4
        const leftDir = rightDir.clone().multiplyScalar(-1)
          .applyAxisAngle(headToNose, angle)
          .addScaledVector(headToNose, 0.3)
          .normalize()
        const rightDirW = rightDir.clone()
          .applyAxisAngle(headToNose, -angle)
          .addScaledVector(headToNose, 0.3)
          .normalize()
        
        springs.whiskers.push(new WhiskerSpring(keypoints[KP.SNOUT], leftDir, 20 + Math.random() * 5))
        springs.whiskers.push(new WhiskerSpring(keypoints[KP.SNOUT], rightDirW, 20 + Math.random() * 5))
      }
    }
    
    const smoothed = [...keypoints]
    smoothed[KP.LEFT_EAR] = springs.leftEar.update(keypoints[KP.LEFT_EAR])
    smoothed[KP.RIGHT_EAR] = springs.rightEar.update(keypoints[KP.RIGHT_EAR])
    smoothed[KP.TAIL_MID] = springs.tailMid.update(keypoints[KP.TAIL_MID])
    smoothed[KP.TAIL_TIP] = springs.tailTip.update(keypoints[KP.TAIL_TIP])
    
    // Apply foot IK - keep feet on ground
    smoothed[KP.LEFT_ANKLE] = applyFootIK(keypoints[KP.LEFT_ANKLE])
    smoothed[KP.RIGHT_ANKLE] = applyFootIK(keypoints[KP.RIGHT_ANKLE])
    smoothed[KP.LEFT_WRIST] = applyFootIK(keypoints[KP.LEFT_WRIST])
    smoothed[KP.RIGHT_WRIST] = applyFootIK(keypoints[KP.RIGHT_WRIST])
    
    return smoothed
  }, [])

  // Build mouse mesh
  const buildMouseMesh = useCallback((rawKeypoints, movementVelocity) => {
    const group = new THREE.Group()
    const materials = getMaterials()
    const animState = animationStateRef.current
    
    const keypoints = applySpringPhysics(rawKeypoints, movementVelocity)
    
    const spineDir = new THREE.Vector3().subVectors(keypoints[KP.NECK], keypoints[KP.TAIL_BASE]).normalize()
    const upDir = new THREE.Vector3(0, 1, 0)
    const rightDir = new THREE.Vector3().crossVectors(spineDir, upDir).normalize()
    
    // === BODY with fur shells ===
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
    
    // Add fur shells to body
    const furShells = createFurShells(bodyMesh, FUR_LAYERS, FUR_LENGTH)
    group.add(furShells)
    group.add(bodyMesh)
    
    // === HEAD with fur ===
    const headCenter = new THREE.Vector3().lerpVectors(keypoints[KP.HEAD], keypoints[KP.NOSE], 0.25)
    const headGeometry = new THREE.SphereGeometry(14, 24, 16)
    const headMesh = new THREE.Mesh(headGeometry, materials.body)
    headMesh.position.copy(headCenter)
    headMesh.scale.set(1.1, 0.9, 1.3)
    headMesh.castShadow = true
    
    const headFur = createFurShells(headMesh, 3, 2)
    headFur.position.copy(headCenter)
    headFur.scale.set(1.1, 0.9, 1.3)
    group.add(headFur)
    group.add(headMesh)
    
    // === SNOUT ===
    const snoutGeometry = new THREE.SphereGeometry(7, 16, 12)
    const snoutMesh = new THREE.Mesh(snoutGeometry, materials.body)
    snoutMesh.position.copy(keypoints[KP.SNOUT])
    snoutMesh.scale.set(0.8, 0.7, 1.2)
    snoutMesh.castShadow = true
    group.add(snoutMesh)
    
    // Nose
    const noseGeometry = new THREE.SphereGeometry(3.5, 12, 12)
    const noseMesh = new THREE.Mesh(noseGeometry, materials.nose)
    noseMesh.position.copy(keypoints[KP.NOSE])
    noseMesh.castShadow = true
    group.add(noseMesh)
    
    // === EYES with blinking eyelids ===
    const headToNose = new THREE.Vector3().subVectors(keypoints[KP.NOSE], keypoints[KP.HEAD]).normalize()
    const eyeForwardOffset = headToNose.clone().multiplyScalar(6)
    const eyeRadius = 4
    
    const eyeGeometry = new THREE.SphereGeometry(eyeRadius, 16, 16)
    
    // Left eye
    const leftEyePos = keypoints[KP.HEAD].clone().add(eyeForwardOffset).addScaledVector(rightDir, -7).addScaledVector(upDir, 3)
    const leftEye = new THREE.Mesh(eyeGeometry, materials.eye)
    leftEye.position.copy(leftEyePos)
    leftEye.scale.set(1, 1.1, 0.9)
    group.add(leftEye)
    
    // Left eyelid
    const leftLid = createEyelid(leftEyePos, eyeRadius, headToNose, upDir)
    leftLid.rotation.x = Math.PI * (1 - animState.blinkProgress * 0.5) // Animate blink
    animState.leftEyelid = leftLid
    group.add(leftLid)
    
    // Eye highlights
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
    
    // Right eyelid
    const rightLid = createEyelid(rightEyePos, eyeRadius, headToNose, upDir)
    rightLid.rotation.x = Math.PI * (1 - animState.blinkProgress * 0.5)
    animState.rightEyelid = rightLid
    group.add(rightLid)
    
    const rightHighlight = new THREE.Mesh(highlightGeom, highlightMat)
    rightHighlight.position.copy(rightEyePos).addScaledVector(headToNose, 2).addScaledVector(upDir, 1.5)
    group.add(rightHighlight)
    
    // === EARS ===
    const earGeometry = new THREE.CircleGeometry(12, 16)
    
    const leftEar = new THREE.Mesh(earGeometry, materials.ear)
    leftEar.position.copy(keypoints[KP.LEFT_EAR])
    leftEar.lookAt(keypoints[KP.LEFT_EAR].clone().add(rightDir).addScaledVector(upDir, 0.5))
    leftEar.castShadow = true
    group.add(leftEar)
    
    const rightEar = new THREE.Mesh(earGeometry, materials.ear)
    rightEar.position.copy(keypoints[KP.RIGHT_EAR])
    rightEar.lookAt(keypoints[KP.RIGHT_EAR].clone().sub(rightDir).addScaledVector(upDir, 0.5))
    rightEar.castShadow = true
    group.add(rightEar)
    
    // === WHISKERS - attached directly around the nose (pink ball) ===
    const whiskerGroup = new THREE.Group()
    
    // Whiskers emanate from directly around the nose position
    const nosePos = keypoints[KP.NOSE]
    
    // 4 whiskers per side, arranged around the nose
    const whiskerConfigs = [
      { vOffset: -2, hOffset: 3, length: 22, angleUp: -0.2, angleForward: 0.3 },
      { vOffset: 0, hOffset: 3.5, length: 25, angleUp: 0, angleForward: 0.2 },
      { vOffset: 2, hOffset: 3.5, length: 25, angleUp: 0.1, angleForward: 0.2 },
      { vOffset: 4, hOffset: 3, length: 20, angleUp: 0.3, angleForward: 0.3 },
    ]
    
    whiskerConfigs.forEach(config => {
      // Left whisker
      const leftBase = nosePos.clone()
        .addScaledVector(rightDir, -config.hOffset)
        .addScaledVector(upDir, config.vOffset)
      
      const leftDir = rightDir.clone().multiplyScalar(-1)
        .addScaledVector(headToNose, config.angleForward)
        .addScaledVector(upDir, config.angleUp)
        .normalize()
      
      const leftGeom = new THREE.BufferGeometry().setFromPoints([
        leftBase,
        leftBase.clone().addScaledVector(leftDir, config.length)
      ])
      whiskerGroup.add(new THREE.Line(leftGeom, materials.whisker))
      
      // Right whisker (mirror)
      const rightBase = nosePos.clone()
        .addScaledVector(rightDir, config.hOffset)
        .addScaledVector(upDir, config.vOffset)
      
      const rightDirW = rightDir.clone()
        .addScaledVector(headToNose, config.angleForward)
        .addScaledVector(upDir, config.angleUp)
        .normalize()
      
      const rightGeom = new THREE.BufferGeometry().setFromPoints([
        rightBase,
        rightBase.clone().addScaledVector(rightDirW, config.length)
      ])
      whiskerGroup.add(new THREE.Line(rightGeom, materials.whisker))
    })
    
    group.add(whiskerGroup)
    
    // === TAIL ===
    const tailPoints = [
      keypoints[KP.TAIL_BASE],
      keypoints[KP.TAIL_MID],
      keypoints[KP.TAIL_TIP],
    ]
    const tailCurve = new THREE.CatmullRomCurve3(tailPoints)
    const tailGeometry = new THREE.TubeGeometry(tailCurve, 20, 4, 8, false)
    
    // Taper tail
    const tailPositions = tailGeometry.attributes.position
    for (let i = 0; i < tailPositions.count; i++) {
      const segmentRatio = Math.floor(i / 8) / 20
      const taper = 1 - segmentRatio * 0.85
      const center = tailCurve.getPoint(Math.min(1, segmentRatio))
      const x = tailPositions.getX(i)
      const y = tailPositions.getY(i)
      const z = tailPositions.getZ(i)
      const offset = new THREE.Vector3(x - center.x, y - center.y, z - center.z).multiplyScalar(taper)
      tailPositions.setXYZ(i, center.x + offset.x, center.y + offset.y, center.z + offset.z)
    }
    tailGeometry.attributes.position.needsUpdate = true
    tailGeometry.computeVertexNormals()
    
    const tailMesh = new THREE.Mesh(tailGeometry, materials.tail)
    tailMesh.castShadow = true
    group.add(tailMesh)
    
    // === LIMBS with IK ===
    const upperLegLength = 25
    const lowerLegLength = 20
    const upperArmLength = 18
    const lowerArmLength = 15
    
    // Front left leg
    const leftElbowIK = solveIK2Bone(
      keypoints[KP.LEFT_SHOULDER],
      keypoints[KP.LEFT_WRIST],
      upperArmLength, lowerArmLength,
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
    
    // Front right leg
    const rightElbowIK = solveIK2Bone(
      keypoints[KP.RIGHT_SHOULDER],
      keypoints[KP.RIGHT_WRIST],
      upperArmLength, lowerArmLength,
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
    
    // Back left leg
    const leftKneeIK = solveIK2Bone(
      keypoints[KP.LEFT_HIP],
      keypoints[KP.LEFT_ANKLE],
      upperLegLength, lowerLegLength,
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
    
    // Back right leg
    const rightKneeIK = solveIK2Bone(
      keypoints[KP.RIGHT_HIP],
      keypoints[KP.RIGHT_ANKLE],
      upperLegLength, lowerLegLength,
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

  // Initialize scene
  useEffect(() => {
    if (!containerRef.current) return

    const container = containerRef.current
    let width = container.clientWidth || 800
    let height = container.clientHeight || 600

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x1a1a2e)
    sceneRef.current = scene

    const camera = new THREE.PerspectiveCamera(60, width / height, 1, 5000)
    camera.position.set(300, 300, 600)
    cameraRef.current = camera

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
    renderer.setSize(width, height)
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.shadowMap.enabled = true
    renderer.shadowMap.type = THREE.PCFSoftShadowMap
    renderer.toneMapping = THREE.ACESFilmicToneMapping
    renderer.toneMappingExposure = 1.2
    container.appendChild(renderer.domElement)
    rendererRef.current = renderer

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

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.4)
    scene.add(ambientLight)
    
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
    
    const fillLight = new THREE.DirectionalLight(0x8888ff, 0.3)
    fillLight.position.set(-100, 100, -100)
    scene.add(fillLight)
    
    const rimLight = new THREE.DirectionalLight(0xffffaa, 0.4)
    rimLight.position.set(-50, 50, -150)
    scene.add(rimLight)

    // Ground
    const groundGeometry = new THREE.PlaneGeometry(1000, 1000)
    const groundMaterial = new THREE.ShadowMaterial({ opacity: 0.3 })
    const ground = new THREE.Mesh(groundGeometry, groundMaterial)
    ground.rotation.x = -Math.PI / 2
    ground.position.y = GROUND_Y
    ground.receiveShadow = true
    scene.add(ground)

    const gridHelper = new THREE.GridHelper(800, 20, 0x444466, 0x333355)
    gridHelper.position.set(200, 0.1, 100)
    scene.add(gridHelper)

    const axesHelper = new THREE.AxesHelper(200)
    scene.add(axesHelper)

    // Animation loop with blinking
    const animState = animationStateRef.current
    let lastTime = performance.now()
    
    const animate = () => {
      requestAnimationFrame(animate)
      
      const now = performance.now()
      const dt = (now - lastTime) / 1000
      lastTime = now
      
      animState.time += dt
      
      // Blinking logic
      animState.blinkTimer += dt
      if (!animState.isBlinking && animState.blinkTimer >= animState.nextBlinkTime) {
        animState.isBlinking = true
        animState.blinkProgress = 0
      }
      
      if (animState.isBlinking) {
        animState.blinkProgress += dt * 8 // Blink speed
        if (animState.blinkProgress >= 1) {
          animState.blinkProgress = 0
          animState.isBlinking = false
          animState.blinkTimer = 0
          animState.nextBlinkTime = 2 + Math.random() * 4
        }
      }
      
      // Update eyelids if they exist
      if (animState.leftEyelid && animState.rightEyelid) {
        const blinkAngle = animState.isBlinking 
          ? Math.sin(animState.blinkProgress * Math.PI) * Math.PI * 0.5
          : 0
        animState.leftEyelid.rotation.x = Math.PI - blinkAngle
        animState.rightEyelid.rotation.x = Math.PI - blinkAngle
      }
      
      controls.update()
      renderer.render(scene, camera)
    }
    animate()

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

  // Update mesh
  useEffect(() => {
    if (!sceneRef.current || !poseData || poseData.length === 0) return

    const scene = sceneRef.current
    const keypoints = parseKeypoints(poseData)
    
    if (!keypoints) return

    const interpolated = interpolateKeypoints(keypoints, prevKeypointsRef.current, 0.5)
    
    // Calculate movement velocity for whisker physics
    const centroid = new THREE.Vector3()
    keypoints.forEach(kp => centroid.add(kp))
    centroid.divideScalar(keypoints.length)
    
    const movementVelocity = prevCentroidRef.current 
      ? new THREE.Vector3().subVectors(centroid, prevCentroidRef.current)
      : new THREE.Vector3()
    
    prevKeypointsRef.current = keypoints
    prevCentroidRef.current = centroid.clone()

    if (mouseGroupRef.current) {
      scene.remove(mouseGroupRef.current)
      mouseGroupRef.current.traverse((child) => {
        if (child.geometry) child.geometry.dispose()
      })
    }

    const mouseGroup = buildMouseMesh(interpolated, movementVelocity)
    scene.add(mouseGroup)
    mouseGroupRef.current = mouseGroup

    if (controlsRef.current) {
      controlsRef.current.target.lerp(centroid, 0.1)
      controlsRef.current.update()
    }

  }, [poseData, parseKeypoints, interpolateKeypoints, buildMouseMesh])

  // Zoom handlers
  const handleZoomIn = useCallback(() => {
    if (!cameraRef.current || !controlsRef.current) return
    const direction = new THREE.Vector3()
    cameraRef.current.getWorldDirection(direction)
    cameraRef.current.position.addScaledVector(direction, 50)
    controlsRef.current.update()
  }, [])

  const handleZoomOut = useCallback(() => {
    if (!cameraRef.current || !controlsRef.current) return
    const direction = new THREE.Vector3()
    cameraRef.current.getWorldDirection(direction)
    cameraRef.current.position.addScaledVector(direction, -50)
    controlsRef.current.update()
  }, [])

  const handleZoomReset = useCallback(() => {
    if (!cameraRef.current || !controlsRef.current) return
    cameraRef.current.position.set(300, 300, DEFAULT_CAMERA_DISTANCE)
    controlsRef.current.update()
    setZoomPercentage(100)
  }, [])

  const handleZoomSlider = useCallback((e) => {
    if (!cameraRef.current || !controlsRef.current) return
    const newZoom = parseInt(e.target.value)
    const newDistance = DEFAULT_CAMERA_DISTANCE / (newZoom / 100)
    const direction = new THREE.Vector3()
    direction.subVectors(cameraRef.current.position, controlsRef.current.target).normalize()
    cameraRef.current.position.copy(controlsRef.current.target).addScaledVector(direction, newDistance)
    controlsRef.current.update()
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
