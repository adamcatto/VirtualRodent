# Plan: Animated Mouse Model from Pose Keypoints

## Current State
- 23 keypoints with x, y, z coordinates at 50Hz
- Keypoints cover: nose, head, neck, spine, tail, 4 legs (shoulder/elbow/wrist, hip/knee/ankle), ears, snout
- Basic convex hull rendering provides a rough body shape
- Skeleton view shows joint connections

## Goal
Create a realistic, animated 3D mouse that follows the pose keypoints in real-time.

---

## Approach Options

### Option A: Rigged 3D Model with Inverse Kinematics (Recommended)
**Complexity: Medium-High | Visual Quality: High**

1. **Obtain/Create Mouse 3D Model**
   - Source a rigged mouse model (GLTF/GLB format for Three.js)
   - Options:
     - Sketchfab (many free CC models)
     - Blender's free animal models
     - Generate with AI (e.g., Meshy.ai, Luma AI)
     - Create manually in Blender

2. **Set Up Armature Matching**
   - Map our 23 keypoints to the model's bone hierarchy
   - Key mappings:
     ```
     Keypoint 0 (nose)      → head_tip bone
     Keypoint 1 (head)      → head bone
     Keypoint 2 (neck)      → neck bone
     Keypoint 3 (spine_mid) → spine_mid bone
     Keypoint 4 (spine_end) → spine_base bone
     Keypoints 5-7 (tail)   → tail chain
     Keypoints 8-10         → front_left_arm chain
     Keypoints 11-13        → front_right_arm chain
     Keypoints 14-16        → back_left_leg chain
     Keypoints 17-19        → back_right_leg chain
     Keypoints 20-21 (ears) → ear bones (if available)
     ```

3. **Implementation Steps**
   ```javascript
   // Pseudocode
   import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader'
   
   // Load model
   const model = await loader.loadAsync('/models/mouse.glb')
   
   // Get bone references
   const bones = {
     spine: model.getObjectByName('Spine'),
     head: model.getObjectByName('Head'),
     // ... etc
   }
   
   // Per-frame update
   function updatePose(keypoints) {
     // Option 1: Direct bone positioning
     bones.head.position.copy(keypoints[1])
     
     // Option 2: IK solving (better for limbs)
     ikSolver.solve(bones.frontLeftArm, keypoints[10])  // target wrist position
   }
   ```

4. **Libraries to Use**
   - `three/examples/jsm/loaders/GLTFLoader` - Load rigged models
   - `three/examples/jsm/animation/CCDIKSolver` - Inverse kinematics
   - Optionally: `three-ik` npm package for more advanced IK

---

### Option B: Procedural Mesh Deformation
**Complexity: High | Visual Quality: Medium-High**

1. **Create Base Mouse Mesh Programmatically**
   - Use subdivision surfaces or SDF (Signed Distance Fields)
   - Define base shape as ellipsoid body + head + limbs

2. **Deform Mesh Based on Keypoints**
   ```javascript
   // Vertex skinning - each vertex influenced by nearby keypoints
   function deformMesh(vertices, keypoints) {
     vertices.forEach((vertex, i) => {
       const weights = computeSkinWeights(vertex, keypoints)
       vertex.position = blendPositions(keypoints, weights)
     })
   }
   ```

3. **Add Smooth Interpolation**
   - Use spline interpolation along spine/tail
   - Catmull-Rom splines for smooth curves

---

### Option C: Sprite/Billboard Approach (Quick Win)
**Complexity: Low | Visual Quality: Low-Medium**

1. **Use 2D Mouse Sprite**
   - Create mouse sprite facing different directions
   - Billboard sprite always faces camera

2. **Scale/Rotate Based on Pose**
   - Derive orientation from spine direction
   - Scale based on distance to camera

---

## Recommended Implementation Path

### Phase 1: Enhanced Convex Hull (Current + Improvements) ✓
- [x] Body convex hull from torso keypoints
- [x] Head convex hull
- [x] Tube limbs with tapering
- [x] Eyes, ears, nose details
- [ ] Smooth mesh using subdivision modifier
- [ ] Add fur texture/normal map

### Phase 2: Simple Rigged Model (Next Step)
1. **Get Model**
   - Download mouse GLTF from Sketchfab (e.g., "low poly mouse rigged")
   - Or create simple one in Blender with basic armature

2. **Implement Bone Mapping**
   ```javascript
   // In PoseViewer3D.jsx
   const BONE_TO_KEYPOINT = {
     'Spine': 3,
     'Spine.001': 4,  
     'Neck': 2,
     'Head': 1,
     // ... map all bones
   }
   
   function updateModelPose(model, keypoints) {
     Object.entries(BONE_TO_KEYPOINT).forEach(([boneName, kpIndex]) => {
       const bone = model.getObjectByName(boneName)
       if (bone) {
         bone.position.copy(keypoints[kpIndex])
       }
     })
   }
   ```

3. **Add IK for Limbs**
   - Limbs should reach their target positions naturally
   - Use Three.js CCDIKSolver or implement simple FABRIK

### Phase 3: Advanced Features (Future)
- **Fur Rendering**: Use shell texturing or geometry shaders
- **Secondary Motion**: Ear/tail physics with spring simulation
- **Facial Animation**: Blend shapes for whisker twitching, blinking
- **Ground Contact**: IK foot placement on terrain

---

## File Structure Proposal

```
src/website/frontend/
├── src/
│   ├── components/
│   │   ├── PoseViewer3D.jsx       # Current viewer
│   │   ├── MouseModel.jsx          # NEW: Rigged model component
│   │   └── MouseModelLoader.jsx    # NEW: GLTF loading logic
│   ├── models/
│   │   └── mouse.glb               # NEW: 3D mouse model
│   ├── utils/
│   │   ├── ikSolver.js             # NEW: IK implementation
│   │   └── boneMapping.js          # NEW: Keypoint to bone mapping
```

---

## Resources

### Free Mouse 3D Models
- Sketchfab: https://sketchfab.com/search?q=mouse+rigged&type=models
- TurboSquid free section
- Blender's demo files

### Three.js Animation Docs
- Skeletal Animation: https://threejs.org/docs/#manual/en/introduction/Animation-system
- IK Solver: https://threejs.org/docs/#examples/en/animations/CCDIKSolver
- GLTF Loader: https://threejs.org/docs/#examples/en/loaders/GLTFLoader

### Tutorials
- "Rigging a character in Blender" (YouTube)
- Three.js Journey - Character Animation chapter
- "Procedural Animation in Three.js" blog posts

---

## Quick Start: Adding a Rigged Model

```bash
# 1. Download a mouse model (example from Sketchfab)
# Save as: src/website/frontend/public/models/mouse.glb

# 2. Install dependencies (if not already present)
npm install three  # Already installed

# 3. Create MouseModel component (see Phase 2 above)
```

---

## Timeline Estimate

| Phase | Effort | Time |
|-------|--------|------|
| Phase 1 (Current improvements) | Low | 2-4 hours |
| Phase 2 (Basic rigged model) | Medium | 1-2 days |
| Phase 3 (IK + polish) | High | 3-5 days |
| Phase 4 (Fur, physics) | Very High | 1-2 weeks |

---

## Decision Points

1. **Model Source**: Custom Blender model vs. downloaded asset?
   - Downloaded is faster but may need retopology/re-rigging
   - Custom gives full control but requires Blender skills

2. **IK Approach**: CCDIKSolver vs. custom FABRIK vs. direct positioning?
   - CCDIKSolver: Built into Three.js, good for simple chains
   - FABRIK: Better for multiple end-effectors, need to implement
   - Direct: Simplest but may look unnatural

3. **Performance Target**: Real-time 60fps vs. 30fps acceptable?
   - Affects mesh complexity, IK iterations, fur detail

---

## Completed Implementation

### Phase 1: Enhanced Convex Hull ✅
- [x] Body convex hull from torso keypoints
- [x] Head convex hull  
- [x] Tube limbs with tapering
- [x] Eyes, ears, nose details
- [x] Render mode toggle (skeleton/mesh/both)

### Phase 2: Procedural Mouse Mesh ✅
- [x] Smooth tube body using CatmullRom splines
- [x] Proper head, snout, ears geometry
- [x] Eyes with highlights
- [x] Capsule-based limbs
- [x] Tapered tail

### Phase 3: IK + Polish ✅
- [x] 2-bone IK solver for limbs (arms and legs)
- [x] Spring physics for secondary motion (ears, tail)
- [x] Frame interpolation for smoother animation
- [x] Whiskers
- [x] Enhanced materials (MeshStandardMaterial)
- [x] 3-point lighting (key, fill, rim)
- [x] Shadow casting
- [x] ACES filmic tone mapping

## Future Enhancements (Phase 4)
- [ ] Fur rendering using shell texturing
- [ ] Physically-based animation for whiskers
- [ ] Eyelid blinking animation
- [ ] Ground contact / foot IK
- [ ] Muscle deformation on body

