# Dataset Analysis & Biomechanical Findings (REHAB24-6)

## 1. Dataset Overview & Configuration
* **Input Modality:** 2D skeleton joint coordinates at 30 FPS (`2d_joints`).
* **Frame Dimensions & Native Camera Setup:**
  * **Camera 17 (C17):** Horizontal orientation (landscape, ~1920x1080).
  * **Camera 18 (C18):** Vertical orientation (portrait, ~1080x1920).
* **Kinematic Structure:** 26 internal OptiTrack joints with a defined anatomical hierarchy and 41 surface markers.

## 2. Temporal Characteristics & Slicing
* **Segmentation:** Bound by `first_frame` and `last_frame` from `Segmentation.csv`.
* **Repetition Durations:** Significant variance across subjects.
* **Data Cleaning:** Repetitions with `mocap_erroneous == 1` are excluded to eliminate motion capture tracking artifacts.
* **Standardization Decision:** Selected temporal resampling (1D linear interpolation to a uniform 100 frames) over zero-padding to guarantee speed invariance across varying execution tempos.

## 3. Camera Orientation Strategy per Exercise
Different movement planes require dedicated camera viewpoints to prevent depth occlusions:
* **Sagittal Plane Movements (Side View / C18):** Exercise 5 (Lunges), Exercise 6 (Squats), Exercise 3 (Push-ups).
* **Frontal Plane Movements (Front View / C17):** Exercise 1 (Arm Abduction), Exercise 4 (Leg Abduction), Exercise 2 (Arm VW).

## 4. Production Domain Adaptation (OptiTrack vs. MediaPipe)
* **The Domain Shift:** OptiTrack provides 26 joints (including mid-spine and neck), whereas real-world MediaPipe provides 33 landmarks (lacking explicit mid-spine points).
* **Strategy:** Downscale OptiTrack to a 15-joint "Common Minimum Skeleton" rather than attempting to synthesize/impute missing spine coordinates, preventing cascading prediction error.

## 5. Model Architecture Roadmap
Dedicated binary classification models per exercise evaluated across 4 benchmark architectures:
1. **Pure Neural Network:** Flat spatial baseline with global pooling.
2. **Pure Recurrent Network (LSTM/GRU):** End-to-end temporal tracking.
3. **Hierarchical Hybrid (TimeDistributed Dense + LSTM):** Decoupled spatial feature projection and temporal tracking.
4. **Spatio-Temporal Transformer Encoder:** Self-attention across resampled time steps.
