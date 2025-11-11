# XRT Mocap-Driven Policy Playback for T1 Robot

Complete guide for using XRT motion capture with TWIST T1 student policies in Isaac Gym.

---

## Table of Contents
- [Quick Start](#quick-start)
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Shell Script Helper](#shell-script-helper)
- [Command-Line Reference](#command-line-reference)
- [Output Files](#output-files)
- [Observation Structure](#observation-structure)
- [Troubleshooting](#troubleshooting)
- [Technical Details](#technical-details)
- [Examples](#examples)

---

## Quick Start

### Prerequisites Checklist
- [ ] TWIST environment installed and working
- [ ] GMR library: `cd /home/kelvin/code/GMR && pip install -e .`
- [ ] XRoboToolkit SDK: `pip install xrobotoolkit-sdk`
- [ ] T1 student policy checkpoint in `logs/t1_student/1/model_*.pt`
- [ ] (Optional) XRT headset for real-time streaming

### 3-Step Quick Test

**Step 1: Validate Installation**
```bash
cd /home/kelvin/code/TWIST/legged_gym/legged_gym/scripts
python test_xrt_obs.py
# Expected: ✅ All dimension tests passed!
```

**Step 2: Record XRT Motion (Optional)**
```bash
cd /home/kelvin/code/GMR/scripts
python xrt_to_robot_realtime.py \
    --robot booster_t1 \
    --save_path ../data/test_motion.pkl \
    --no_visualize
# Press Ctrl+C after a few seconds
```

**Step 3: Run Policy**
```bash
cd /home/kelvin/code/TWIST

# Using shell script (easiest)
./play_xrt.sh realtime --cpu

# Or direct Python call
cd legged_gym/legged_gym/scripts
python play_xrt.py \
    --task t1_stu_rl \
    --proj_name t1_student \
    --exptid 1 \
    --checkpoint 10000 \
    --mode realtime \
    --sim_device cpu --rl_device cpu
```

---

## Overview

`play_xrt.py` enables real-time policy playback using XRT motion capture data as reference motion for the T1 humanoid robot in Isaac Gym simulation.

### Integration Components
- **XRT Motion Capture**: Real-time human motion tracking via XRoboToolkit
- **GMR Retargeting**: Converts human motion to T1 robot joint angles
- **TWIST Policy**: Executes trained student policy with retargeted motion as reference
- **Isaac Gym**: Physics simulation and visualization

### Architecture

```
┌─────────────────────┐
│  XRT Headset        │  Real-time mocap OR pre-recorded file
└──────────┬──────────┘
           │ 24 body joints [pos, quat]
           ▼
┌─────────────────────┐
│  GMR Retargeting    │  Human → T1 joint mapping
└──────────┬──────────┘
           │ qpos [3 root + 4 quat + 27 joints]
           ▼
┌─────────────────────┐
│  Observation Builder│  Constructs 1331-dim observation:
│                     │  • mimic_obs (35): reference motion
│                     │  • proprio_obs (86): robot state
│                     │  • history (10 timesteps)
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Student Policy     │  JIT or regular PyTorch model
└──────────┬──────────┘
           │ 27 action dims
           ▼
┌─────────────────────┐
│  Isaac Gym Sim      │  Physics @ 50Hz control
└─────────────────────┘
```

---

## Installation

### 1. GMR Library
```bash
cd /home/kelvin/code/GMR
pip install -e .
```

### 2. XRoboToolkit SDK
```bash
pip install xrobotoolkit-sdk
```

### 3. Additional Dependencies
```bash
pip install scipy imageio termcolor
```

### 4. Verify Installation
```bash
cd /home/kelvin/code/TWIST/legged_gym/legged_gym/scripts
python test_xrt_obs.py
```

---

## Usage

### Mode 1: Real-Time XRT Streaming

Connect to XRT headset and stream motion capture in real-time:

```bash
cd /home/kelvin/code/TWIST/legged_gym/legged_gym/scripts

# Basic real-time streaming
python play_xrt.py \
    --task t1_stu_rl \
    --proj_name t1_student \
    --exptid 1 \
    --checkpoint 10000 \
    --mode realtime

# With JIT model and recording
python play_xrt.py \
    --task t1_stu_rl \
    --proj_name t1_student \
    --exptid 1 \
    --checkpoint 10000 \
    --mode realtime \
    --use_jit \
    --record_video \
    --record_log

# Specify human height manually
python play_xrt.py \
    --task t1_stu_rl \
    --proj_name t1_student \
    --exptid 1 \
    --mode realtime \
    --human_height 1.75
```

**Requirements:**
- XRT headset connected and tracking
- XRoboToolkit service running
- Person wearing headset in tracking volume

### Mode 2: File Replay

Replay from pre-recorded XRT motion capture file:

```bash
# Basic file replay
python play_xrt.py \
    --task t1_stu_rl \
    --proj_name t1_student \
    --exptid 1 \
    --mode file \
    --xrt_file /path/to/recording.pkl

# With custom settings
python play_xrt.py \
    --task t1_stu_rl \
    --proj_name t1_student \
    --exptid 1 \
    --mode file \
    --xrt_file /path/to/recording.pkl \
    --use_jit \
    --record_video \
    --episode_length 30.0
```

**To record XRT data:**
```bash
cd /home/kelvin/code/GMR/scripts
python xrt_to_robot_realtime.py \
    --robot booster_t1 \
    --save_path ../data/my_motion.pkl \
    --no_visualize
```

---

## Shell Script Helper

For convenience, use the provided shell script `play_xrt.sh`:

```bash
cd /home/kelvin/code/TWIST

# Real-time streaming
./play_xrt.sh realtime

# File replay
./play_xrt.sh file /path/to/motion.pkl

# With options
./play_xrt.sh realtime --record --cpu
./play_xrt.sh realtime --headless --use_jit
./play_xrt.sh file /path/to/motion.pkl --record
```

**Available flags:**
- `--record` / `-r`: Enable video and log recording
- `--headless` / `-h`: Run without viewer
- `--use_jit` / `-j`: Use JIT traced model
- `--cpu` / `-c`: Run on CPU (for RTX 5090 compatibility)

**Configuration** (edit in `play_xrt.sh`):
```bash
proj_name="t1_student"    # Your project name
exptid="1"                # Your experiment ID
checkpoint=10000          # Checkpoint number
task_name="t1_stu_rl"     # Task name
```

---

## Command-Line Reference

### Required Arguments
```
--proj_name       Project name (matches logs/{proj_name}/)
--exptid          Experiment ID (matches logs/{proj_name}/{exptid}/)
```

### Task Configuration
```
--task            Task name (default: t1_stu_rl)
--checkpoint      Checkpoint number (default: -1 for latest)
--use_jit         Use JIT traced model for faster inference
```

### XRT Mocap Settings
```
--mode            XRT source mode: 'realtime' or 'file' (default: realtime)
--xrt_file        Path to XRT pickle file (required if mode=file)
--human_height    Human height in meters (auto-estimated if not provided)
```

### Environment Settings
```
--num_envs        Number of parallel environments (default: 1)
--episode_length  Episode length in seconds (default: 60.0)
--headless        Run without visualization
--sim_device      Physics device: 'cuda:0' or 'cpu' (default: cuda:0)
--rl_device       RL device: 'cuda:0' or 'cpu' (default: cuda:0)
```

### Recording Options
```
--record_video    Save MP4 video of simulation
--record_log      Save JSON log of states and actions
```

### All Arguments
See output of `python play_xrt.py --help` for complete list including Isaac Gym physics parameters.

---

## Output Files

### Video Recording
When `--record_video` is enabled:
```
logs/videos_xrt/{run_name}/{proj_name}-{exptid}-xrt-env0.mp4
```

### JSON Logs
When `--record_log` is enabled:
```
logs/env_logs_xrt/{run_name}/{proj_name}-{exptid}-xrt.json
```

**Log format:**
```json
[
  {
    "step": 0,
    "xrt_frame_count": 1,
    "retargeted_qpos": [x, y, z, qw, qx, qy, qz, j1, j2, ..., j27],
    "actions": [a1, a2, ..., a27],
    "reward": 0.5
  }
]
```

---

## Observation Structure

The script builds a **1331-dimensional observation** for the T1 student policy:

### Mimic Observations (35 dims)
From retargeted XRT motion:
1. Root height (1 dim)
2. Root orientation: roll, pitch, yaw (3 dims)
3. Root velocity in local frame (3 dims)
4. Root yaw rate (1 dim)
5. Joint positions (27 dims)

### Proprioceptive Observations (86 dims)
From Isaac Gym simulation:
1. Base angular velocity (3 dims)
2. IMU roll, pitch (2 dims)
3. Joint positions - default (27 dims)
4. Joint velocities (27 dims, ankle zeros)
5. Last action (27 dims)

### History (10 timesteps)
**Total: (35 + 86) × 11 = 1331 dimensions**

The observation stacking is: `[current_obs, history_t-1, ..., history_t-10]`

---

## Troubleshooting

### Common Issues

| Problem | Solution |
|---------|----------|
| **"GMR library not found"** | Install: `cd /home/kelvin/code/GMR && pip install -e .` |
| **"Failed to connect to XRT SDK"** | Check XRT service is running and headset connected |
| **"Checkpoint not found"** | Verify `--proj_name` and `--exptid` match training logs |
| **Task 't1_mimic_stu' not found** | Use `--task t1_stu_rl` instead |
| **Dimension mismatch** | Ensure using T1 task, not G1 or other robot |
| **CUDA kernel error (RTX 5090)** | Use `--sim_device cpu --rl_device cpu` |
| **Low FPS** | Use `--use_jit` and reduce `--num_envs` to 1 |
| **Segmentation fault** | Try `--headless` mode |

### Detailed Solutions

**RTX 5090 Compatibility:**
```bash
# Quick fix: Run on CPU
./play_xrt.sh realtime --cpu

# Or update PyTorch (requires 2.5.0+)
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124
```

**XRT Connection Issues:**
- Verify XRT service: Check if XRoboToolkit background service is running
- Test XRT independently using GMR scripts first
- Check headset is properly calibrated and tracking

**Observation Dimension Errors:**
```bash
# Validate observations first
python test_xrt_obs.py

# Check your model checkpoint was trained with correct config
# T1 student should have: num_obs=1331, num_actions=27
```

---

## Technical Details

### Control Frequency
- Isaac Gym: 50 Hz (dt=0.02s)
- XRT streaming: 30-90 Hz (variable)
- Synchronization: Polls XRT at each control step, reuses last frame if no new data
- Velocity computation: Finite differences over 2 timesteps

### Coordinate Frames
- **XRT**: Headset-centered frame
- **GMR**: Transforms to world frame with `offset_to_ground=True`
- **Policy**: Expects world-frame observations
- **Root orientation**: Converted to roll, pitch, yaw (Euler angles)

### Memory & Performance
- History buffer: 10 × 121 dims × float32 ≈ 5 KB per env
- GMR retargeting: ~1ms per frame
- Main bottleneck: Isaac Gym physics simulation and policy inference

**Expected Performance** (RTX 3090, i9 CPU):
- File replay: 50-60 FPS
- Real-time XRT: 40-50 FPS
- With recording: 30-40 FPS
- JIT vs regular: ~2x speedup

### Joint Mapping (T1 Robot)
27 DoF total:
- Left Arm (7): Shoulder Pitch/Roll, Elbow Pitch/Yaw, Wrist Pitch/Yaw, Hand Roll
- Right Arm (7): Shoulder Pitch/Roll, Elbow Pitch/Yaw, Wrist Pitch/Yaw, Hand Roll
- Waist (1): Waist rotation
- Left Leg (6): Hip Pitch/Roll/Yaw, Knee Pitch, Ankle Pitch/Roll
- Right Leg (6): Hip Pitch/Roll/Yaw, Knee Pitch, Ankle Pitch/Roll

---

## Examples

### 1. Test with Pre-recorded File
```bash
# Record XRT motion
cd /home/kelvin/code/GMR/scripts
python xrt_to_robot_realtime.py \
    --robot booster_t1 \
    --save_path ../data/test_walk.pkl

# Play policy with recording
cd /home/kelvin/code/TWIST
./play_xrt.sh file /home/kelvin/code/GMR/data/test_walk.pkl --record
```

### 2. Real-time Performance Test
```bash
cd /home/kelvin/code/TWIST
./play_xrt.sh realtime --use_jit
```

### 3. Create Publication Demo
```bash
cd /home/kelvin/code/TWIST/legged_gym/legged_gym/scripts
python play_xrt.py \
    --task t1_stu_rl \
    --proj_name t1_student \
    --exptid 1 \
    --mode realtime \
    --use_jit \
    --record_video \
    --record_log \
    --episode_length 30.0
```

### 4. CPU Mode for RTX 5090
```bash
./play_xrt.sh realtime --cpu --record
```

### 5. Multiple Environments (Parallel Testing)
```bash
python play_xrt.py \
    --task t1_stu_rl \
    --proj_name t1_student \
    --exptid 1 \
    --mode file \
    --xrt_file /path/to/motion.pkl \
    --num_envs 4
```

---

## Expected Output

### Successful Startup
```
Loading policy from: ../../logs/t1_student/1
Observation normalization: True
Connecting to XRoboToolkit SDK...
Body tracking data is available!
Estimated human height: 1.75m
Initializing GMR retargeting for T1...
GMR retargeting initialized
Recording video to logs/videos_xrt/...
Starting XRT-driven policy playback! Press Ctrl+C to stop.
Mode: realtime, Control frequency: 50.0 Hz
```

### During Execution
```
Control FPS: 48.52, XRT frames: 324
```
- Control FPS should be 45-50 Hz
- XRT frame count shows mocap frames received

---

## Files Created

- `play_xrt.py` - Main script (~700 lines)
- `play_xrt.sh` - Shell script helper
- `test_xrt_obs.py` - Observation dimension validation
- `README_XRT.md` - This documentation

---

## Future Extensions

Possible improvements:
1. **Multi-person tracking**: Support multiple XRT streams for multi-agent scenarios
2. **Data augmentation**: Add noise to mocap for robustness testing
3. **Online retargeting tuning**: Interactive adjustment of retargeting parameters
4. **Sim2real logging**: Record all data needed for hardware deployment
5. **Interactive controls**: GUI for play/pause, speed control, camera angles

---

## References

- **TWIST**: Teacher-Student Imitation with State Tracking
- **GMR**: General Motion Retargeting library at `/home/kelvin/code/GMR`
- **XRoboToolkit**: XRT motion capture SDK
- **Isaac Gym**: NVIDIA physics simulation framework

---

## Support

For issues and questions:

1. **Check this README** for common solutions
2. **Validate installation** with `test_xrt_obs.py`
3. **Test components independently**:
   - Isaac Gym: Run standard `play.py` first
   - GMR: Test `xrt_to_robot_realtime.py` separately
   - XRT: Verify headset tracking works
4. **Try file mode first** before real-time streaming
5. **Use CPU mode** if encountering GPU compatibility issues

---

**Created for TWIST T1 robot deployment with XRT mocap integration.**
