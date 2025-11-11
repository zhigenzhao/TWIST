# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Play a trained T1 policy using XRT mocap data as reference motion.

This script integrates XRT real-time motion capture with TWIST policy playback in Isaac Gym.
It uses GMR (General Motion Retargeting) to retarget human mocap to T1 robot joint angles,
then feeds those as reference motion observations to the trained student policy.

Usage:
    # Real-time XRT streaming
    python play_xrt.py --task t1_mimic_stu --proj_name PROJECT --exptid EXPT --mode realtime

    # Replay from saved XRT file
    python play_xrt.py --task t1_mimic_stu --proj_name PROJECT --exptid EXPT --mode file --xrt_file path/to/file.pkl

    # With recording
    python play_xrt.py ... --record_video --record_log
"""

import os
import sys
import time
import signal
import argparse
from types import ModuleType
from collections import deque
import pickle
import json

import numpy as np
import faulthandler
from tqdm import tqdm
from termcolor import cprint
from scipy.spatial.transform import Rotation as R

# Add GMR to path if needed
sys.path.append("../../../GMR")

# TWIST imports - MUST be before torch import (Isaac Gym requirement)
from legged_gym.envs import *  # noqa: F403, F401
from legged_gym.gym_utils import task_registry, get_args as get_base_args

# PyTorch import AFTER Isaac Gym
import torch

# GMR imports
try:
    from general_motion_retargeting import GeneralMotionRetargeting as GMR
    from general_motion_retargeting.utils.xrt import (
        connect_xrt_realtime,
        get_xrt_frame_realtime,
        estimate_human_height,
        load_xrt_file
    )
    GMR_AVAILABLE = True
except ImportError:
    cprint("Warning: GMR library not found. Please ensure GMR is installed and in your Python path.", "yellow")
    GMR_AVAILABLE = False

# Patch sys.modules to fake missing modules from numpy 2.x
class FakeModule(ModuleType):
    def __init__(self, name, real=None):
        super().__init__(name)
        if real:
            self.__dict__.update(real.__dict__)

sys.modules["numpy._core"] = FakeModule("numpy._core", np.core if hasattr(np, "core") else np)
sys.modules["numpy._core.multiarray"] = FakeModule("numpy._core.multiarray", getattr(np.core, "multiarray", None))


# Global variables for graceful shutdown
running = True
xrt_client = None


def signal_handler(_sig, _frame):
    """Handle Ctrl+C gracefully."""
    global running
    cprint("\nShutting down...", "yellow")
    running = False


def get_load_path(root, checkpoint=-1, model_name_include="jit"):
    """Get the path to the model checkpoint."""
    if checkpoint == -1:
        models = [file for file in os.listdir(root) if model_name_include in file]
        models.sort(key=lambda m: "{0:0>15}".format(m))
        model = models[-1]
        checkpoint = model.split("_")[-1].split(".")[0]
    return model, checkpoint


def quaternion_to_euler(quat):
    """
    Convert quaternion to euler angles (roll, pitch, yaw).

    Args:
        quat: numpy array [w, x, y, z] (scalar-first format)

    Returns:
        roll, pitch, yaw in radians
    """
    rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # Convert to scipy format [x,y,z,w]
    return rot.as_euler('xyz', degrees=False)


def compute_velocity_from_position(pos_history, dt=0.02):
    """
    Compute velocity from position history using finite differences.

    Args:
        pos_history: deque of positions (at least 2 elements)
        dt: time step in seconds

    Returns:
        velocity as numpy array
    """
    if len(pos_history) < 2:
        return np.zeros(3)
    return (pos_history[-1] - pos_history[-2]) / dt


def quat_rotate_inverse(quat, vec):
    """
    Rotate a vector by the inverse of a quaternion.

    Args:
        quat: numpy array [w, x, y, z]
        vec: numpy array [x, y, z]

    Returns:
        rotated vector
    """
    rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # scipy format [x,y,z,w]
    return rot.inv().apply(vec)


class XRTMocapSource:
    """Handles XRT mocap data retrieval in both realtime and file modes."""

    def __init__(self, mode='realtime', xrt_file_path=None, human_height=None):
        """
        Initialize XRT mocap source.

        Args:
            mode: 'realtime' or 'file'
            xrt_file_path: path to XRT pickle file (if mode='file')
            human_height: human height in meters (None to auto-estimate)
        """
        self.mode = mode
        self.human_height = human_height
        self.xrt_client = None
        self.file_frames = None
        self.file_index = 0

        if mode == 'realtime':
            self._init_realtime()
        elif mode == 'file':
            self._init_file(xrt_file_path)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _init_realtime(self):
        """Initialize real-time XRT connection."""
        global xrt_client
        if not GMR_AVAILABLE:
            raise RuntimeError("GMR library not available. Cannot use realtime mode.")

        cprint("Connecting to XRoboToolkit SDK...", "cyan")
        self.xrt_client = connect_xrt_realtime()
        xrt_client = self.xrt_client

        if self.xrt_client is None:
            raise RuntimeError("Failed to connect to XRoboToolkit SDK. Make sure the service is running.")

        # Wait for first frame
        cprint("Waiting for body tracking data...", "cyan")
        frame_data = None
        wait_count = 0
        while frame_data is None and running:
            frame_data = get_xrt_frame_realtime(self.xrt_client)
            if frame_data is None:
                time.sleep(0.01)
                wait_count += 1
                if wait_count > 500:  # 5 seconds timeout
                    raise RuntimeError("Timeout waiting for XRT body tracking data")

        if not running:
            raise KeyboardInterrupt()

        cprint("Body tracking data is available!", "green")

        # Estimate human height if not provided
        if self.human_height is None:
            self.human_height = estimate_human_height(frame_data)
            cprint(f"Estimated human height: {self.human_height:.2f}m", "green")
        else:
            cprint(f"Using specified human height: {self.human_height:.2f}m", "green")

    def _init_file(self, file_path):
        """Initialize from XRT pickle file."""
        if file_path is None:
            raise ValueError("xrt_file_path must be provided when mode='file'")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"XRT file not found: {file_path}")

        cprint(f"Loading XRT data from {file_path}...", "cyan")

        if GMR_AVAILABLE:
            self.file_frames = load_xrt_file(file_path)
        else:
            # Manual pickle loading if GMR not available
            with open(file_path, 'rb') as f:
                self.file_frames = pickle.load(f)

        cprint(f"Loaded {len(self.file_frames)} frames", "green")

        # Estimate human height from first frame if not provided
        if self.human_height is None and len(self.file_frames) > 0:
            if GMR_AVAILABLE:
                self.human_height = estimate_human_height(self.file_frames[0])
            else:
                # Simple height estimation
                first_frame = self.file_frames[0]
                head_z = first_frame.get("Head", (np.zeros(3), None))[0][2]
                left_foot_z = first_frame.get("Left_Foot", (np.zeros(3), None))[0][2]
                right_foot_z = first_frame.get("Right_Foot", (np.zeros(3), None))[0][2]
                self.human_height = head_z - min(left_foot_z, right_foot_z) + 0.15
                self.human_height = np.clip(self.human_height, 1.4, 2.2)

            cprint(f"Estimated human height: {self.human_height:.2f}m", "green")
        elif self.human_height is not None:
            cprint(f"Using specified human height: {self.human_height:.2f}m", "green")

    def get_frame(self):
        """
        Get next XRT frame.

        Returns:
            frame_data dict or None if no data available
        """
        if self.mode == 'realtime':
            return get_xrt_frame_realtime(self.xrt_client)
        else:  # file mode
            if self.file_index >= len(self.file_frames):
                # Loop back to start
                self.file_index = 0

            frame = self.file_frames[self.file_index]
            self.file_index += 1
            return frame

    def disconnect(self):
        """Cleanup and disconnect."""
        if self.xrt_client is not None:
            self.xrt_client.disconnect()


class ObservationBuilder:
    """Builds observations for the T1 student policy from XRT mocap and sim state."""

    def __init__(self, num_envs, history_len, device):
        """
        Initialize observation builder.

        Args:
            num_envs: number of parallel environments
            history_len: number of history timesteps
            device: torch device
        """
        self.num_envs = num_envs
        self.history_len = history_len
        self.device = device

        # T1 specific dimensions
        self.num_actions = 27
        self.n_mimic_obs = 8 + 27  # 35 dims
        self.n_proprio = 3 + 2 + 3 * 27  # 86 dims
        self.n_obs_single = self.n_mimic_obs + self.n_proprio  # 121 dims
        self.num_observations = self.n_obs_single * (history_len + 1)  # 1331 dims

        # History buffer for observations
        self.obs_history = deque(maxlen=history_len)

        # Position history for velocity computation
        self.pos_history = deque(maxlen=2)
        self.rot_history = deque(maxlen=2)

        # Initialize with zeros
        for _ in range(history_len):
            self.obs_history.append(np.zeros(self.n_obs_single))

    def build_mimic_obs(self, retargeted_qpos):
        """
        Build mimic observations from retargeted qpos.

        Args:
            retargeted_qpos: numpy array [root_pos(3), root_rot(4), dof_pos(27)]

        Returns:
            mimic_obs: numpy array (35,)
        """
        root_pos = retargeted_qpos[:3]
        root_rot = retargeted_qpos[3:7]  # [w, x, y, z]
        dof_pos = retargeted_qpos[7:]

        # Store position history for velocity
        self.pos_history.append(root_pos)
        self.rot_history.append(root_rot)

        # Compute velocities
        root_vel = compute_velocity_from_position(self.pos_history, dt=0.02)

        # Compute angular velocity (yaw rate)
        if len(self.rot_history) >= 2:
            _, _, yaw_prev = quaternion_to_euler(self.rot_history[-2])
            _, _, yaw_curr = quaternion_to_euler(self.rot_history[-1])
            yaw_rate = (yaw_curr - yaw_prev) / 0.02
        else:
            yaw_rate = 0.0

        # Convert to euler
        roll, pitch, yaw = quaternion_to_euler(root_rot)

        # Convert velocity to local frame
        root_vel_local = quat_rotate_inverse(root_rot, root_vel)

        # Build mimic observation (35 dims)
        mimic_obs = np.concatenate([
            [root_pos[2]],           # 1 dim - height
            [roll, pitch, yaw],      # 3 dims - orientation
            root_vel_local,          # 3 dims - velocity (local frame)
            [yaw_rate],              # 1 dim - yaw rate
            dof_pos                  # 27 dims - joint positions
        ])

        return mimic_obs

    def build_proprio_obs(self, env, env_id=0):
        """
        Build proprioceptive observations from environment state.

        Args:
            env: IsaacGym environment
            env_id: environment index

        Returns:
            proprio_obs: numpy array (86,)
        """
        # Get observations from environment
        base_ang_vel = env.base_ang_vel[env_id].cpu().numpy() * env.obs_scales.ang_vel

        # IMU (roll, pitch)
        imu_obs = np.array([env.roll[env_id].item(), env.pitch[env_id].item()])

        # Joint positions (reindexed and scaled)
        dof_pos = env.dof_pos[env_id].cpu().numpy()
        default_dof_pos = env.default_dof_pos_all[env_id].cpu().numpy()
        dof_pos_obs = env.reindex((dof_pos - default_dof_pos) * env.obs_scales.dof_pos).cpu().numpy()

        # Joint velocities (reindexed and scaled)
        dof_vel = env.dof_vel[env_id].cpu().numpy()
        dof_vel_obs = env.reindex(dof_vel * env.obs_scales.dof_vel).cpu().numpy()

        # Zero out ankle velocities (indices 4, 5, 10, 11)
        ankle_idx = [4, 5, 10, 11]
        for idx in ankle_idx:
            dof_vel_obs[idx] = 0.0

        # Last action
        last_action = env.action_history_buf[env_id, -1].cpu().numpy()

        # Build proprioceptive observation (86 dims)
        proprio_obs = np.concatenate([
            base_ang_vel,     # 3 dims
            imu_obs,          # 2 dims
            dof_pos_obs,      # 27 dims
            dof_vel_obs,      # 27 dims
            last_action       # 27 dims
        ])

        return proprio_obs

    def build_full_obs(self, mimic_obs, proprio_obs):
        """
        Build full observation with history.

        Args:
            mimic_obs: numpy array (35,)
            proprio_obs: numpy array (86,)

        Returns:
            full_obs: torch tensor (1331,)
        """
        # Single frame observation
        obs_single = np.concatenate([mimic_obs, proprio_obs])

        # Add to history
        self.obs_history.append(obs_single)

        # Stack: [current, history[-1], ..., history[-10]]
        obs_list = [obs_single] + list(self.obs_history)
        full_obs = np.concatenate(obs_list[:self.history_len + 1])

        return torch.from_numpy(full_obs).float().to(self.device)


def play_xrt(args):
    """Main function to play policy with XRT mocap."""
    faulthandler.enable()

    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)

    # Check GMR availability
    if not GMR_AVAILABLE:
        cprint("Error: GMR library is required for this script", "red")
        sys.exit(1)

    # Load policy
    log_pth = "../../logs/{}/".format(args.proj_name) + args.exptid
    cprint(f"Loading policy from: {log_pth}", "cyan")

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # Configure environment for playback
    # Set num_envs - use default if not specified
    if args.num_envs is not None:
        env_cfg.env.num_envs = args.num_envs
    elif not hasattr(env_cfg.env, 'num_envs') or env_cfg.env.num_envs is None:
        env_cfg.env.num_envs = 1  # Default to 1 environment

    env_cfg.env.episode_length_s = args.episode_length
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.max_difficulty = False  # Use easier terrain for XRT playback

    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.action_delay = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False

    env_cfg.env.record_video = args.record_video
    env_cfg.env.rand_reset = False

    if_normalize = env_cfg.env.normalize_obs
    cprint(f"Observation normalization: {if_normalize}", "green")

    # Create environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    # Load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg, log_pth = task_registry.make_alg_runner(
        log_root=log_pth, env=env, name=args.task, args=args, train_cfg=train_cfg, return_log_dir=True
    )

    if args.use_jit:
        path = os.path.join(log_pth, "traced")
        model, _ = get_load_path(root=path, checkpoint=args.checkpoint)
        path = os.path.join(path, model)
        cprint(f"Loading JIT policy: {path}", "green")
        policy_jit = torch.jit.load(path, map_location=env.device)
    else:
        policy = ppo_runner.get_inference_policy(device=env.device)
        if if_normalize:
            try:
                normalizer = ppo_runner.get_normalizer(device=env.device)
            except Exception:
                cprint("No normalizer found", "yellow")
                normalizer = None

    # Initialize XRT mocap source
    try:
        mocap_source = XRTMocapSource(
            mode=args.mode,
            xrt_file_path=args.xrt_file if args.mode == 'file' else None,
            human_height=args.human_height
        )
    except Exception as e:
        cprint(f"Failed to initialize XRT mocap: {e}", "red")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Initialize GMR retargeting
    cprint("Initializing GMR retargeting for T1...", "cyan")
    retargeter = GMR(
        actual_human_height=mocap_source.human_height,
        src_human="xrt",
        tgt_robot="booster_t1"  # T1 robot
    )
    cprint("GMR retargeting initialized", "green")

    # Initialize observation builder
    obs_builder = ObservationBuilder(
        num_envs=env.num_envs,
        history_len=env_cfg.env.history_len,
        device=env.device
    )

    # Setup recording
    if args.record_video:
        import imageio
        mp4_writers = []
        env.enable_viewer_sync = True

        for i in range(env.num_envs):
            video_name = args.proj_name + "-" + args.exptid + f"-xrt-env{i}.mp4"
            run_name = log_pth.split("/")[-1]
            path = f"../../logs/videos_xrt/{run_name}"
            if not os.path.exists(path):
                os.makedirs(path)
            video_name = os.path.join(path, video_name)
            mp4_writer = imageio.get_writer(video_name, fps=int(1 / env.dt))
            cprint(f"Recording video to {video_name}", "green")
            mp4_writers.append(mp4_writer)

    if args.record_log:
        run_name = log_pth.split("/")[-1]
        logs_dict = []
        dict_name = args.proj_name + "-" + args.exptid + "-xrt.json"
        path = f"../../logs/env_logs_xrt/{run_name}"
        if not os.path.exists(path):
            os.makedirs(path)
        dict_name = os.path.join(path, dict_name)
        cprint(f"Logging to {dict_name}", "green")

    # Main control loop
    traj_length = int(env.max_episode_length * 2)  # Run for 2 episodes worth

    # Reset environment
    env.reset()

    # FPS tracking
    fps_counter = 0
    fps_start_time = time.time()
    fps_display_interval = 2.0

    cprint("Starting XRT-driven policy playback! Press Ctrl+C to stop.", "green")
    cprint(f"Mode: {args.mode}, Control frequency: {1/env.dt:.1f} Hz", "cyan")

    last_xrt_frame = None
    xrt_frame_count = 0

    for step_i in tqdm(range(traj_length)):
        if not running:
            break

        try:
            # Get XRT frame
            xrt_frame = mocap_source.get_frame()

            if xrt_frame is None:
                if args.mode == 'realtime':
                    # No data available, use last frame or skip
                    if last_xrt_frame is not None:
                        xrt_frame = last_xrt_frame
                    else:
                        time.sleep(0.001)
                        continue
                else:
                    # File mode should always have data
                    continue
            else:
                last_xrt_frame = xrt_frame
                xrt_frame_count += 1

            # Retarget to T1
            retargeted_qpos = retargeter.retarget(xrt_frame, offset_to_ground=True)

            # Build observations for each environment
            actions_list = []

            for env_id in range(env.num_envs):
                # Build mimic observation from retargeted motion
                mimic_obs = obs_builder.build_mimic_obs(retargeted_qpos)

                # Build proprioceptive observation from environment state
                proprio_obs = obs_builder.build_proprio_obs(env, env_id)

                # Build full observation with history
                full_obs = obs_builder.build_full_obs(mimic_obs, proprio_obs)

                # Run policy
                if args.use_jit:
                    action = policy_jit(full_obs.unsqueeze(0))
                else:
                    if if_normalize and normalizer is not None:
                        normalized_obs = normalizer.normalize(full_obs.unsqueeze(0))
                    else:
                        normalized_obs = full_obs.unsqueeze(0)
                    action = policy(normalized_obs, hist_encoding=True)

                actions_list.append(action.squeeze(0))

            # Stack actions
            actions = torch.stack(actions_list, dim=0)

            # Step environment
            _, _, rews, _, _ = env.step(actions.detach())

            # Record video
            if args.record_video:
                imgs = env.render_record(mode="rgb_array")
                if imgs is not None:
                    for i in range(env.num_envs):
                        mp4_writers[i].append_data(imgs[i])

            # Record log
            if args.record_log:
                log_entry = {
                    "step": step_i,
                    "xrt_frame_count": xrt_frame_count,
                    "retargeted_qpos": retargeted_qpos.tolist(),
                    "actions": actions[0].cpu().numpy().tolist(),  # Log first env
                    "reward": rews[0].item(),
                }
                logs_dict.append(log_entry)

            # FPS measurement
            fps_counter += 1
            current_time = time.time()
            if current_time - fps_start_time >= fps_display_interval:
                actual_fps = fps_counter / (current_time - fps_start_time)
                cprint(f"Control FPS: {actual_fps:.2f}, XRT frames: {xrt_frame_count}", "yellow")
                fps_counter = 0
                fps_start_time = current_time

        except KeyboardInterrupt:
            break
        except Exception as e:
            cprint(f"Error in control loop: {e}", "red")
            import traceback
            traceback.print_exc()
            break

    # Cleanup
    cprint("Cleaning up...", "cyan")

    if args.record_video:
        for mp4_writer in mp4_writers:
            mp4_writer.close()
        cprint("Video recording saved", "green")

    if args.record_log:
        with open(dict_name, "w") as f:
            json.dump(logs_dict, f, indent=2)
        cprint(f"Logs saved to {dict_name}", "green")

    mocap_source.disconnect()

    cprint("Shutdown complete.", "green")


def get_xrt_args():
    """Get arguments by extending the base get_args() with XRT-specific arguments."""
    import sys

    # XRT-specific argument names
    xrt_args_names = ['--mode', '--xrt_file', '--human_height', '--episode_length']

    # Save original sys.argv
    original_argv = sys.argv.copy()

    # Extract XRT arguments and their values
    xrt_values = {}
    filtered_argv = [original_argv[0]]  # Keep script name

    i = 1
    while i < len(original_argv):
        if original_argv[i] in xrt_args_names:
            # This is an XRT argument
            arg_name = original_argv[i][2:]  # Remove '--'
            if i + 1 < len(original_argv):
                xrt_values[arg_name] = original_argv[i + 1]
                i += 2
            else:
                i += 1
        else:
            # This is a base argument, keep it
            filtered_argv.append(original_argv[i])
            i += 1

    # Temporarily replace sys.argv with filtered version
    sys.argv = filtered_argv

    # Get base args (without XRT arguments)
    args = get_base_args()

    # Restore original sys.argv
    sys.argv = original_argv

    # Add XRT-specific arguments with defaults
    args.mode = xrt_values.get('mode', 'realtime')
    args.xrt_file = xrt_values.get('xrt_file', None)
    args.human_height = float(xrt_values['human_height']) if 'human_height' in xrt_values else None
    args.episode_length = float(xrt_values['episode_length']) if 'episode_length' in xrt_values else 60.0

    return args


if __name__ == "__main__":
    args = get_xrt_args()

    # Validate arguments
    if args.mode == 'file' and args.xrt_file is None:
        cprint("Error: --xrt_file must be provided when mode=file", "red")
        sys.exit(1)

    play_xrt(args)
