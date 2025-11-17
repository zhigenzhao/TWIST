import os
import sys
from pathlib import Path
import time
import signal
import sys
from collections import deque

sys.path.append("../")
sys.path.append("./rl_policy")
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

import argparse
import numpy as np
import yaml
from termcolor import colored
import mujoco as mj

from sim2real.rl_policy.base_policy import BasePolicy
from booster_robotics_sdk_python import RobotMode
from motion_loader import MotionLoader

np.set_printoptions(precision=3, suppress=True)


def euler_from_quat(quat):
    eulerVec = np.zeros(3)
    qw, qx, qy, qz = quat
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    eulerVec[0] = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (qw * qy - qz * qx)
    if np.abs(sinp) >= 1:
        eulerVec[1] = np.copysign(np.pi / 2, sinp)
    else:
        eulerVec[1] = np.arcsin(sinp)

    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    eulerVec[2] = np.arctan2(siny_cosp, cosy_cosp)
    return eulerVec


def quat_rotate_inverse(q, v):
    q = np.asarray(q)
    v = np.asarray(v)

    q_w = q[:, -1]  # w
    q_vec = q[:, :3]  # x, y, z

    a = v * (2.0 * q_w**2 - 1.0)[:, np.newaxis]
    b = np.cross(q_vec, v) * (2.0 * q_w)[:, np.newaxis]
    dot = np.sum(q_vec * v, axis=1, keepdims=True)
    c = q_vec * (2.0 * dot)

    return a - b + c


class TwistPolicy(BasePolicy):
    def __init__(self, config, model_path):
        super().__init__(config, model_path)

        # Load robot model for reference motion
        self.mj_model = mj.MjModel.from_xml_path(self.config.get("ROBOT_SCENE"))
        self.mj_data = mj.MjData(self.mj_model)
        mj.mj_resetDataKeyframe(self.mj_model, self.mj_data, 0)

        # Observation and action dimensions
        self.n_policy_dofs = self.config.get("TWIST_POLICY_DOFS")
        self.dim_obs_history_len = self.config.get("TWIST_HIST_LEN")
        self.policy_dof_indices = self.config.get("TWIST_POLICY_INDICES")
        self.ankle_indices = self.config.get("TWIST_ANKLE_INDICES")
        self.n_mimic_obs = 8 + self.n_policy_dofs
        self.n_proprio_obs = 3 + 2 + 3 * self.n_policy_dofs
        self.dim_single_obs = self.n_mimic_obs + self.n_proprio_obs
        self.dim_action = self.num_dofs

        # Initialize observation history buffer using deque (same as server_low_level_g1_sim.py)
        self.obs_history_buf = deque(maxlen=self.dim_obs_history_len)
        for _ in range(self.dim_obs_history_len):
            self.obs_history_buf.append(np.zeros(self.dim_single_obs))

        self.last_action = np.zeros((1, self.n_policy_dofs))
        self.obs_scales = {
            "dof_pos": 1.0,
            "dof_vel": 0.05,
            "ang_vel": 0.25,
        }

        # Load motion file for reference trajectories
        self.motion_file_path = self.config.get("MOTION_FILE_PATH", None)
        self.default_motion_reference = {
            "root_pos": np.array(self.config.get("TWIST_DEFAULT_ROOT_POS")),
            "root_quat": np.array([0, 0, 0, 1]),
            "root_vel": np.zeros(3),
            "root_ang_vel": np.zeros(3),
            "dof_pos": np.array(self.config.get("TWIST_DEFAULT_JOINT_POS")),
        }
        if self.motion_file_path:
            self.motion_loader = MotionLoader(self.motion_file_path, self.default_motion_reference)

        # Bug: force start booster
        print(self.command_sender.client.ChangeMode(RobotMode.kCustom))

    def get_mimic_obs(self):
        if self.motion_loader:
            ref = self.motion_loader.get_motion_reference()
        else:
            ref = self.default_motion_reference

        root_pos = ref.get("root_pos", np.zeros(3))
        root_rot = ref.get("root_quat", np.array([0, 0, 0, 1]))
        root_rot = root_rot[[3, 0, 1, 2]]
        root_linvel = ref.get("root_vel", np.zeros(3))
        root_angvel = ref.get("root_ang_vel", np.zeros(3))
        dof_pos = ref.get("dof_pos", np.zeros(self.n_policy_dofs))

        # Convert to euler
        roll, pitch, yaw = euler_from_quat(root_rot)

        # Vel Transform
        root_linvel = quat_rotate_inverse(root_rot.reshape(1, -1), root_linvel.reshape(1, -1))
        root_angvel = quat_rotate_inverse(root_rot.reshape(1, -1), root_angvel.reshape(1, -1))

        # Construct mimic observation: 1 + 3 + 3 + 1 + n_policy_dofs
        mimic_obs = np.concatenate(
            [
                root_pos[2:3],  # 1
                np.array([roll, pitch, yaw]),  # 3
                root_linvel[0, :],  # 3
                root_angvel[0, 2:3],  # 1
                dof_pos,  # n_policy_dofs
            ]
        ).reshape(1, -1)

        return mimic_obs

    def get_proprio_obs(self, robot_state_data):
        obs_dict = self.get_current_obs_buffer_dict(robot_state_data)

        # Extract roll and pitch from IMU
        roll, pitch, yaw = euler_from_quat(obs_dict["base_quat"][0])
        imu_obs = np.concatenate([roll.reshape(1, 1), pitch.reshape(1, 1)], axis=1)
        ang_vel_scaled = obs_dict["base_ang_vel"] * self.obs_scales["ang_vel"]

        # qpos and qvel
        dof_pos = obs_dict["dof_pos"]
        dof_pos = dof_pos[:, self.policy_dof_indices] * self.obs_scales["dof_pos"]
        dof_vel = obs_dict["dof_vel"]
        dof_vel[:, self.ankle_indices] = 0.0
        dof_vel = dof_vel[:, self.policy_dof_indices] * self.obs_scales["dof_vel"] * 0.01

        # Last action
        last_action = self.last_action.copy()

        # Prepare full proprioceptive observation: 3 + 2 + 3 * n_policy_dofs
        proprio_obs = np.concatenate(
            [
                ang_vel_scaled,  # 3
                imu_obs,  # 2
                dof_pos,  # n_policy_dofs
                dof_vel,  # n_policy_dofs
                last_action,  # n_policy_dofs
            ],
            axis=1,
        )

        return proprio_obs

    def construct_current_obs(self, robot_state_data):
        mimic_obs = self.get_mimic_obs()
        proprio_obs = self.get_proprio_obs(robot_state_data)
        current_obs = np.concatenate(
            [
                mimic_obs,
                proprio_obs,
            ],
            axis=1,
        )
        return current_obs

    def update_obs_history(self, current_obs):
        # Flatten current_obs from (1, dim) to (dim,) for deque storage
        current_obs_flat = current_obs.flatten()

        # Get history before appending current (oldest -> newest order)
        obs_hist = np.array(self.obs_history_buf)
        obs_full = np.concatenate([current_obs_flat, obs_hist.flatten()])
        self.obs_history_buf.append(current_obs_flat)

        return obs_full.reshape(1, -1)

    def rl_inference(self, robot_state_data):
        current_obs = self.construct_current_obs(robot_state_data)
        obs_full = self.update_obs_history(current_obs)
        policy_action = self.policy({"observations": obs_full.astype(np.float32)})
        self.last_action = policy_action.copy()
        scaled_policy_action = policy_action * self.policy_action_scale

        return scaled_policy_action

    def handle_keyboard_button(self, keycode):
        """Handle keyboard button presses."""
        super().handle_keyboard_button(keycode)

        if keycode == "q":
            self.motion_loader.play()
        elif keycode == "w":
            self.motion_loader.stop()
        elif keycode == "r":
            self.motion_loader.go_to_start(3.0)
        elif keycode == "t":
            self.motion_loader.go_to_default(3.0)

    def policy_action(self):
        """Execute TWIST policy action and send commands to robot."""
        cmd_q = np.zeros(self.num_dofs)
        cmd_dq = np.zeros(self.num_dofs)
        cmd_tau = np.zeros(self.num_dofs)
        robot_state_data = self.state_processor.robot_state_data

        if robot_state_data is None:
            return

        scaled_policy_action = self.rl_inference(robot_state_data)

        if self.get_ready_state:
            # 1. Set to Default Joint Position: interpolate from current to default
            q_target = self.get_init_target(robot_state_data)
            self.init_count = min(self.init_count, 500)
        elif not self.use_policy_action:
            # 2. No Policy Action: hold current position
            q_target = robot_state_data[:, 7 : 7 + self.num_dofs]
        else:
            # 3. Apply policy action to all joints
            q_target = np.zeros_like(self.get_init_target(robot_state_data))
            q_target[:, self.policy_dof_indices] = scaled_policy_action + self.default_dof_angles[self.policy_dof_indices].reshape(1, -1)

        # Clip q target to motor limits
        if self.motor_pos_lower_limit_list and self.motor_pos_upper_limit_list:
            q_target[0] = np.clip(q_target[0], self.motor_pos_lower_limit_list, self.motor_pos_upper_limit_list)

        # Send command
        cmd_q = q_target[0]
        self.command_sender.send_command(cmd_q, cmd_dq, cmd_tau, robot_state_data[0, 7 : 7 + self.num_dofs])


def signal_handler(sig, frame):
    sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TWIST Policy")
    parser.add_argument("--config", type=str, help="config file path")
    parser.add_argument("--model_path", type=str, help="path to ONNX model")
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.safe_load(file)

    model_path = args.model_path if args.model_path else config.get("model_path")
    if not model_path:
        raise ValueError("model_path must be provided either via --model_path argument or in config file")

    policy = TwistPolicy(config=config, model_path=model_path)
    signal.signal(signal.SIGINT, signal_handler)
    policy.run()
