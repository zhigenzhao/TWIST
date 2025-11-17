import sys
import pickle as pkl
import threading
import time
from types import ModuleType
from typing import Dict, Optional
import numpy as np


class FakeModule(ModuleType):
    """Compatibility layer for numpy version differences."""

    def __init__(self, name, real=None):
        super().__init__(name)
        if real:
            self.__dict__.update(real.__dict__)


# Patch potentially missing modules for pickle compatibility
sys.modules["numpy._core"] = FakeModule("numpy._core", np.core if hasattr(np, "core") else np)
sys.modules["numpy._core.multiarray"] = FakeModule("numpy._core.multiarray", getattr(np.core, "multiarray", None))


class MotionLoader:
    """Thread-safe motion data loader and playback manager."""

    def __init__(self, pkl_path: str, default_motion_reference: Optional[Dict[str, np.ndarray]] = None):
        """
        Load motion data from pickle file.

        Args:
            pkl_path: Path to the .pkl file containing motion data
            default_motion_reference: Default reference to return when not playing, should contain:
                - root_pos, root_quat, root_vel, root_ang_vel, dof_pos
        """
        # Load motion data
        with open(pkl_path, "rb") as f:
            data = pkl.load(f)

        self.fps: float = data["fps"]
        self.root_pos: np.ndarray = data["root_pos"]  # (N, 3)
        self.root_rot: np.ndarray = data["root_rot"]  # (N, 4) quaternion
        self.dof_pos: np.ndarray = data["dof_pos"]  # (N, 27)
        self.local_body_pos: np.ndarray = data["local_body_pos"]  # (N, 32, 3)
        self.link_body_list: list = data["link_body_list"]

        self.num_frames = len(self.root_pos)
        self.dt = 1.0 / self.fps

        # Compute velocities using finite differences
        self.root_vel = self._compute_velocity(self.root_pos)  # (N, 3)
        self.root_ang_vel = self._compute_angular_velocity(self.root_rot)  # (N, 3)

        # Default motion reference
        self.default_reference = default_motion_reference

        # Playback state
        self._lock = threading.Lock()
        self._current_frame = 0
        self._is_playing = False

        # Transition state
        self._is_transitioning = False
        self._transition_start_ref = None
        self._transition_target_ref = None
        self._transition_progress = 0.0
        self._transition_duration = 1.0

        # Current reference (updated by playback or set by transitions)
        self._current_ref = default_motion_reference.copy() if default_motion_reference else None

        # Playback thread
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._start_thread()

        print(f"[MotionLoader] Loaded motion with {self.num_frames} frames at {self.fps} fps")

    def _compute_velocity(self, positions: np.ndarray) -> np.ndarray:
        """Compute linear velocities from positions using finite differences."""
        vel = np.zeros_like(positions)
        vel[:-1] = (positions[1:] - positions[:-1]) / self.dt
        vel[-1] = vel[-2]  # Repeat last velocity
        return vel

    def _compute_angular_velocity(self, quaternions: np.ndarray) -> np.ndarray:
        """Compute angular velocities from quaternions using finite differences."""
        ang_vel = np.zeros((len(quaternions), 3))
        for i in range(len(quaternions) - 1):
            q0 = quaternions[i]
            q1 = quaternions[i + 1]
            # Compute relative rotation: q_diff = q1 * q0^{-1}
            q0_inv = self._quat_conjugate(q0)
            q_diff = self._quat_multiply(q1, q0_inv)
            # Convert to angular velocity
            ang_vel[i] = self._quat_to_angular_velocity(q_diff, self.dt)
        ang_vel[-1] = ang_vel[-2]  # Repeat last angular velocity
        return ang_vel

    def _quat_conjugate(self, q: np.ndarray) -> np.ndarray:
        """Compute quaternion conjugate (w, x, y, z) -> (w, -x, -y, -z)."""
        return np.array([q[0], -q[1], -q[2], -q[3]])

    def _quat_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions (w, x, y, z format)."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ]
        )

    def _quat_to_angular_velocity(self, q: np.ndarray, dt: float) -> np.ndarray:
        """Convert quaternion difference to angular velocity."""
        # For small rotations: omega ≈ 2 * vec(q) / dt
        w, x, y, z = q
        angle = 2 * np.arccos(np.clip(w, -1, 1))
        if angle < 1e-6:
            return np.zeros(3)
        axis = np.array([x, y, z]) / np.sin(angle / 2)
        return axis * angle / dt

    def _start_thread(self):
        """Start the internal update loop thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._update_loop, daemon=True)
        self._thread.start()
        print("[MotionLoader] Update thread started")

    def _update_loop(self):
        """Internal loop that updates playback state at the specified fps."""
        last_time = time.time()

        while not self._stop_event.is_set():
            current_time = time.time()
            elapsed = current_time - last_time

            if elapsed >= self.dt:
                with self._lock:
                    if self._is_transitioning:
                        # Interpolate current reference
                        alpha = min(self._transition_progress / self._transition_duration, 1.0)
                        alpha = 3 * alpha**2 - 2 * alpha**3
                        self._current_ref = {
                            "root_pos": ((1 - alpha) * self._transition_start_ref["root_pos"] +
                                        alpha * self._transition_target_ref["root_pos"]).copy(),
                            "root_quat": ((1 - alpha) * self._transition_start_ref["root_quat"] +
                                         alpha * self._transition_target_ref["root_quat"]).copy(),
                            "root_vel": ((1 - alpha) * self._transition_start_ref["root_vel"] +
                                        alpha * self._transition_target_ref["root_vel"]).copy(),
                            "root_ang_vel": ((1 - alpha) * self._transition_start_ref["root_ang_vel"] +
                                            alpha * self._transition_target_ref["root_ang_vel"]).copy(),
                            "dof_pos": ((1 - alpha) * self._transition_start_ref["dof_pos"] +
                                       alpha * self._transition_target_ref["dof_pos"]).copy(),
                        }

                        # Update transition progress
                        self._transition_progress += elapsed
                        if self._transition_progress >= self._transition_duration:
                            # Transition complete
                            self._current_ref = self._transition_target_ref.copy()
                            self._is_transitioning = False
                            print("[MotionLoader] Transition complete")

                    elif self._is_playing:
                        # Update current reference from motion data
                        self._current_frame += 1
                        if self._current_frame >= self.num_frames:
                            self._current_frame = self.num_frames - 1
                            self._is_playing = False
                            print(f"[MotionLoader] Motion playback ended at frame {self._current_frame}")

                        frame_idx = self._current_frame
                        self._current_ref = {
                            "root_pos": self.root_pos[frame_idx].copy(),
                            "root_quat": self.root_rot[frame_idx].copy(),
                            "root_vel": self.root_vel[frame_idx].copy(),
                            "root_ang_vel": self.root_ang_vel[frame_idx].copy(),
                            "dof_pos": self.dof_pos[frame_idx].copy(),
                        }

                last_time = current_time

            # Small sleep to prevent busy-waiting
            time.sleep(0.001)

    def play(self):
        """Start playing the motion at the specified fps."""
        with self._lock:
            if not self._is_transitioning:
                self._is_playing = True
                print(f"[MotionLoader] Play started at frame {self._current_frame}/{self.num_frames}")

    def pause(self):
        """Pause the motion playback."""
        with self._lock:
            self._is_playing = False
            print(f"[MotionLoader] Paused at frame {self._current_frame}/{self.num_frames}")

    def reset(self):
        """Reset to the start frame."""
        with self._lock:
            self._current_frame = 0
            self._is_playing = False
            self._is_transitioning = False
            print("[MotionLoader] Reset to frame 0")

    def _get_current_reference_unsafe(self) -> Dict[str, np.ndarray]:
        """Get current reference without lock (for internal use only)."""
        return self._current_ref.copy()

    def go_to_start(self, duration: float = 1.0):
        """
        Smoothly transition from current reference to the start of the motion.

        Args:
            duration: Transition duration in seconds
        """
        with self._lock:
            self._is_playing = False
            self._is_transitioning = True
            self._transition_start_ref = self._current_ref.copy()
            self._transition_target_ref = {
                "root_pos": self.default_reference["root_pos"].copy(),
                "root_quat": self.default_reference["root_quat"].copy(),
                "root_vel": self.default_reference["root_vel"].copy(),
                "root_ang_vel": self.default_reference["root_ang_vel"].copy(),
                "dof_pos": self.dof_pos[0].copy(),
            }
            self._transition_progress = 0.0
            self._transition_duration = duration
            print(f"[MotionLoader] Starting transition to start (duration: {duration}s)")

    def go_to_default(self, duration: float = 1.0):
        """
        Smoothly transition from current reference to the default motion reference.

        Args:
            duration: Transition duration in seconds
        """
        if self.default_reference is None:
            print("[MotionLoader] Warning: No default reference set, cannot transition to default")
            return

        with self._lock:
            self._is_playing = False
            self._is_transitioning = True
            self._transition_start_ref = self._current_ref.copy()
            self._transition_target_ref = self.default_reference.copy()
            self._transition_progress = 0.0
            self._transition_duration = duration
            print(f"[MotionLoader] Starting transition to default (duration: {duration}s)")

    def get_motion_reference(self) -> Dict[str, np.ndarray]:
        """
        Get current motion reference.

        Returns:
            Dictionary containing:
                - root_pos: (3,) root position
                - root_quat: (4,) root rotation quaternion
                - root_vel: (3,) root linear velocity
                - root_ang_vel: (3,) root angular velocity
                - dof_pos: (27,) DOF positions
        """
        with self._lock:
            return self._get_current_reference_unsafe()

    def stop(self):
        """Stop the update thread (cleanup)."""
        print("[MotionLoader] Stopping update thread")
        self._stop_event.set()
        if self._thread:
            self._thread.join()
        print("[MotionLoader] Update thread stopped")

    def __del__(self):
        """Cleanup on deletion."""
        self.stop()
