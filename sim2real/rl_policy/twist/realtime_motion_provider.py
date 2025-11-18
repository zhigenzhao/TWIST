import sys
import threading
import time
from typing import Dict, Optional
import numpy as np

try:
    import xrobotoolkit_sdk as xrt_sdk
    from general_motion_retargeting import GeneralMotionRetargeting as GMR
    from general_motion_retargeting.utils.xrt import connect_xrt_realtime, get_xrt_frame_realtime, estimate_human_height
    XRT_AVAILABLE = True
except ImportError as e:
    XRT_AVAILABLE = False
    print(f"[RealtimeMotionProvider] Warning: XRT/GMR not available: {e}")


class RealtimeMotionProvider:
    """
    Real-time motion provider that connects to XRoboToolkit SDK and uses GMR
    for retargeting to robot motion. Provides the same interface as MotionLoader.
    """

    def __init__(
        self,
        robot_type: str = "booster_t1_29dof",
        human_height: Optional[float] = None,
        default_motion_reference: Optional[Dict[str, np.ndarray]] = None,
        update_rate: float = 100.0,  # Hz
    ):
        """
        Initialize real-time motion provider.

        Args:
            robot_type: Target robot type for GMR retargeting
            human_height: Human height in meters (if None, will auto-estimate from first frame)
            default_motion_reference: Default reference to return when not connected, should contain:
                - root_pos, root_quat, root_vel, root_ang_vel, dof_pos
            update_rate: Update rate in Hz for fetching and retargeting XRT data
        """
        if not XRT_AVAILABLE:
            raise RuntimeError(
                "XRoboToolkit SDK or GeneralMotionRetargeting not available. "
                "Please install required packages."
            )

        self.robot_type = robot_type
        self.human_height = human_height
        self.default_reference = default_motion_reference
        self.update_rate = update_rate
        self.dt = 1.0 / update_rate

        # Connection state
        self.xrt_client = None
        self.retarget = None
        self._connected = False
        self._valid_reference = False

        # Playback state
        self._lock = threading.Lock()
        self._current_ref = default_motion_reference.copy() if default_motion_reference else None

        # History for velocity computation (store last 2 frames)
        self._prev_root_pos = None
        self._prev_root_quat = None
        self._prev_timestamp = None

        # Update thread
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Initialize connection
        self._initialize_connection()

        # Start update thread
        self._start_thread()

    def _initialize_connection(self):
        """Initialize connection to XRoboToolkit SDK and GMR retargeting."""
        print("[RealtimeMotionProvider] Connecting to XRoboToolkit SDK...")
        self.xrt_client = connect_xrt_realtime()

        if self.xrt_client is None:
            print("[RealtimeMotionProvider] Failed to connect to XRoboToolkit SDK")
            return

        print("[RealtimeMotionProvider] Waiting for body tracking data...")
        frame_data = None
        timeout = 10.0  # 10 second timeout
        start_time = time.time()

        while frame_data is None and (time.time() - start_time) < timeout:
            frame_data = get_xrt_frame_realtime(self.xrt_client)
            if frame_data is None:
                time.sleep(0.01)

        if frame_data is None:
            print("[RealtimeMotionProvider] Timeout waiting for body tracking data")
            return

        print("[RealtimeMotionProvider] Body tracking data available")

        # Estimate human height if not provided
        if self.human_height is None:
            self.human_height = estimate_human_height(frame_data)
            print(f"[RealtimeMotionProvider] Estimated human height: {self.human_height:.2f}m")
        else:
            print(f"[RealtimeMotionProvider] Using specified human height: {self.human_height:.2f}m")

        # Initialize GMR retargeting
        print(f"[RealtimeMotionProvider] Initializing GMR retargeting for {self.robot_type}...")
        self.retarget = GMR(
            actual_human_height=self.human_height,
            src_human="xrt",
            tgt_robot=self.robot_type,
            auto_ground_offset=0.0,
        )

        self._connected = True
        self._valid_reference = True
        print("[RealtimeMotionProvider] Initialization complete")

    def _start_thread(self):
        """Start the internal update loop thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._update_loop, daemon=True)
        self._thread.start()
        print("[RealtimeMotionProvider] Update thread started")

    def _update_loop(self):
        """Internal loop that fetches XRT data and retargets at the specified rate."""
        last_time = time.time()

        while not self._stop_event.is_set():
            current_time = time.time()
            elapsed = current_time - last_time

            if elapsed >= self.dt:
                if self._connected and self.retarget is not None:
                    try:
                        # Get current XRT frame
                        frame_data = get_xrt_frame_realtime(self.xrt_client)

                        if frame_data is None:
                            # Connection lost or no data
                            with self._lock:
                                self._valid_reference = False
                            print("[RealtimeMotionProvider] Warning: No XRT data available")
                        else:
                            # Retarget to robot (with ground alignment)
                            qpos = self.retarget.retarget(frame_data, offset_to_ground=True)

                            # Extract components
                            root_pos = qpos[:3]
                            root_quat_wxyz = qpos[3:7]  # GMR outputs w,x,y,z format
                            # Convert to x,y,z,w format (MotionLoader expects this)
                            root_quat = root_quat_wxyz[[1, 2, 3, 0]]
                            dof_pos = qpos[7:]

                            # Compute velocities from finite differences
                            root_vel = np.zeros(3)
                            root_ang_vel = np.zeros(3)

                            if self._prev_root_pos is not None and self._prev_timestamp is not None:
                                dt_actual = current_time - self._prev_timestamp
                                if dt_actual > 0:
                                    # Linear velocity
                                    root_vel = (root_pos - self._prev_root_pos) / dt_actual

                                    # Angular velocity from quaternion difference
                                    root_ang_vel = self._compute_angular_velocity(
                                        self._prev_root_quat, root_quat, dt_actual
                                    )

                            # Update reference
                            with self._lock:
                                self._current_ref = {
                                    "root_pos": root_pos.copy(),
                                    "root_quat": root_quat.copy(),
                                    "root_vel": root_vel.copy(),
                                    "root_ang_vel": root_ang_vel.copy(),
                                    "dof_pos": dof_pos.copy(),
                                }
                                self._valid_reference = True

                            # Store for next iteration
                            self._prev_root_pos = root_pos.copy()
                            self._prev_root_quat = root_quat.copy()
                            self._prev_timestamp = current_time

                    except Exception as e:
                        print(f"[RealtimeMotionProvider] Error during retargeting: {e}")
                        with self._lock:
                            self._valid_reference = False

                last_time = current_time

            # Small sleep to prevent busy-waiting
            time.sleep(0.001)

    def _compute_angular_velocity(self, q0: np.ndarray, q1: np.ndarray, dt: float) -> np.ndarray:
        """
        Compute angular velocity from two quaternions (x,y,z,w format).

        Args:
            q0: Previous quaternion (x,y,z,w)
            q1: Current quaternion (x,y,z,w)
            dt: Time difference

        Returns:
            Angular velocity (3,)
        """
        # Convert to w,x,y,z for computation
        q0_wxyz = q0[[3, 0, 1, 2]]
        q1_wxyz = q1[[3, 0, 1, 2]]

        # Compute relative rotation: q_diff = q1 * q0^{-1}
        q0_inv = self._quat_conjugate(q0_wxyz)
        q_diff = self._quat_multiply(q1_wxyz, q0_inv)

        # Convert to angular velocity
        w, x, y, z = q_diff
        angle = 2 * np.arccos(np.clip(w, -1, 1))
        if angle < 1e-6:
            return np.zeros(3)
        axis = np.array([x, y, z]) / np.sin(angle / 2)
        return axis * angle / dt

    def _quat_conjugate(self, q: np.ndarray) -> np.ndarray:
        """Compute quaternion conjugate (w,x,y,z) -> (w,-x,-y,-z)."""
        return np.array([q[0], -q[1], -q[2], -q[3]])

    def _quat_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions (w,x,y,z format)."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ])

    def has_valid_reference(self) -> bool:
        """Check if a valid motion reference is available."""
        with self._lock:
            return self._valid_reference

    def is_connected(self) -> bool:
        """Check if connected to XRT."""
        return self._connected

    def get_motion_reference(self) -> Dict[str, np.ndarray]:
        """
        Get current motion reference (same interface as MotionLoader).

        Returns:
            Dictionary containing:
                - root_pos: (3,) root position
                - root_quat: (4,) root rotation quaternion (x,y,z,w)
                - root_vel: (3,) root linear velocity
                - root_ang_vel: (3,) root angular velocity
                - dof_pos: (n_dofs,) DOF positions
        """
        with self._lock:
            if self._current_ref is None:
                # Return default reference if available
                if self.default_reference is not None:
                    return self.default_reference.copy()
                else:
                    raise RuntimeError("No motion reference available")
            return self._current_ref.copy()

    def stop(self):
        """Stop the update thread (cleanup)."""
        print("[RealtimeMotionProvider] Stopping update thread")
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)

        # Disconnect XRT client
        if self.xrt_client is not None:
            try:
                self.xrt_client.disconnect()
            except Exception as e:
                print(f"[RealtimeMotionProvider] Error disconnecting XRT: {e}")

        print("[RealtimeMotionProvider] Stopped")

    def __del__(self):
        """Cleanup on deletion."""
        self.stop()

    # Compatibility methods with MotionLoader (not used in TWIST, but for completeness)
    def play(self):
        """Compatibility with MotionLoader API (no-op for real-time)."""
        print("[RealtimeMotionProvider] Note: play() is no-op for real-time provider")

    def pause(self):
        """Compatibility with MotionLoader API (no-op for real-time)."""
        print("[RealtimeMotionProvider] Note: pause() is no-op for real-time provider")

    def reset(self):
        """Compatibility with MotionLoader API (resets velocity computation)."""
        with self._lock:
            self._prev_root_pos = None
            self._prev_root_quat = None
            self._prev_timestamp = None
        print("[RealtimeMotionProvider] Reset velocity computation")

    def go_to_start(self, duration: float = 1.0):
        """Compatibility with MotionLoader API (no-op for real-time)."""
        print("[RealtimeMotionProvider] Note: go_to_start() is no-op for real-time provider")

    def go_to_default(self, duration: float = 1.0):
        """Compatibility with MotionLoader API (no-op for real-time)."""
        print("[RealtimeMotionProvider] Note: go_to_default() is no-op for real-time provider")
