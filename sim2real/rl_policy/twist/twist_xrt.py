import os
import sys
from pathlib import Path
import signal
import argparse
import numpy as np
import yaml
from termcolor import colored

sys.path.append("../")
sys.path.append("./rl_policy")
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from sim2real.rl_policy.twist.twist import TwistPolicy
from sim2real.rl_policy.twist.realtime_motion_provider import RealtimeMotionProvider
from booster_robotics_sdk_python import RobotMode

# Try to import XRT SDK for controller button support
try:
    import xrobotoolkit_sdk as xrt_sdk
    XRT_BUTTONS_AVAILABLE = True
except ImportError:
    XRT_BUTTONS_AVAILABLE = False

np.set_printoptions(precision=3, suppress=True)


class TwistXRTPolicy(TwistPolicy):
    """
    TWIST Policy with real-time XRoboToolkit retargeting.
    Extends TwistPolicy to use RealtimeMotionProvider instead of MotionLoader.
    """

    def __init__(self, config, model_path):
        """
        Initialize TWIST XRT policy.

        Args:
            config: Configuration dictionary
            model_path: Path to ONNX model
        """
        # Store config for parent initialization
        self._xrt_config = config.copy()

        # Call parent init (but we'll override motion_loader)
        # Note: We temporarily set MOTION_FILE_PATH to None to prevent file loading
        original_motion_path = config.get("MOTION_FILE_PATH")
        config["MOTION_FILE_PATH"] = None
        super().__init__(config, model_path)
        config["MOTION_FILE_PATH"] = original_motion_path  # Restore for reference

        # Replace motion_loader with real-time provider
        robot_type = self._xrt_config.get("XRT_ROBOT_TYPE", "booster_t1_29dof")
        human_height = self._xrt_config.get("XRT_HUMAN_HEIGHT", None)
        update_rate = self._xrt_config.get("XRT_UPDATE_RATE", 100.0)

        print(colored("[TwistXRTPolicy] Initializing real-time motion provider...", "cyan"))
        self.motion_loader = RealtimeMotionProvider(
            robot_type=robot_type,
            human_height=human_height,
            default_motion_reference=self.default_motion_reference,
            update_rate=update_rate,
        )

        # XRT-specific state
        self._xrt_connected = self.motion_loader.is_connected()
        self._last_valid_state = True
        self._connection_loss_count = 0
        self._max_connection_loss = 10  # Stop after 10 consecutive failures

        # XR controller button state tracking (for rising edge detection)
        self._prev_a_button_state = False
        self._prev_b_button_state = False

        if not self._xrt_connected:
            print(colored("[TwistXRTPolicy] Warning: XRT not connected. Policy will not run.", "yellow"))
        else:
            print(colored("[TwistXRTPolicy] Real-time XRT retargeting ready!", "green"))

    def handle_keyboard_button(self, keycode):
        """Handle keyboard button presses (override to remove file-based controls)."""
        # Call base policy keyboard handler (handles space, s, etc.)
        from sim2real.rl_policy.base_policy import BasePolicy
        BasePolicy.handle_keyboard_button(self, keycode)

        # XRT-specific controls
        if keycode == "r":
            print(colored("[TwistXRTPolicy] Resetting velocity computation", "cyan"))
            self.motion_loader.reset()
        elif keycode == "i":
            # Info about connection status
            if self.motion_loader.is_connected():
                valid = self.motion_loader.has_valid_reference()
                status = "Valid reference" if valid else "No valid reference"
                print(colored(f"[TwistXRTPolicy] XRT Connected | {status}", "green"))
            else:
                print(colored("[TwistXRTPolicy] XRT Disconnected", "red"))

    def _poll_xr_buttons(self):
        """Poll XR controller buttons and handle presses."""
        if not XRT_BUTTONS_AVAILABLE:
            return

        try:
            # Get current button states
            curr_a_button = xrt_sdk.get_A_button()
            curr_b_button = xrt_sdk.get_B_button()

            # Detect rising edge (button press, not hold)
            # Button A: Start policy action (same as ']' key)
            if curr_a_button and not self._prev_a_button_state:
                print(colored("[TwistXRTPolicy] A button pressed: Starting policy action", "green"))
                self._handle_start_policy()

            # Button B: Stop policy action (same as 'o' key)
            if curr_b_button and not self._prev_b_button_state:
                print(colored("[TwistXRTPolicy] B button pressed: Stopping policy action", "yellow"))
                self._handle_stop_policy()

            # Update previous button states
            self._prev_a_button_state = curr_a_button
            self._prev_b_button_state = curr_b_button

        except Exception as e:
            # Silently ignore errors (e.g., XRT SDK not fully initialized)
            pass

    def policy_action(self):
        """
        Execute TWIST policy action with XRT safety checks.
        Overrides parent to add XRT connection monitoring.
        """
        # Poll XR controller buttons for A/B button presses
        self._poll_xr_buttons()

        robot_state_data = self.state_processor.robot_state_data

        if robot_state_data is None:
            return

        # Safety check: Verify XRT connection and valid reference
        if not self.motion_loader.has_valid_reference():
            if self._last_valid_state:
                print(colored("[TwistXRTPolicy] Warning: XRT reference lost!", "red"))
                self._last_valid_state = False
                self._connection_loss_count = 0

            self._connection_loss_count += 1

            if self._connection_loss_count >= self._max_connection_loss:
                print(colored(
                    f"[TwistXRTPolicy] XRT connection lost for {self._connection_loss_count} frames. "
                    "Stopping policy execution.",
                    "red"
                ))
                # Stop using policy action (will hold current position)
                self.use_policy_action = False

            # Hold current position
            cmd_q = robot_state_data[0, 7 : 7 + self.num_dofs]
            cmd_dq = np.zeros(self.num_dofs)
            cmd_tau = np.zeros(self.num_dofs)
            self.command_sender.send_command(cmd_q, cmd_dq, cmd_tau, robot_state_data[0, 7 : 7 + self.num_dofs])
            return
        else:
            # Connection restored
            if not self._last_valid_state:
                print(colored("[TwistXRTPolicy] XRT reference restored!", "green"))
                self._last_valid_state = True
                self._connection_loss_count = 0

        # Normal policy execution (same as parent)
        cmd_q = np.zeros(self.num_dofs)
        cmd_dq = np.zeros(self.num_dofs)
        cmd_tau = np.zeros(self.num_dofs)

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

    def cleanup(self):
        """Cleanup resources on shutdown."""
        print(colored("[TwistXRTPolicy] Cleaning up...", "cyan"))
        if hasattr(self, 'motion_loader') and self.motion_loader is not None:
            self.motion_loader.stop()
        print(colored("[TwistXRTPolicy] Cleanup complete", "green"))


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    print(colored("\n[TwistXRTPolicy] Shutdown signal received", "yellow"))
    sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TWIST Policy with Real-Time XRT Retargeting")
    parser.add_argument("--config", type=str, help="config file path")
    parser.add_argument("--model_path", type=str, help="path to ONNX model")
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.safe_load(file)

    model_path = args.model_path if args.model_path else config.get("model_path")
    if not model_path:
        raise ValueError("model_path must be provided either via --model_path argument or in config file")

    print(colored("=" * 80, "cyan"))
    print(colored("TWIST Real-Time XRT Retargeting Policy", "cyan", attrs=["bold"]))
    print(colored("=" * 80, "cyan"))
    print(colored(f"Config: {args.config}", "white"))
    print(colored(f"Model: {model_path}", "white"))
    print(colored("=" * 80, "cyan"))
    print(colored("\nKeyboard Controls:", "yellow"))
    print(colored("  ]              : Start policy action", "white"))
    print(colored("  o              : Stop policy and set action to 0", "white"))
    print(colored("  s              : Toggle get ready state", "white"))
    print(colored("  r              : Reset velocity computation", "white"))
    print(colored("  i              : Show XRT connection info", "white"))
    print(colored("  Ctrl+C         : Exit", "white"))
    if XRT_BUTTONS_AVAILABLE:
        print(colored("\nXR Controller Buttons:", "yellow"))
        print(colored("  A Button       : Start policy action (same as ']')", "white"))
        print(colored("  B Button       : Stop policy and set action to 0 (same as 'o')", "white"))
    print(colored("=" * 80, "cyan"))
    print()

    # Initialize policy
    policy = TwistXRTPolicy(config=config, model_path=model_path)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        policy.run()
    except KeyboardInterrupt:
        pass
    finally:
        policy.cleanup()
        print(colored("\n[TwistXRTPolicy] Shutdown complete", "green"))
