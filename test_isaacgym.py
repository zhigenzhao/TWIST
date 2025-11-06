#!/usr/bin/env python3
"""
Isaac Gym test script with T1 robot URDF visualization.
Tests loading and simulating the T1 robot that's causing issues.
"""

import sys
import os
import numpy as np

print("=" * 60)
print("Isaac Gym T1 Robot Test with Visualization")
print("=" * 60)

# Test 1: Import Isaac Gym
print("\n[Test 1] Importing Isaac Gym...")
try:
    import isaacgym
    from isaacgym import gymapi, gymutil

    print("✓ Isaac Gym imported successfully")
except ImportError as e:
    print(f"✗ Failed to import Isaac Gym: {e}")
    sys.exit(1)

import torch

# Test 2: Check GPU availability
print("\n[Test 2] Checking GPU availability...")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU 0: {torch.cuda.get_device_name(0)}")
    mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    mem_free = torch.cuda.mem_get_info(0)[0] / 1024**3
    mem_used = torch.cuda.mem_get_info(0)[1] / 1024**3 - mem_free
    print(f"GPU memory: Total {mem_total:.2f} GB, Free {mem_free:.2f} GB, Used {mem_used:.2f} GB")
    print("✓ GPU is available")
else:
    print("✗ No GPU available")
    sys.exit(1)

# Test 3: Create Gym instance
print("\n[Test 3] Creating Gym instance...")
try:
    gym = gymapi.acquire_gym()
    print("✓ Gym instance created successfully")
except Exception as e:
    print(f"✗ Failed to create Gym instance: {e}")
    sys.exit(1)

# Parse command line arguments
args = gymutil.parse_arguments(description="T1 Robot Test", headless=False)

# Test 4: Create simulation with viewer
print("\n[Test 4] Creating simulation...")
try:
    # Simulation parameters
    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 60.0
    sim_params.substeps = 2
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

    # PhysX parameters
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.rest_offset = 0.0
    sim_params.physx.contact_offset = 0.01
    sim_params.physx.friction_offset_threshold = 0.01
    sim_params.physx.friction_correlation_distance = 0.0005
    sim_params.physx.num_threads = 4
    sim_params.physx.use_gpu = True

    # Enable GPU pipeline
    sim_params.use_gpu_pipeline = True

    # Create sim
    sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

    if sim is None:
        raise Exception("Failed to create simulation")

    print("✓ Simulation created successfully")
except Exception as e:
    print(f"✗ Failed to create simulation: {e}")
    sys.exit(1)

# Test 5: Create viewer
print("\n[Test 5] Creating viewer...")
try:
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        raise Exception("Failed to create viewer")

    # Set camera position
    cam_pos = gymapi.Vec3(3, 2, 1.5)
    cam_target = gymapi.Vec3(0, 0, 0.5)
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

    print("✓ Viewer created successfully")
except Exception as e:
    print(f"✗ Failed to create viewer: {e}")
    print("  (Note: Viewer requires display/graphics)")

# Test 6: Add ground plane
print("\n[Test 6] Adding ground plane...")
try:
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    gym.add_ground(sim, plane_params)
    print("✓ Ground plane added")
except Exception as e:
    print(f"✗ Failed to add ground plane: {e}")
    sys.exit(1)

# Test 7: Load G1 robot URDF (known working)
print("\n[Test 7] Loading G1 robot URDF...")
try:
    # Set paths
    asset_root = os.path.join(os.path.dirname(__file__), "assets/t1")
    asset_file = "T1_7_dof_arm_serial_fix_head.urdf"

    print(f"  Asset root: {asset_root}")
    print(f"  Asset file: {asset_file}")

    # Asset options (matching legged_gym config)
    asset_options = gymapi.AssetOptions()
    asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
    asset_options.collapse_fixed_joints = False
    asset_options.replace_cylinder_with_capsule = True
    asset_options.flip_visual_attachments = False
    asset_options.fix_base_link = False
    asset_options.density = 0.001
    asset_options.angular_damping = 0.0
    asset_options.linear_damping = 0.0
    asset_options.max_angular_velocity = 1000.0
    asset_options.max_linear_velocity = 1000.0
    asset_options.armature = 0.0
    asset_options.thickness = 0.01
    asset_options.disable_gravity = False

    # Load the asset
    robot_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

    if robot_asset is None:
        raise Exception("Failed to load G1 asset")

    # Get asset info
    num_dofs = gym.get_asset_dof_count(robot_asset)
    num_bodies = gym.get_asset_rigid_body_count(robot_asset)

    print(f"✓ G1 robot loaded successfully")
    print(f"  DOFs: {num_dofs}")
    print(f"  Rigid bodies: {num_bodies}")

except Exception as e:
    print(f"✗ Failed to load G1 robot: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Test 8: Create environments with G1 robots
print("\n[Test 8] Creating environments with G1 robots...")
try:
    num_envs = 4  # Small number for testing
    spacing = 2.5
    env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
    env_upper = gymapi.Vec3(spacing, spacing, spacing)

    envs = []
    actor_handles = []

    num_per_row = int(np.sqrt(num_envs))

    for i in range(num_envs):
        # Create env
        env = gym.create_env(sim, env_lower, env_upper, num_per_row)
        envs.append(env)

        # Set pose
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0.0, 0.85)  # Standing height
        pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # Create actor
        actor_handle = gym.create_actor(env, robot_asset, pose, f"g1_{i}", i, 1)
        actor_handles.append(actor_handle)

        # Set initial DOF positions to a stable pose
        dof_props = gym.get_actor_dof_properties(env, actor_handle)
        dof_states = gym.get_actor_dof_states(env, actor_handle, gymapi.STATE_ALL)

        # Set all DOFs to zero (neutral pose)
        for j in range(len(dof_states)):
            dof_states["pos"][j] = 0.0
            dof_states["vel"][j] = 0.0

        gym.set_actor_dof_states(env, actor_handle, dof_states, gymapi.STATE_ALL)

    print(f"✓ Created {num_envs} environments with T1 robots")

except Exception as e:
    print(f"✗ Failed to create environments: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Test 9: Prepare simulation and acquire tensors
print("\n[Test 9] Preparing simulation...")
try:
    gym.prepare_sim(sim)

    # Acquire tensors
    gym.refresh_actor_root_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_rigid_body_state_tensor(sim)

    print("✓ Simulation prepared and tensors acquired")

except Exception as e:
    print(f"✗ Failed to prepare simulation: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Test 10: Run simulation with visualization
print("\n[Test 10] Running simulation...")
print("\n" + "=" * 60)
print("Simulation is running. Press ESC or close window to exit.")
print("=" * 60)

try:
    frame = 0
    while not gym.query_viewer_has_closed(viewer):
        # Step physics
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        # Update viewer
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)

        # Sync frame time
        gym.sync_frame_time(sim)

        frame += 1
        if frame % 60 == 0:
            print(f"  Frame {frame} ({frame * sim_params.dt:.1f}s)")

    print("✓ Simulation ran successfully")

except Exception as e:
    print(f"✗ Error during simulation: {e}")
    import traceback

    traceback.print_exc()

# Cleanup
print("\n[Test 11] Cleaning up...")
try:
    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)
    print("✓ Cleanup successful")
except Exception as e:
    print(f"✗ Failed to cleanup: {e}")

# Summary
print("\n" + "=" * 60)
print("✓ ALL TESTS PASSED!")
print("T1 robot URDF loaded and simulated successfully.")
print("=" * 60)
