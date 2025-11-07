from legged_gym.envs.base.humanoid_mimic_config import HumanoidMimicCfg, HumanoidMimicCfgPPO
from legged_gym import LEGGED_GYM_ROOT_DIR


class T1MimicPrivCfg(HumanoidMimicCfg):
    class env(HumanoidMimicCfg.env):
        tar_obs_steps = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45,
                         50, 55, 60, 65, 70, 75, 80, 85, 90, 95,]

        num_envs = 4096
        num_actions = 23  # T1 has 23 DOF (same as G1)
        obs_type = 'priv' # 'student'
        n_priv_latent = 4 + 1 + 2*num_actions
        extra_critic_obs = 3
        n_priv = 0

        n_proprio = 3 + 2 + 3*num_actions
        n_priv_mimic_obs = len(tar_obs_steps) * (8 + num_actions + 3*9) # 9 key bodies
        n_mimic_obs = 8 + 23 # 23 for dof pos
        n_priv_info = 3 + 1 + 3*9 + 2 + 4 + 1 + 2*num_actions
        history_len = 10

        n_obs_single = n_priv_mimic_obs + n_proprio + n_priv_info
        n_priv_obs_single = n_priv_mimic_obs + n_proprio + n_priv_info

        num_observations = n_obs_single
        num_privileged_obs = n_priv_obs_single

        env_spacing = 3.
        send_timeouts = True
        episode_length_s = 10

        randomize_start_pos = True
        randomize_start_yaw = False

        history_encoding = True
        contact_buf_len = 10
        normalize_obs = True

        enable_early_termination = True
        pose_termination = True
        pose_termination_dist = 0.7
        rand_reset = True
        track_root = False

        # T1 joint structure: 6 (left leg) + 6 (right leg) + 3 (waist+head) + 4 (left arm) + 4 (right arm) = 23
        # TODODONE: Task specific -- follow G1, increase waist value and a small head value, probably need further tuning
        dof_err_w = [1.0, 0.8, 0.8, 1.0, 0.5, 0.5, # Left Leg (6): hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
                     1.0, 0.8, 0.8, 1.0, 0.5, 0.5, # Right Leg (6)
                     0.8,                           # Waist (1): increased from G1's 0.6 since T1 has only 1 waist joint
                     0.4, 0.4,                      # Head (2): yaw, pitch - decreased from 0.5 (less critical)
                     0.8, 0.8, 0.8, 1.0,            # Left Arm (4): shoulder_pitch, shoulder_roll, elbow_pitch, elbow_yaw
                     0.8, 0.8, 0.8, 1.0,            # Right Arm (4)
                     ]

        global_obs = False

    class terrain(HumanoidMimicCfg.terrain):
        mesh_type = 'trimesh'
        height = [0, 0.00]
        horizontal_scale = 0.1

    class init_state(HumanoidMimicCfg.init_state):
        pos = [0, 0, 0.72]  # TODODONE: use the data from robot_lab https://github.com/fan-ziqi/robot_lab/blob/main/source/robot_lab/robot_lab/assets/booster.py

        # T1 joint names (23 joints)
        default_joint_angles = {
            # Left Leg (6)
            'Left_Hip_Pitch': -0.2,
            'Left_Hip_Roll': 0.0,
            'Left_Hip_Yaw': 0.0,
            'Left_Knee_Pitch': 0.4,
            'Left_Ankle_Pitch': -0.2,
            'Left_Ankle_Roll': 0.0,

            # Right Leg (6)
            'Right_Hip_Pitch': -0.2,
            'Right_Hip_Roll': 0.0,
            'Right_Hip_Yaw': 0.0,
            'Right_Knee_Pitch': 0.4,
            'Right_Ankle_Pitch': -0.2,
            'Right_Ankle_Roll': 0.0,

            # Waist & Head (3)
            'Waist': 0.0,           # Single waist joint (T1 has 1, not 3 like G1!)
            'AAHead_yaw': 0.0,      # T1 has active head
            'Head_pitch': 0.0,

            # Left Arm (4)
            'Left_Shoulder_Pitch': 0.0,
            'Left_Shoulder_Roll': 0.4,
            'Left_Elbow_Pitch': 1.2,
            'Left_Elbow_Yaw': 0.0,  # T1 has elbow yaw (wrist rotation)

            # Right Arm (4)
            'Right_Shoulder_Pitch': 0.0,
            'Right_Shoulder_Roll': -0.4,
            'Right_Elbow_Pitch': 1.2,
            'Right_Elbow_Yaw': 0.0,
        }

    class control(HumanoidMimicCfg.control):
        # TODODONE Tune and verify the pd gain
        # ========================= PD Gains Sources =========================
        # Sources:
        # 1. robot_lab (github.com/fan-ziqi/robot_lab) - T1 full-body config
        #    - File: source/robot_lab/robot_lab/assets/booster.py
        #    - Task: Velocity tracking on rough terrain (23 DOF)
        # 2. Booster Gym (github.com/BoosterRobotics/booster_gym) - Official T1 locomotion
        #    - File: envs/T1.yaml
        #    - Task: Locomotion (12 DOF legs only)
        # 3. FFTAI GR1T1/T2 (github.com/fan-ziqi/robot_lab) - Humanoid with active head
        #    - File: source/robot_lab/robot_lab/assets/fftai.py
        #    - Used for: Head gains (T1 has active head, no direct T1 source available)
        # =====================================================================

        stiffness = {'Hip_Yaw': 200,       # Source: robot_lab + Booster Gym (T1 legs)
                     'Hip_Roll': 200,      # Source: robot_lab + Booster Gym (T1 legs)
                     'Hip_Pitch': 200,     # Source: robot_lab + Booster Gym (T1 legs)
                     'Knee': 200,          # Source: robot_lab + Booster Gym (T1 legs)
                     'Ankle': 50,          # Source: robot_lab + Booster Gym (T1 legs)
                     'Waist': 200,         # Source: robot_lab (T1 leg actuator group)
                     'Head': 10,           # Source: FFTAI GR1T1/T2 (proxy - T1 has active head)
                     'Shoulder': 40,       # Source: robot_lab (T1 arms)
                     'Elbow': 40,          # Source: robot_lab (T1 arms)
                     }  # [N*m/rad]
        damping = {  'Hip_Yaw': 5,         # Source: robot_lab + Booster Gym (T1 legs)
                     'Hip_Roll': 5,        # Source: robot_lab + Booster Gym (T1 legs)
                     'Hip_Pitch': 5,       # Source: robot_lab + Booster Gym (T1 legs)
                     'Knee': 5,            # Source: robot_lab + Booster Gym (T1 legs)
                     'Ankle': 1,           # Source: robot_lab + Booster Gym (T1 legs)
                     'Waist': 5,           # Source: robot_lab (T1 leg actuator group)
                     'Head': 1,            # Source: FFTAI GR1T1/T2 (proxy - T1 has active head)
                     'Shoulder': 10,       # Source: robot_lab (T1 arms)
                     'Elbow': 10,          # Source: robot_lab (T1 arms)
                     }  # [N*m*s/rad]

        action_scale = 0.5
        decimation = 10

    class sim(HumanoidMimicCfg.sim):
        dt = 0.002 # 1/500

    class normalization(HumanoidMimicCfg.normalization):
        clip_actions = 5.0

    class asset(HumanoidMimicCfg.asset):
        # T1 URDF path
        file = f'{LEGGED_GYM_ROOT_DIR}/../GMR/assets/booster_t1/T1_serial.urdf'

        
        #TODODONE: Based on T1 URDF, name are the same below
        torso_name: str = 'Trunk'              # T1's main torso
        chest_name: str = 'H2'                 # Head link (after pitch joint)

        # Link names for body parts
        thigh_name: str = 'Hip_Pitch'          # Thigh links
        shank_name: str = 'Shank'              # Knee/shin links
        foot_name: str = 'foot_link'           # Foot links
        waist_name: list = ['Waist']           # T1 has single waist link
        upper_arm_name: str = 'AL2'            # Shoulder links (AL2/AR2)
        lower_arm_name: str = 'AL3'            # Elbow links (AL3/AR3)
        hand_name: list = ['left_hand_link', 'right_hand_link']

        # Contact bodies
        feet_bodies = ['left_foot_link', 'right_foot_link']
        n_lower_body_dofs: int = 12  # 6 per leg

        penalize_contacts_on = ["shoulder", "elbow", "Hip", "Shank"]
        terminate_after_contacts_on = ['Trunk']

        # TODODONE Verify the armature for t1
        # ========================= DOF Armature Sources =========================
        #
        # 2. robot_lab (github.com/fan-ziqi/robot_lab):
        #    - File: source/robot_lab/robot_lab/assets/booster.py
        #    - Value: armature = 0.01 for all joints
        #    - Task: Full-body velocity tracking (23 DOF)
        #
        # 3. T1 URDF/XML files (GMR/assets/booster_t1/):
        #    - Value: armature = 0.01 (generic placeholder, not robot-specific)
        # =========================================================================

        dof_armature = [0.01] * 23  # Source: robot_lab T1 config (all joints), also is said in xml
        collapse_fixed_joints = False

    class rewards(HumanoidMimicCfg.rewards):
        regularization_names = []
        regularization_scale = 1.0
        regularization_scale_range = [0.8,2.0]
        regularization_scale_curriculum = False
        regularization_scale_gamma = 0.0001

        class scales:
            tracking_joint_dof = 0.6
            tracking_joint_vel = 0.2
            tracking_root_pose = 0.6
            tracking_root_vel = 1.0
            tracking_keybody_pos = 2.0

            feet_slip = -0.1
            feet_contact_forces = -5e-4
            feet_stumble = -1.25

            dof_pos_limits = -5.0
            dof_torque_limits = -1.0

            dof_vel = -1e-4
            dof_acc = -5e-8
            action_rate = -0.01

            feet_air_time = 5.0
            ang_vel_xy = -0.01

            ankle_dof_acc = -5e-8 * 2
            ankle_dof_vel = -1e-4 * 2

        min_dist = 0.1
        max_dist = 0.4
        max_knee_dist = 0.4
        feet_height_target = 0.2
        feet_air_time_target = 0.5
        only_positive_rewards = False
        tracking_sigma = 0.2
        tracking_sigma_ang = 0.125
        max_contact_force = 100
        soft_torque_limit = 0.95
        torque_safety_limit = 0.9
        root_height_diff_threshold = 0.2

    class domain_rand:
        # TODODONE: May need adjustment for T1
        domain_rand_general = True

        randomize_gravity = (True and domain_rand_general)
        gravity_rand_interval_s = 4
        gravity_range = (-0.1, 0.1) # keep twist's

        randomize_friction = (True and domain_rand_general)
        friction_range = [0.1, 2.]

        randomize_base_mass = (True and domain_rand_general)
        added_mass_range = [-6., 6] # source   https://github.com/BoosterRobotics/booster_gym/blob/main/envs/T1.yaml line 239

        randomize_base_com = (True and domain_rand_general)
        added_com_range = [-0.1, 0.1] #same source

        push_robots = (True and domain_rand_general)
        push_interval_s = 4
        max_push_vel_xy = 1.0

        push_end_effector = (True and domain_rand_general)
        push_end_effector_interval_s = 2
        max_push_force_end_effector = 20.0

        randomize_motor = (True and domain_rand_general)
        motor_strength_range = [0.8, 1.2]

        action_delay = (True and domain_rand_general)
        action_buf_len = 8

    class noise(HumanoidMimicCfg.noise):
        add_noise = True
        noise_increasing_steps = 3000
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 0.1
            lin_vel = 0.1
            ang_vel = 0.1
            gravity = 0.05
            imu = 0.1

    class motion(HumanoidMimicCfg.motion):
        motion_curriculum = True
        motion_curriculum_gamma = 0.01

        # TODODONE: these T1 link names same in T1 URDF!
        # 9 key bodies for tracking rewards
        key_bodies = [
            "left_hand_link",      # Left hand
            "right_hand_link",     # Right hand
            "left_foot_link",      # Left foot
            "right_foot_link",     # Right foot
            "Shank_Left",          # Left knee
            "Shank_Right",         # Right knee
            "AL3",                 # Left elbow
            "AR3",                 # Right elbow
            "H2"                   # Head
        ]

        upper_key_bodies = [
            "left_hand_link",
            "right_hand_link",
            "AL3",
            "AR3",
            "H2"
        ]

        # TODO: Update this path after retargeting T1 data
        motion_file = f"{LEGGED_GYM_ROOT_DIR}/motion_data_configs/t1_twist_dataset.yaml"

        reset_consec_frames = 30


class T1MimicStuCfg(T1MimicPrivCfg):
    class env(T1MimicPrivCfg.env):
        tar_obs_steps = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45,
                         50, 55, 60, 65, 70, 75, 80, 85, 90, 95,]

        num_envs = 4096
        num_actions = 23
        obs_type = 'student'
        n_priv_latent = 4 + 1 + 2*num_actions
        extra_critic_obs = 3
        n_priv = 0

        n_proprio = 3 + 2 + 3*num_actions
        n_priv_mimic_obs = len(tar_obs_steps) * (8 + num_actions + 3*9) # Hardcode for now, 9 is the number of key bodies
        n_mimic_obs = 8 + 23 # 23 for dof pos

        n_priv_info = 3 + 1 + 3*9 + 2 + 4 + 1 + 2*num_actions # base lin vel, root height, key body pos, contact mask, priv latent
        history_len = 10

        n_obs_single = n_mimic_obs + n_proprio
        n_priv_obs_single = n_priv_mimic_obs + n_proprio + n_priv_info

        num_observations = n_obs_single * (history_len + 1)

        num_privileged_obs = n_priv_obs_single

class T1MimicStuRLCfg(T1MimicPrivCfg):
    class env(T1MimicPrivCfg.env):
        tar_obs_steps = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45,
                         50, 55, 60, 65, 70, 75, 80, 85, 90, 95,]

        num_envs = 4096
        num_actions = 23
        obs_type = 'student'
        n_priv_latent = 4 + 1 + 2*num_actions
        extra_critic_obs = 3
        n_priv = 0

        n_proprio = 3 + 2 + 3*num_actions
        n_priv_mimic_obs = len(tar_obs_steps) * (8 + num_actions + 3*9) # Hardcode for now, 9 is the number of key bodies
        n_mimic_obs = 8 + 23 # 23 for dof pos

        n_priv_info = 3 + 1 + 3*9 + 2 + 4 + 1 + 2*num_actions # base lin vel, root height, key body pos, contact mask, priv latent
        history_len = 10

        n_obs_single = n_mimic_obs + n_proprio
        n_priv_obs_single = n_priv_mimic_obs + n_proprio + n_priv_info

        num_observations = n_obs_single * (history_len + 1)

        num_privileged_obs = n_priv_obs_single

    class rewards(HumanoidMimicCfg.rewards):
        # Same structure as teacher
        regularization_names = []
        regularization_scale = 1.0
        regularization_scale_range = [0.8,2.0]
        regularization_scale_curriculum = False
        regularization_scale_gamma = 0.0001

        class scales:
            tracking_joint_dof = 0.6
            tracking_joint_vel = 0.2
            tracking_root_pose = 0.6
            tracking_root_vel = 1.0
            tracking_keybody_pos = 2.0

            feet_slip = -0.1
            feet_contact_forces = -5e-4
            feet_stumble = -1.25

            dof_pos_limits = -5.0
            dof_torque_limits = -1.0

            dof_vel = -1e-4
            dof_acc = -5e-8
            action_rate = -0.01

            feet_air_time = 5.0
            ang_vel_xy = -0.01

            ankle_dof_acc = -5e-8 * 2
            ankle_dof_vel = -1e-4 * 2

        min_dist = 0.1
        max_dist = 0.4
        max_knee_dist = 0.4
        feet_height_target = 0.2
        feet_air_time_target = 0.5
        only_positive_rewards = False
        tracking_sigma = 0.2
        tracking_sigma_ang = 0.125
        max_contact_force = 100
        soft_torque_limit = 0.95
        torque_safety_limit = 0.9
        root_height_diff_threshold = 0.2


# PPO config for teacher
class T1MimicPrivCfgPPO(HumanoidMimicCfgPPO):
    seed = 1

    class runner(HumanoidMimicCfgPPO.runner):
        policy_class_name = 'ActorCriticMimic'
        algorithm_class_name = 'PPO'
        runner_class_name = 'OnPolicyRunnerMimic'
        num_steps_per_env = 24 # originally 24
        max_iterations = 40000 # originally 1_000_002

        save_interval = 500
        experiment_name = 't1_priv_mimic'  # Changed from G1
        run_name = ''
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None

    class algorithm(HumanoidMimicCfgPPO.algorithm):
        grad_penalty_coef_schedule = [0.00, 0.00, 700,1000] #originally [0.00, 0.00, 700, 1000]
        std_schedule = [1.0, 0.4, 4000, 1500] # originally [1.0, 0.4, 4000, 1500]
        entropy_coef = 0.005#originally 0.005

        learning_rate = 2e-4 #originally 2e-4
        num_learning_epochs = 5 #originally 5
        desired_kl = 0.008 #originally 0.008

    class policy(HumanoidMimicCfgPPO.policy):
        # T1: 12 (legs) + 3 (waist+head) + 8 (arms) = 23
        action_std = [0.7] * 12 + [0.4] * 3 + [0.5] * 8
        init_noise_std = 1.0
        obs_context_len = 11
        actor_hidden_dims = [512, 512, 256, 128]
        critic_hidden_dims = [512, 512, 256, 128]
        activation = 'silu'
        layer_norm = True
        motion_latent_dim = 128


# DAgger config for student
class T1MimicStuRLCfgDAgger(T1MimicStuRLCfg):
    seed = 1

    class teachercfg(T1MimicPrivCfgPPO):
        pass

    class runner(T1MimicPrivCfgPPO.runner):
        policy_class_name = 'ActorCriticMimic'
        algorithm_class_name = 'DaggerPPO'
        runner_class_name = 'OnPolicyDaggerRunner'
        max_iterations = 50000 #originally 1_000_002
        warm_iters = 100

        save_interval = 500
        experiment_name = 't1_stu_rl'         # Changed from G1
        run_name = ''
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None

        teacher_experiment_name = 't1_priv_mimic'  # Changed from G1
        teacher_proj_name = 't1_priv_mimic'        # Changed from G1
        teacher_checkpoint = -1
        eval_student = False

    class algorithm(HumanoidMimicCfgPPO.algorithm):
        grad_penalty_coef_schedule = [0.00, 0.00, 700, 1000]
        std_schedule = [1.0, 0.4, 4000, 1500]
        entropy_coef = 0.005

        dagger_coef_anneal_steps = 60000
        dagger_coef = 0.1
        dagger_coef_min = 0.01

    class policy(HumanoidMimicCfgPPO.policy):
        action_std = [0.7] * 12 + [0.4] * 3 + [0.5] * 8
        init_noise_std = 1.0
        obs_context_len = 11
        actor_hidden_dims = [512, 512, 256, 128]
        critic_hidden_dims = [512, 512, 256, 128]
        activation = 'silu'
        layer_norm = True
        motion_latent_dim = 128
