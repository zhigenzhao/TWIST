from legged_gym.envs.base.humanoid_mimic_config import HumanoidMimicCfg, HumanoidMimicCfgPPO
from legged_gym import LEGGED_GYM_ROOT_DIR


class T1MimicPrivCfg(HumanoidMimicCfg):
    class env(HumanoidMimicCfg.env):
        tar_obs_steps = [
            1,
            5,
            10,
            15,
            20,
            25,
            30,
            35,
            40,
            45,
            50,
            55,
            60,
            65,
            70,
            75,
            80,
            85,
            90,
            95,
        ]

        num_envs = 4096
        num_actions = 27
        obs_type = "priv"  # 'student'
        n_priv_latent = 4 + 1 + 2 * num_actions
        extra_critic_obs = 3
        n_priv = 0

        n_proprio = 3 + 2 + 3 * num_actions
        n_priv_mimic_obs = len(tar_obs_steps) * (8 + num_actions + 3 * 9)  # Hardcode for now, 9 is base, 9 is the number of key bodies
        n_mimic_obs = 8 + 27
        n_priv_info = 3 + 1 + 3 * 9 + 2 + 4 + 1 + 2 * num_actions  # base lin vel, root height, key body pos, contact mask, priv latent
        history_len = 10

        n_obs_single = n_priv_mimic_obs + n_proprio + n_priv_info
        n_priv_obs_single = n_priv_mimic_obs + n_proprio + n_priv_info

        num_observations = n_obs_single

        num_privileged_obs = n_priv_obs_single

        env_spacing = 3.0  # not used with heightfields/trimeshes
        send_timeouts = True  # send time out information to the algorithm
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

        # fmt: off
        dof_err_w = [
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,              # Left Arm
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,              # Right Arm
            0.8,                                            # Waist
            1.0, 1.0, 1.0, 1.0, 0.5, 0.5,                   # Left Leg
            1.0, 1.0, 1.0, 1.0, 0.5, 0.5,                   # Right Leg
        ]
        # fmt: on

        global_obs = False
        # global_obs = True

    class terrain(HumanoidMimicCfg.terrain):
        mesh_type = "trimesh"
        # mesh_type = 'plane'
        # height = [0, 0.02]
        height = [0, 0.01]
        horizontal_scale = 0.1

    class init_state(HumanoidMimicCfg.init_state):
        pos = [0, 0, 0.68]
        default_joint_angles = {
            # "AAHead_yaw": 0.0,
            # "Head_pitch": 0.0,
            "Left_Shoulder_Pitch": 0.2,
            "Right_Shoulder_Pitch": 0.2,
            "Left_Shoulder_Roll": -1.35,
            "Right_Shoulder_Roll": 1.35,
            "Left_Elbow_Pitch": 0.0,
            "Right_Elbow_Pitch": 0.0,
            "Left_Elbow_Yaw": -0.5,
            "Right_Elbow_Yaw": 0.5,
            "Left_Wrist_Pitch": 0.0,
            "Right_Wrist_Pitch": 0.0,
            "Left_Wrist_Yaw": 0.0,
            "Right_Wrist_Yaw": 0.0,
            "Left_Hand_Roll": 0.0,
            "Right_Hand_Roll": 0.0,
            "Waist": 0.0,
            "Left_Hip_Pitch": -0.2,
            "Left_Hip_Roll": 0.0,
            "Left_Hip_Yaw": 0.0,
            "Left_Knee_Pitch": 0.4,
            "Left_Ankle_Pitch": -0.25,
            "Left_Ankle_Roll": 0.0,
            "Right_Hip_Pitch": -0.2,
            "Right_Hip_Roll": 0.0,
            "Right_Hip_Yaw": 0.0,
            "Right_Knee_Pitch": 0.4,
            "Right_Ankle_Pitch": -0.25,
            "Right_Ankle_Roll": 0.0,
        }

    class control(HumanoidMimicCfg.control):
        stiffness = {
            # "Head": 20,
            "Hip": 200,
            "Knee": 200,
            "Waist": 200,
            "Ankle": 50,
            "Shoulder": 50,
            "Elbow": 50,
            "Wrist": 50,
            "Hand": 50,
        }  # [N*m/rad]
        damping = {
            # "Head": 0.2,
            "Hip": 5.0,
            "Knee": 5.0,
            "Waist": 5.0,
            "Ankle": 3.0,
            "Shoulder": 3.0,
            "Elbow": 3.0,
            "Wrist": 3.0,
            "Hand": 3.0,
        }  #  # [N*m*s/rad]

        action_scale = 0.5
        decimation = 10
        # decimation = 4

    class sim(HumanoidMimicCfg.sim):
        dt = 0.002  # 1/500
        # dt = 1/200 # 0.005

    class normalization(HumanoidMimicCfg.normalization):
        clip_actions = 5.0

    class asset(HumanoidMimicCfg.asset):
        # file = f'{LEGGED_GYM_ROOT_DIR}/../assets/g1/g1_custom_collision.urdf'
        file = f"{LEGGED_GYM_ROOT_DIR}/../assets/t1/T1_7_dof_arm_serial_fix_head.urdf"

        # for both joint and link name
        torso_name: str = "Waist"  # humanoid pelvis part
        chest_name: str = "Trunk"  # humanoid chest part

        # for link name
        thigh_name: str = "Hip"
        shank_name: str = "Shank"
        foot_name: str = "foot_link"  # foot_pitch is not used
        waist_name: list = ["Trunk", "Waist"]
        upper_arm_name: str = "UpperArm"
        lower_arm_name: str = "LowerArm"
        hand_name: list = ["right_hand_link", "left_hand_link"]

        feet_bodies = ["left_toe_link", "right_toe_link"]
        n_lower_body_dofs: int = 12

        penalize_contacts_on = ["Shoulder", "Elbow", "Hip", "Knee"]
        terminate_after_contacts_on = ["Trunk", "Waist"]

        # ========================= Inertia =========================
        dof_armature = [0.01] * 27
        # ========================= Inertia =========================

        collapse_fixed_joints = False

    class rewards(HumanoidMimicCfg.rewards):
        regularization_names = [
            "feet_stumble",
            "feet_contact_forces",
            "lin_vel_z",
            "ang_vel_xy",
            "orientation",
            "dof_pos_limits",
            "dof_torque_limits",
            "collision",
            "torque_penalty",
            "thigh_torque_roll_yaw",
            "thigh_roll_yaw_acc",
            "dof_acc",
            "dof_vel",
            "action_rate",
        ]
        regularization_scale = 1.0
        regularization_scale_range = [0.8, 2.0]
        regularization_scale_curriculum = False
        regularization_scale_gamma = 0.0001

        class scales:
            tracking_joint_dof = 0.6
            tracking_joint_vel = 0.2
            tracking_root_pose = 0.6
            tracking_root_vel = 1.0
            # tracking_keybody_pos = 0.6
            tracking_keybody_pos = 2.0

            # alive = 0.5

            feet_slip = -0.1
            feet_contact_forces = -5e-4
            # collision = -10.0
            feet_stumble = -1.25

            dof_pos_limits = -5.0
            dof_torque_limits = -1.0

            dof_vel = -1e-4
            dof_acc = -5e-8
            action_rate = -0.01

            # feet_height = 5.0
            feet_air_time = 5.0

            ang_vel_xy = -0.01
            # orientation = -0.4

            # base_acc = -5e-7
            # orientation = -1.0

            # =========================
            # waist_dof_acc = -5e-8 * 2
            # waist_dof_vel = -1e-4 * 2

            ankle_dof_acc = -5e-8 * 2
            ankle_dof_vel = -1e-4 * 2

            # ankle_action = -0.02

        min_dist = 0.05
        max_dist = 0.4
        max_knee_dist = 0.4
        feet_height_target = 0.15
        feet_air_time_target = 0.5
        only_positive_rewards = False
        tracking_sigma = 0.2
        tracking_sigma_ang = 0.125
        max_contact_force = 100  # Forces above this value are penalized
        soft_torque_limit = 0.95
        torque_safety_limit = 0.9
        root_height_diff_threshold = 0.2

    class domain_rand:
        domain_rand_general = True  # manually set this, setting from parser does not work;

        randomize_gravity = True and domain_rand_general
        gravity_rand_interval_s = 4
        gravity_range = (-0.1, 0.1)

        randomize_friction = True and domain_rand_general
        friction_range = [0.1, 2.5]

        randomize_base_mass = True and domain_rand_general
        added_mass_range = [-1.0, 3.0]

        randomize_base_com = True and domain_rand_general
        added_com_range = [-0.02, 0.02]

        push_robots = True and domain_rand_general
        push_interval_s = 4
        max_push_vel_xy = 1.0

        push_end_effector = True and domain_rand_general
        push_end_effector_interval_s = 2
        max_push_force_end_effector = 20.0

        randomize_motor = True and domain_rand_general
        motor_strength_range = [0.8, 1.2]

        action_delay = True and domain_rand_general
        action_buf_len = 8

    class noise(HumanoidMimicCfg.noise):
        add_noise = True
        noise_increasing_steps = 3000

        class noise_scales:
            dof_pos = 0.01
            dof_vel = 0.15
            lin_vel = 0.1
            ang_vel = 0.1
            gravity = 0.05
            imu = 0.15

    class motion(HumanoidMimicCfg.motion):
        motion_curriculum = True
        motion_curriculum_gamma = 0.01
        key_bodies = [
            "left_hand_link",
            "right_hand_link",
            "left_toe_link",
            "right_toe_link",
            "Shank_Left",
            "Shank_Right",
            "AL5",
            "AR5",
            "H2",
        ]  # 9 key bodies
        upper_key_bodies = ["left_hand_link", "right_hand_link", "AL5", "AR5", "H2"]

        # motion_file = f"{LEGGED_GYM_ROOT_DIR}/motion_data_configs/twist_dataset_simple.yaml"
        motion_file = f"{LEGGED_GYM_ROOT_DIR}/motion_data_configs/feasible_motion_dataset_t1_29dof_pace.yaml"
        reset_consec_frames = 30


class T1MimicStuCfg(T1MimicPrivCfg):
    class env(T1MimicPrivCfg.env):
        tar_obs_steps = [
            1,
            5,
            10,
            15,
            20,
            25,
            30,
            35,
            40,
            45,
            50,
            55,
            60,
            65,
            70,
            75,
            80,
            85,
            90,
            95,
        ]

        num_envs = 4096
        num_actions = 27
        obs_type = "student"
        n_priv_latent = 4 + 1 + 2 * num_actions
        extra_critic_obs = 3
        n_priv = 0

        n_proprio = 3 + 2 + 3 * num_actions
        n_priv_mimic_obs = len(tar_obs_steps) * (8 + num_actions + 3 * 9)  # Hardcode for now, 9 is the number of key bodies
        n_mimic_obs = 8 + 27  # 27 for dof pos

        n_priv_info = 3 + 1 + 3 * 9 + 2 + 4 + 1 + 2 * num_actions  # base lin vel, root height, key body pos, contact mask, priv latent
        history_len = 10

        n_obs_single = n_mimic_obs + n_proprio
        n_priv_obs_single = n_priv_mimic_obs + n_proprio + n_priv_info

        num_observations = n_obs_single * (history_len + 1)

        num_privileged_obs = n_priv_obs_single


class T1MimicStuRLCfg(T1MimicPrivCfg):
    class env(T1MimicPrivCfg.env):
        tar_obs_steps = [
            1,
            5,
            10,
            15,
            20,
            25,
            30,
            35,
            40,
            45,
            50,
            55,
            60,
            65,
            70,
            75,
            80,
            85,
            90,
            95,
        ]

        num_envs = 4096
        num_actions = 27
        obs_type = "student"
        n_priv_latent = 4 + 1 + 2 * num_actions
        extra_critic_obs = 3
        n_priv = 0

        n_proprio = 3 + 2 + 3 * num_actions
        n_priv_mimic_obs = len(tar_obs_steps) * (8 + num_actions + 3 * 9)  # Hardcode for now, 9 is the number of key bodies
        n_mimic_obs = 8 + 27  # 27 for dof pos

        n_priv_info = 3 + 1 + 3 * 9 + 2 + 4 + 1 + 2 * num_actions  # base lin vel, root height, key body pos, contact mask, priv latent
        history_len = 10

        n_obs_single = n_mimic_obs + n_proprio
        n_priv_obs_single = n_priv_mimic_obs + n_proprio + n_priv_info

        num_observations = n_obs_single * (history_len + 1)

        num_privileged_obs = n_priv_obs_single

    class rewards(HumanoidMimicCfg.rewards):
        regularization_names = [
            "feet_stumble",
            "feet_contact_forces",
            "lin_vel_z",
            "ang_vel_xy",
            "orientation",
            "dof_pos_limits",
            "dof_torque_limits",
            "collision",
            "torque_penalty",
            "thigh_torque_roll_yaw",
            "thigh_roll_yaw_acc",
            "dof_acc",
            "dof_vel",
            "action_rate",
        ]
        regularization_scale = 1.0
        regularization_scale_range = [0.8, 2.0]
        regularization_scale_curriculum = False
        regularization_scale_gamma = 0.0001

        class scales:
            tracking_joint_dof = 0.6
            tracking_joint_vel = 0.2
            tracking_root_pose = 0.6
            tracking_root_vel = 1.0
            # tracking_keybody_pos = 0.6
            tracking_keybody_pos = 2.0

            # alive = 0.5

            feet_slip = -0.1  # higher than teacher
            feet_contact_forces = -5e-4
            # collision = -10.0
            feet_stumble = -1.25

            dof_pos_limits = -5.0
            dof_torque_limits = -1.0

            dof_vel = -1e-4
            dof_acc = -7e-8
            action_rate = -0.1

            feet_air_time = 5.0

            ang_vel_xy = -0.01
            # orientation = -0.4

            # base_acc = -5e-7
            # orientation = -1.0

            # =========================
            # waist_dof_acc = -5e-8 * 2
            # waist_dof_vel = -1e-4 * 2

            ankle_dof_acc = -5e-8 * 2
            ankle_dof_vel = -1e-4 * 2
            ankle_action = -0.02

        min_dist = 0.05
        max_dist = 0.4
        max_knee_dist = 0.4
        feet_height_target = 0.15
        feet_air_time_target = 0.5
        only_positive_rewards = False
        tracking_sigma = 0.2
        tracking_sigma_ang = 0.125
        max_contact_force = 100  # Forces above this value are penalized
        soft_torque_limit = 0.95
        torque_safety_limit = 0.9
        root_height_diff_threshold = 0.2


class T1MimicPrivCfgPPO(HumanoidMimicCfgPPO):
    seed = 1

    class runner(HumanoidMimicCfgPPO.runner):
        policy_class_name = "ActorCriticMimic"
        algorithm_class_name = "PPO"
        runner_class_name = "OnPolicyRunnerMimic"
        max_iterations = 30_002  # number of policy updates

        # logging
        save_interval = 2000  # check for potential saves every this many iterations
        experiment_name = "test"
        run_name = ""
        # load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt

    class algorithm(HumanoidMimicCfgPPO.algorithm):
        grad_penalty_coef_schedule = [0.00, 0.00, 700, 1000]
        std_schedule = [1.0, 0.4, 4000, 1500]
        entropy_coef = 0.005

        # Transformer params
        # learning_rate = 1e-4 #1.e-3 #5.e-4
        # schedule = 'fixed' # could be adaptive, fixed

    class policy(HumanoidMimicCfgPPO.policy):
        action_std = [0.5] * 14 + [0.4] * 1 + [0.7] * 12
        init_noise_std = 1.0
        obs_context_len = 11
        actor_hidden_dims = [512, 512, 256, 128]
        critic_hidden_dims = [512, 512, 256, 128]
        activation = "silu"
        layer_norm = True
        motion_latent_dim = 128


class T1MimicStuRLCfgDAgger(T1MimicStuRLCfg):
    seed = 1

    class teachercfg(T1MimicPrivCfgPPO):
        pass

    class runner(T1MimicPrivCfgPPO.runner):
        policy_class_name = "ActorCriticMimic"
        algorithm_class_name = "DaggerPPO"
        runner_class_name = "OnPolicyDaggerRunner"
        max_iterations = 30_002
        warm_iters = 100

        # logging
        save_interval = 2000
        experiment_name = "test"
        run_name = ""
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None

        teacher_experiment_name = "test"
        teacher_proj_name = "t1_priv_mimic"
        teacher_checkpoint = -1
        eval_student = False

    class algorithm(HumanoidMimicCfgPPO.algorithm):
        grad_penalty_coef_schedule = [0.00, 0.00, 700, 1000]
        std_schedule = [1.0, 0.4, 4000, 1500]
        entropy_coef = 0.005

        dagger_coef_anneal_steps = 60000  # Total steps to anneal dagger_coef to dagger_coef_min

        dagger_coef = 0.1
        dagger_coef_min = 0.01  # Minimum value for dagger_coef
        # dagger_coef = 0.0
        # dagger_coef_min = 0.0  # Minimum value for dagger_coef

    class policy(HumanoidMimicCfgPPO.policy):
        action_std = [0.5] * 14 + [0.4] * 1 + [0.7] * 12
        init_noise_std = 1.0
        obs_context_len = 11
        actor_hidden_dims = [512, 512, 256, 128]
        critic_hidden_dims = [512, 512, 256, 128]
        activation = "silu"
        layer_norm = True
        motion_latent_dim = 128
