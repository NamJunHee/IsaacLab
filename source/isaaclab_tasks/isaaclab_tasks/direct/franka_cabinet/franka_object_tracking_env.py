# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector
from pxr import UsdGeom, Usd, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import sample_uniform

from isaaclab.sensors import CameraCfg, Camera
from isaaclab.assets import RigidObjectCfg, RigidObject
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg, CollisionPropertiesCfg
from builtin_interfaces.msg import Time

# from PIL import Image
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

@configclass
class FrankaObjectTrackingEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 8.3333  # 500 timesteps
    decimation = 2
    action_space = 9
    observation_space = 23
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=3.0, replicate_physics=True)

    # robot
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Franka/franka_instanceable.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=12, solver_velocity_iteration_count=1
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                # "panda_joint1": 1.157,
                # "panda_joint2": -1.066,
                # "panda_joint3": -0.155,
                # "panda_joint4": -2.239,
                # "panda_joint5": -1.841,
                # "panda_joint6": 1.003,
                # "panda_joint7": 0.469,
                # "panda_finger_joint.*": 0.035,
                
                "panda_joint1": 0.000,
                "panda_joint2": -1.231,
                "panda_joint3": -0.000,
                "panda_joint4": -2.696,
                "panda_joint5": -0.000,
                "panda_joint6": 2.433,
                "panda_joint7": 0.707,
                "panda_finger_joint.*": 0.035,
            },
            pos=(1.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit=87.0,
                velocity_limit=2.175,
                # stiffness=80.0,
                stiffness=200.0,
                # damping=4.0,
                damping=10.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit=12.0,
                velocity_limit=2.61,
                # stiffness=80.0,
                stiffness=200.0,
                # damping=4.0,
                damping=10.0,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint.*"],
                effort_limit=200.0,
                velocity_limit=0.2,
                stiffness=2e3,
                damping=1e2,
            ),
        },
    )

    # cabinet
    cabinet = ArticulationCfg(
        prim_path="/World/envs/env_.*/Cabinet",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd",
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0, 0.4),
            rot=(0.1, 0.0, 0.0, 0.0),
            joint_pos={
                "door_left_joint": 0.0,
                "door_right_joint": 0.0,
                "drawer_bottom_joint": 0.0,
                "drawer_top_joint": 0.0,
            },
        ),
        actuators={
            "drawers": ImplicitActuatorCfg(
                joint_names_expr=["drawer_top_joint", "drawer_bottom_joint"],
                effort_limit=87.0,
                velocity_limit=100.0,
                stiffness=10.0,
                damping=1.0,
            ),
            "doors": ImplicitActuatorCfg(
                joint_names_expr=["door_left_joint", "door_right_joint"],
                effort_limit=87.0,
                velocity_limit=100.0,
                stiffness=10.0,
                damping=2.5,
            ),
        },
    )

    # ground plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    
    #카메라
    # camera = CameraCfg(
    #     prim_path="/World/envs/env_.*/Robot/panda_hand/hand_camera", 
    #     update_period=0.1,
    #     height=480,
    #     width=640,
    #     data_types=["rgb", "depth"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=16.0,
    #         focus_distance=400.0,
    #         horizontal_aperture=35.0,
    #         clipping_range=(0.1, 1.0e5),
    #     ),
    #     offset=CameraCfg.OffsetCfg(
    #         pos=(0.0, 0.0, 0.05),
    #         rot=(0.0, 0.707, 0.707, 0.0),
    #         convention="ROS",
    #     )
    # )
    
    #큐브
    cube = RigidObjectCfg(
        prim_path="/World/envs/env_.*/cube",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.1, 0, 0.055], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                # usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/red_block.usd",ee
                scale=(0.8, 0.8, 0.8),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=True,
                ),
        ),
    )

    #mustard
    box = RigidObjectCfg(
        prim_path="/World/envs/env_.*/base_link",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.3, 0, 0.055], rot=[0.923, 0, 0, -0.382]),
        spawn=UsdFileCfg(
                # usd_path="/home/nmail-njh/NMAIL/01_Project/Robot_Grasping/objects_usd/google_objects_usd/003_cracker_box/003_cracker_box.usd",
                usd_path="/home/nmail-njh/NMAIL/01_Project/Robot_Grasping/objects_usd/google_objects_usd/006_mustard_bottle/006_mustard_bottle.usd",
                scale=(1.0, 1.0, 1.0),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=True,
                    kinematic_enabled = True,
                ),
            ),
    )
    
    action_scale = 7.5
    dof_velocity_scale = 0.1

    # reward scales
    dist_reward_scale = 1.5
    rot_reward_scale = 1.5
    open_reward_scale = 10.0
    action_penalty_scale = 0.05
    finger_reward_scale = 2.0
    
    #time
    current_time = 0

class FrankaObjectTrackingEnv(DirectRLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: FrankaObjectTrackingEnvCfg

    def __init__(self, cfg: FrankaObjectTrackingEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        def get_env_local_pose(env_pos: torch.Tensor, xformable: UsdGeom.Xformable, device: torch.device):
            """Compute pose in env-local coordinates"""
            world_transform = xformable.ComputeLocalToWorldTransform(0)
            world_pos = world_transform.ExtractTranslation()
            world_quat = world_transform.ExtractRotationQuat()

            px = world_pos[0] - env_pos[0]
            py = world_pos[1] - env_pos[1]
            pz = world_pos[2] - env_pos[2]
            qx = world_quat.imaginary[0]
            qy = world_quat.imaginary[1]
            qz = world_quat.imaginary[2]
            qw = world_quat.real

            return torch.tensor([px, py, pz, qw, qx, qy, qz], device=device)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # create auxiliary variables for computing applied action, observations and rewards
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)

        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)
        self.robot_dof_speed_scales[self._robot.find_joints("panda_finger_joint1")[0]] = 0.1
        self.robot_dof_speed_scales[self._robot.find_joints("panda_finger_joint2")[0]] = 0.1

        self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

        stage = get_current_stage()
        hand_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_link7")),
            self.device,
        )
        lfinger_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_leftfinger")),
            self.device,
        )
        rfinger_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_rightfinger")),
            self.device,
        )

        finger_pose = torch.zeros(7, device=self.device)
        finger_pose[0:3] = (lfinger_pose[0:3] + rfinger_pose[0:3]) / 2.0
        finger_pose[3:7] = lfinger_pose[3:7]
        hand_pose_inv_rot, hand_pose_inv_pos = tf_inverse(hand_pose[3:7], hand_pose[0:3])

        robot_local_grasp_pose_rot, robot_local_pose_pos = tf_combine(
            hand_pose_inv_rot, hand_pose_inv_pos, finger_pose[3:7], finger_pose[0:3]
        )
        robot_local_pose_pos += torch.tensor([0, 0.04, 0], device=self.device)
        self.robot_local_grasp_pos = robot_local_pose_pos.repeat((self.num_envs, 1))
        self.robot_local_grasp_rot = robot_local_grasp_pose_rot.repeat((self.num_envs, 1))
        
        box_local_pose = torch.tensor([0.0, 0.0,0.0, 1.0, 0.0, 0.0, 0.0], device=self.device)
        self.box_local_pos = box_local_pose[0:3].repeat((self.num_envs, 1))
        self.box_local_rot = box_local_pose[3:7].repeat((self.num_envs, 1))

        self.gripper_forward_axis = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        self.gripper_up_axis = torch.tensor([0, 1, 0], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        
        # self.cube_z_axis = torch.tensor([0,0,1], device=self.device, dtype=torch.float32).repeat(
        #     (self.num_envs,1)
        # )
        self.box_z_axis = torch.tensor([0,0,1], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs,1)
        )

        self.hand_link_idx = self._robot.find_bodies("panda_link7")[0][0]
        self.left_finger_link_idx = self._robot.find_bodies("panda_leftfinger")[0][0]
        self.right_finger_link_idx = self._robot.find_bodies("panda_rightfinger")[0][0]
        
        # self.cube_idx = self._cube.find_bodies("cube")[0][0]
        self.box_idx = self._box.find_bodies("base_link")[0][0]

        self.robot_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.robot_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        
        # self.cube_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        # self.cube_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        # self.cube_center = self._cube.data.body_link_pos_w[:,0,:].clone()
        
        self.box_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.box_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.box_center = self._box.data.body_link_pos_w[:,0,:].clone()
        
        rclpy.init()
        # self.node = rclpy.create_node('isaac_camera_publisher')
        # self.camera_info_publisher = self.node.create_publisher(CameraInfo, '/isaac_camera_info_rect',10)
        # self.rgb_publisher = self.node.create_publisher(Image, '/isaac_image_rect',10)
        # self.depth_publisher = self.node.create_publisher(Image, '/isaac_depth',10)
        
        # self.node = rclpy.create_node('camera_publisher')
        # self.camera_info_publisher = self.node.create_publisher(CameraInfo, '/camera_info_rect',10)
        # self.rgb_publisher = self.node.create_publisher(Image, '/image_rect',10)
        # self.depth_publisher = self.node.create_publisher(Image, '/depth',10)
        # self.bridge = CvBridge()
        # self.timer = self.node.create_timer(0.1, self.publish_camera_data)
        
    # def publish_camera_data(self):
    #     env_id = 0
        
    #     zero_time = Time()
    #     zero_time.sec = 0
    #     zero_time.nanosec = 0
        
    #     rgb_data = self._camera.data.output["rgb"]
    #     depth_data = self._camera.data.output["depth"]
        
    #     rgb_image = (rgb_data.cpu().numpy()[env_id]).astype(np.uint8)
    #     # rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)  # BGR to RGB 변환
        
    #     # depth_image = (depth_data.cpu().numpy()[env_id]).astype(np.uint8)
    #     depth_image = (depth_data.cpu().numpy()[env_id]).astype(np.float32)
        
    #     # Publish Camera Info
    #     camera_info_msg = CameraInfo()
    #     # camera_info_msg.header.stamp = self.node.get_clock().now().to_msg()
    #     camera_info_msg.header.stamp = zero_time
    #     camera_info_msg.header.frame_id = 'tf_camera'
        
    #     camera_info_msg.height = 480 #rgb_image.shape[0]
    #     camera_info_msg.width = 640 #rgb_image.shape[1]
    #     camera_info_msg.distortion_model = 'plumb_bob'
        
    #     intrinsic_matrices = self._camera.data.intrinsic_matrices.cpu().numpy().flatten().tolist()
    #     camera_info_msg.k = intrinsic_matrices[:9]
    #     camera_info_msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
    #     camera_info_msg.r = [1.0, 0.0, 0.0,
    #                          0.0, 1.0, 0.0,
    #                          0.0, 0.0, 1.0]
    #     camera_info_msg.p = intrinsic_matrices[:3] + [0.0] + intrinsic_matrices[3:6] + [0.0] + [0.0, 0.0, 1.0, 0.0]
    #     # camera_info_msg.p = [1.0, 0.0, 0.0, 0.0,
    #     #                      0.0, 1.0, 0.0, 0.0,
    #     #                      0.0, 0.0, 1.0, 0.0]
         
    #     camera_info_msg.binning_x = 0
    #     camera_info_msg.binning_y = 0

    #     camera_info_msg.roi.x_offset = 0
    #     camera_info_msg.roi.y_offset = 0
    #     camera_info_msg.roi.height = 0
    #     camera_info_msg.roi.width = 0
    #     camera_info_msg.roi.do_rectify = False
        
    #     self.camera_info_publisher.publish(camera_info_msg)
    #     self.node.get_logger().info('Published camera info')
        
    #     # Publish RGB Image
    #     rgb_msg = self.bridge.cv2_to_imgmsg(rgb_image, encoding='rgb8')
    #     # rgb_msg.header.stamp = self.node.get_clock().now().to_msg()
    #     rgb_msg.header.stamp = zero_time
    #     rgb_msg.header.frame_id = 'tf_camera'
    #     self.rgb_publisher.publish(rgb_msg)
    #     self.node.get_logger().info('Published RGB image')

    #     # Publish Depth Image
    #     depth_msg = self.bridge.cv2_to_imgmsg(depth_image, encoding='32FC1')
    #     # depth_msg.header.stamp = self.node.get_clock().now().to_msg()
    #     depth_msg.header.stamp = zero_time
    #     depth_msg.header.frame_id = 'tf_camera'
    #     self.depth_publisher.publish(depth_msg)
    #     depth_msg.step = depth_image.shape[1] * 4
    #     self.node.get_logger().info('Published Depth image')
                
    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
    
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        
        # 카메라 추가
        # self._camera = Camera(self.cfg.camera)
        # self.scene.sensors["hand_camera"] = self._camera
        
        # 큐브 추가
        # self._cube = RigidObject(self.cfg.cube)
        # self.scene.rigid_objects["cube"] = self._cube
        
        # 상자 추가
        self._box = RigidObject(self.cfg.box)
        self.scene.rigid_objects["base_link"] = self._box

    # pre-physics step calls

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone().clamp(-1.0, 1.0)
        targets = self.robot_dof_targets + self.robot_dof_speed_scales * self.dt * self.actions * self.cfg.action_scale
        self.robot_dof_targets[:] = torch.clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        
        self.cfg.current_time = self.cfg.current_time + self.dt
        current_time = torch.tensor(self.cfg.current_time, device=self.device, dtype=torch.float32)
                
        # 물체 원 운동 (실제 운동 제어 코드)---------------------------------------------------------------------------------------------------------------
        R = 0.15
        omega = 1.5
        # noise_level = 0.02
        # random_noise = (torch.rand(3, device=self.device) * 2 - 1) * noise_level
        
        offset_x = R * torch.cos(omega * current_time) #+ random_noise[0]
        offset_y = R * torch.sin(omega * current_time) #+ random_noise[1]
        offset_z = 0.055
        
        offset_pos = torch.tensor([offset_x, offset_y, offset_z], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        
        new_pos = self.box_center + offset_pos
        new_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device, dtype=torch.float32).unsqueeze(0).repeat(self.num_envs, 1)
        
        new_box_pose = torch.cat([new_pos, new_rot], dim = -1)
        # self._box.write_root_pose_to_sim(new_box_pose)    
        
    def _apply_action(self):
        # print("robot_stop")
        self._robot.set_joint_position_target(self.robot_dof_targets)
        
    # post-physics step calls

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = self._box.data.body_link_pos_w[:, 0,2] > 0.3
        # truncated = self.episode_length_buf >= self.max_episode_length - 30 # 물체 원운동 환경 초기화 주기
        truncated = self.episode_length_buf >= self.max_episode_length - 400 # 물체 램덤 생성 환경 초기화 주기
        
        #환경 고정
        terminated = 0
        # truncated = 0
        
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        # Refresh the intermediate values after the physics steps
        self._compute_intermediate_values()
        robot_left_finger_pos = self._robot.data.body_link_pos_w[:, self.left_finger_link_idx]
        robot_right_finger_pos = self._robot.data.body_link_pos_w[:, self.right_finger_link_idx]

        return self._compute_rewards(
            self.actions,
            self.robot_grasp_pos,
            self.box_grasp_pos,
            self.robot_grasp_rot,
            self.box_grasp_rot,
            self.gripper_forward_axis,
            self.gripper_up_axis,
            self.cfg.dist_reward_scale,
            self.cfg.rot_reward_scale,
            self.cfg.action_penalty_scale,
        )

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        
        # robot state
        # joint_pos = self._robot.data.default_joint_pos[env_ids] + sample_uniform(
        #     -0.125,
        #     0.125,
        #     (len(env_ids), self._robot.num_joints),
        #     self.device,
        # )
        # joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        # joint_vel = torch.zeros_like(joint_pos)
        # self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        # self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        
        # init_joint_position (reward 함수를 위한 변수) ---------------------------------------------------
        # self.init_robot_joint_position = self._robot.data.joint_pos.clone()
        # self.init_robot_grasp_pos = self.robot_grasp_pos.clone()
        
        #물체 원 운동 (원 운동 시 환경 초기화 코드)-----------------------------------------------------------------------------------------------------------------
        reset_pos = self.box_center
        reset_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device, dtype=torch.float32).unsqueeze(0).repeat(self.num_envs, 1)
        reset_box_pose = torch.cat([reset_pos, reset_rot], dim = -1)
        
        # self._box.write_root_pose_to_sim(reset_box_pose)
        
        #물체 랜덤 위치 생성 (실제 물체 생성 코드) ------------------------------------------------------------------------------------------------------------
        pos_range = {
            "x" : ( 0.05, 0.25),
            "y" : (-0.3, 0.3),
            "z" : ( 0.055, 0.3)
        }
        fixed_z = 0.055
        
        random_position = torch.stack([
            torch.rand(self.num_envs, device=self.device) * (pos_range["x"][1] - pos_range["x"][0]) + pos_range["x"][0],
            torch.rand(self.num_envs, device=self.device) * (pos_range["y"][1] - pos_range["y"][0]) + pos_range["y"][0],
            torch.rand(self.num_envs, device=self.device) * (pos_range["z"][1] - pos_range["z"][0]) + pos_range["z"][0],
        ], dim = 1)
        rand_reset_pos = self.box_center + random_position
        
        random_angles = torch.rand(self.num_envs, device=self.device) * 2 * torch.pi  # 0 ~ 2π 랜덤 값
        rand_reset_rot = torch.stack([
            torch.cos(random_angles / 2),  # w
            torch.zeros(self.num_envs, device=self.device),  # x
            torch.zeros(self.num_envs, device=self.device),  # y
            torch.sin(random_angles / 2)  # z (z축 회전)
        ], dim=1)
        
        rand_reset_box_pose = torch.cat([rand_reset_pos, rand_reset_rot], dim=-1)
        zero_root_velocity = torch.zeros((self.num_envs, 6), device=self.device)

        self._box.write_root_pose_to_sim(rand_reset_box_pose)
        self._box.write_root_velocity_to_sim(zero_root_velocity)
        
        self.cfg.current_time = 0
        self._compute_intermediate_values(env_ids)

    def _get_observations(self) -> dict:
        dof_pos_scaled = (
            2.0
            * (self._robot.data.joint_pos - self.robot_dof_lower_limits)
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
            - 1.0
        )
        to_target = self.box_grasp_pos - self.robot_grasp_pos

        obs = torch.cat(
            (
                dof_pos_scaled,
                self._robot.data.joint_vel * self.cfg.dof_velocity_scale,
                to_target,
                self._box.data.body_link_pos_w[:, 0, 2].unsqueeze(-1),
                self._box.data.body_link_vel_w[:, 0, 2].unsqueeze(-1),
            ),
            dim=-1,
        )
        return {"policy": torch.clamp(obs, -5.0, 5.0),}
    
    # auxiliary methods

    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        hand_pos = self._robot.data.body_link_pos_w[env_ids, self.hand_link_idx]
        hand_rot = self._robot.data.body_link_quat_w[env_ids, self.hand_link_idx]
        
        box_pos = self._box.data.body_link_pos_w[env_ids, self.box_idx]
        box_rot = self._box.data.body_link_quat_w[env_ids, self.box_idx]
        
        (
            self.robot_grasp_rot[env_ids],
            self.robot_grasp_pos[env_ids],
            self.box_grasp_rot[env_ids],
            self.box_grasp_pos[env_ids],
        ) = self._compute_grasp_transforms(
            hand_rot,
            hand_pos,
            self.robot_local_grasp_rot[env_ids],
            self.robot_local_grasp_pos[env_ids],
            box_rot,
            box_pos,
            self.box_local_rot[env_ids],
            self.box_local_pos[env_ids],
        )
        
    def _compute_rewards(
        self,
        actions,
        franka_grasp_pos, 
        box_pos,    
        franka_grasp_rot,
        box_rot,
        gripper_forward_axis,
        gripper_up_axis,
        dist_reward_scale,
        rot_reward_scale,
        action_penalty_scale,
    ):
        joint_penalty_scale = 10.0
        alignment_reward_scale = 10.0
        
        if not hasattr(self, "init_robot_grasp_pos"):
            self.init_robot_grasp_pos = franka_grasp_pos.clone()  # 환경 초기 그리퍼 위치 저장
            
        if not hasattr(self, "init_robot_joint_position"):
            self.init_robot_joint_position = self._robot.data.joint_pos.clone()
            
        if not hasattr(self, "init_robot_grasp_rot"):
            self.init_robot_grasp_rot = franka_grasp_rot.clone()
        
        eps = 1e-6  # NaN 방지용 작은 값
        
        # tracking은 잘 됨
        # ✅ 1. 거리 유지 보상 (그리퍼와 물체 간 거리 일정 유지)
        target_distance = 0.2  # 목표 거리 (예: 20cm)
        distance_error = torch.abs(torch.norm(franka_grasp_pos - box_pos, p=2, dim=-1) - target_distance)
        distance_reward = torch.exp(-distance_error * dist_reward_scale)

        # 잡기축 정의 (그리퍼 초기 위치 → 물체 위치 벡터)
        # grasp_axis = box_pos - self.init_robot_grasp_pos
        grasp_axis = self.init_robot_grasp_pos - box_pos
        grasp_axis = grasp_axis / (torch.norm(grasp_axis, p=2, dim=-1, keepdim=True) + eps)  # 정규화

        # 그리퍼 전방축과 잡기축 정렬 보상
        gripper_forward = tf_vector(franka_grasp_rot, gripper_forward_axis)
        alignment_score = torch.sum(gripper_forward * grasp_axis, dim=-1)  # 내적 계산
        alignment_reward = (alignment_score + 1) / 2  # [-1,1] → [0,1] 변환

        # 그리퍼 위치가 잡기축 위에 있는지 확인
        gripper_proj_dist = torch.norm(torch.cross(franka_grasp_pos - self.init_robot_grasp_pos, grasp_axis), p=2, dim=-1)
        position_alignment_reward = torch.exp(-gripper_proj_dist * alignment_reward_scale)  # 잡기축 벗어나면 패널티

        # 최종 정렬 보상
        total_alignment_reward = 0.7 * alignment_reward + 0.3 * position_alignment_reward

        # ✅ 3. 그리퍼가 초기 자세에서 많이 벗어날수록 패널티 적용 (이상한 자세 방지)
        joint_deviation = torch.abs(self._robot.data.joint_pos - self.init_robot_joint_position)
        joint_penalty = torch.sum(joint_deviation, dim=-1)
        joint_penalty = torch.tanh(joint_penalty)

        # ✅ 4. 행동 크기가 클수록 패널티 적용 (이상한 행동 방지)
        action_penalty = 0.1 * torch.sum(actions**2, dim=-1)

        # ✅ 5. 최종 보상 계산
        rewards = (
            dist_reward_scale * distance_reward  # 거리 유지 보상
            + alignment_reward_scale * total_alignment_reward  # 정렬 보상
            - joint_penalty_scale * joint_penalty  # 자세 안정성 패널티
            - action_penalty_scale * action_penalty  # 행동 크기 패널티
        )

        return rewards
        
        # # ✅ 물체와 일정한 거리 유지 보상
        # target_distance = 0.2  # 20cm
        # distance_error = torch.abs(torch.norm(franka_grasp_pos - box_pos, p=2, dim=-1) - target_distance)
        # distance_reward = torch.exp(-distance_error * (dist_reward_scale * 0.5))

        # # ✅ 그리퍼 전방 축이 물체를 바라보도록 정렬 보상
        # grasp_axis = self.init_robot_grasp_pos - box_pos  
        # grasp_axis = grasp_axis / (torch.norm(grasp_axis, p=2, dim=-1, keepdim=True) + eps)

        # gripper_forward = tf_vector(franka_grasp_rot, gripper_forward_axis)  
        # alignment = torch.sum(gripper_forward * grasp_axis, dim=-1)  
        # grasp_axis = torch.where(alignment.view(-1, 1) < 0, -grasp_axis, grasp_axis)  
        # alignment_reward = (torch.sum(gripper_forward * grasp_axis, dim=-1) + 1) / 2  

        # # ✅ 추가된 코드: 그리퍼 위쪽 축(Up Axis)이 자연스럽게 위쪽을 바라보도록 유도
        # gripper_up = tf_vector(franka_grasp_rot, gripper_up_axis)  

        # # 📌 그리퍼 위쪽 축이 자연스럽게 위쪽을 향하도록 보상 부여
        # desired_up_direction = torch.cross(grasp_axis, gripper_forward)  
        # desired_up_direction = desired_up_direction / (torch.norm(desired_up_direction, p=2, dim=-1, keepdim=True) + eps)

        # up_alignment = torch.sum(gripper_up * desired_up_direction, dim=-1)  
        # up_alignment = torch.clamp(up_alignment, -1, 1)  
        # up_alignment_reward = (up_alignment + 1) / 2  

        # # ✅ 기존 alignment_reward와 up_alignment_reward를 결합하여 최종 정렬 보상
        # alignment_reward = 0.7 * alignment_reward + 0.3 * up_alignment_reward

        # # ✅ 관절 안정성 유지 패널티
        # joint_penalty = torch.exp(-torch.sum(torch.abs(self._robot.data.joint_pos - self.init_robot_joint_position), dim=-1) * 3.0)

        # # ✅ 행동 패널티
        # action_penalty = 0.1 * torch.sum(actions**2, dim=-1)

        # # ✅ 최종 보상 계산
        # rewards = (
        #     dist_reward_scale * distance_reward  
        #     + alignment_reward_scale * alignment_reward  
        #     - joint_penalty_scale * joint_penalty  
        #     - action_penalty_scale * action_penalty  
        # )

        # # ✅ 물체와 일정한 거리 유지 보상 수정
        # target_distance = 0.2  # 10cm
        # distance_error = torch.abs(torch.norm(franka_grasp_pos - box_pos, p=2, dim=-1) - target_distance)
        # distance_reward = torch.exp(-distance_error * (dist_reward_scale * 0.5))

        # # ✅ Alignment 보상 수정 (절대값 활용)
        # grasp_axis = self.init_robot_grasp_pos - box_pos  
        # grasp_axis = grasp_axis / (torch.norm(grasp_axis, p=2, dim=-1, keepdim=True) + eps)

        # # gripper_forward = tf_vector(franka_grasp_rot, gripper_forward_axis)
        # # alignment_reward = (torch.abs(torch.sum(gripper_forward * grasp_axis, dim=-1)) + 1) / 2
        
        # gripper_forward = tf_vector(franka_grasp_rot, gripper_forward_axis)  # 그리퍼 전방 축 벡터
        # alignment = torch.sum(gripper_forward * grasp_axis, dim=-1)  # [-1, 1] 범위
        # grasp_axis = torch.where(alignment.view(-1, 1) < 0, -grasp_axis, grasp_axis)  # 반대 방향이면 뒤집음
        # alignment_reward = (torch.sum(gripper_forward * grasp_axis, dim=-1) + 1) / 2  # 정규화

        # # ✅ 관절 안정성 유지 패널티 수정 (패널티 강도 증가)
        # joint_penalty = torch.exp(-torch.sum(torch.abs(self._robot.data.joint_pos - self.init_robot_joint_position), dim=-1) * 3.0)

        # # ✅ 행동 패널티 수정 (효과 증가)
        # action_penalty = 0.1 * torch.sum(actions**2, dim=-1)

        # # ✅ 최종 보상 계산
        # rewards = (
        #     dist_reward_scale * distance_reward  # 목표 거리 유지 보상
        #     + alignment_reward_scale * alignment_reward  # 동적으로 결정된 잡기 축과 정렬 보상
        #     - joint_penalty_scale * joint_penalty  # 이상한 자세 방지
        #     - action_penalty_scale * action_penalty  # 불필요한 움직임 최소화
        # )

        # 물체와 일정한 거리 유지 보상
        # target_distance = 0.25  # 10 cm
        # distance_error = torch.abs(torch.norm(franka_grasp_pos - box_pos, p=2, dim=-1) - target_distance)
        # distance_reward = torch.exp(-distance_error * dist_reward_scale)
        
        # # 로봇과 물체의 상대 벡터를 기반으로 최적의 잡기 축 계산
        # grasp_axis = self.init_robot_grasp_pos - box_pos  # 동적 잡기 축 결정
        # grasp_axis = grasp_axis / (torch.norm(grasp_axis, p=2, dim=-1, keepdim=True) + eps)
        # # print(f"box_pos : {box_pos}")

        # # 그리퍼 전방 축이 잡기 축과 정렬되도록 보상 적용
        # gripper_forward = tf_vector(franka_grasp_rot, gripper_forward_axis)  # 그리퍼 전방 축 벡터
        # alignment = torch.sum(gripper_forward * grasp_axis, dim=-1)  # [-1, 1] 범위
        # grasp_axis = torch.where(alignment.view(-1, 1) < 0, -grasp_axis, grasp_axis)  # 반대 방향이면 뒤집음
        # alignment_reward = (torch.sum(gripper_forward * grasp_axis, dim=-1) + 1) / 2  # 정규화

        # # 관절 안정성 유지 (이상한 자세 방지)
        # joint_deviation = torch.abs(self._robot.data.joint_pos - self.init_robot_joint_position)
        # joint_penalty = torch.sum(joint_deviation, dim=-1)
        # joint_penalty = torch.tanh(joint_penalty)

        # # 행동 패널티 (불필요한 움직임 최소화)
        # action_penalty = torch.sum(actions**2, dim=-1)
        # action_penalty = torch.tanh(action_penalty)

        # # 최종 보상 계산
        # rewards = (
        #     dist_reward_scale * distance_reward  # 목표 거리 유지 보상
        #     + alignment_reward_scale * alignment_reward  # 동적으로 결정된 잡기 축과 정렬 보상
        #     - joint_penalty_scale * joint_penalty  # 이상한 자세 방지
        #     - action_penalty_scale * action_penalty  # 불필요한 움직임 최소화
        # )
        
        # if not hasattr(self, "init_robot_grasp_pos"):
        #     self.init_robot_grasp_pos = franka_grasp_pos.clone()  # 환경 초기 그리퍼 위치 저장
            
        # if not hasattr(self, "init_robot_joint_position"):
        #     self.init_robot_joint_position = self._robot.data.joint_pos.clone()

        # eps = 1e-6  # NaN 방지용 작은 값
        
        # # 그리퍼에서 물체로 향하는 방향으로 grasp_axis 수정
        # grasp_axis = self.init_robot_grasp_pos - box_pos  
        # grasp_axis = grasp_axis / (torch.norm(grasp_axis, p=2, dim=-1, keepdim=True) + eps)

        # # 그리퍼 전방 축 벡터 계산
        # gripper_forward = tf_vector(franka_grasp_rot, gripper_forward_axis)

        # # 그리퍼 정렬 보상 수정
        # alignment = torch.sum(gripper_forward * grasp_axis, dim=-1)  # [-1, 1] 범위
        # grasp_axis = torch.where(alignment.view(-1, 1) < 0, -grasp_axis, grasp_axis)  # 반대 방향이면 뒤집음
        # alignment_reward = (torch.sum(gripper_forward * grasp_axis, dim=-1) + 1) / 2  # 정규화

        # # 물체와 일정한 거리 유지 보상
        # target_distance = 0.25  # 30 cm
        # distance_error = torch.abs(torch.norm(franka_grasp_pos - box_pos, p=2, dim=-1) - target_distance)
        # distance_reward = torch.exp(-distance_error * dist_reward_scale)

        # # 관절 안정성 유지 (이상한 자세 방지)
        # joint_deviation = torch.abs(self._robot.data.joint_pos - self.init_robot_joint_position)
        # # joint_penalty = torch.sum(joint_deviation, dim=-1)
        # # joint_penalty = torch.tanh(joint_penalty)
        # joint_penalty = torch.exp(-torch.sum(joint_deviation, dim=-1) * 2.0)

        # # 행동 패널티 (불필요한 움직임 최소화)
        # action_penalty = torch.sum(actions**2, dim=-1)
        # action_penalty = torch.tanh(action_penalty)

        # # 최종 보상 계산
        # rewards = (
        #     dist_reward_scale * distance_reward  # 목표 거리 유지 보상
        #     + alignment_reward_scale * alignment_reward  # 동적으로 결정된 잡기 축과 정렬 보상
        #     - joint_penalty_scale * joint_penalty  # 이상한 자세 방지
        #     - action_penalty_scale * action_penalty  # 불필요한 움직임 최소화
        # )

        # 로봇과 물체의 상대 벡터를 기반으로 최적의 잡기 축 계산
        # grasp_axis = self.init_robot_grasp_pos - box_pos  # 동적 잡기 축 결정
        # grasp_axis = grasp_axis / (torch.norm(grasp_axis, p=2, dim=-1, keepdim=True) + eps)
        # # print(f"box_pos : {box_pos}")

        # # 그리퍼 전방 축이 잡기 축과 정렬되도록 보상 적용
        # gripper_forward = tf_vector(franka_grasp_rot, gripper_forward_axis)  # 그리퍼 전방 축 벡터
        # alignment_reward = torch.sum(gripper_forward * grasp_axis, dim=-1)  # 내적 계산
        # alignment_reward = (alignment_reward + 1) / 2  # [-1,1] → [0,1] 변환
        # # print(f"alignment_reward : {alignment_reward}")
        
        # # 물체와 일정한 거리 유지 보상
        # target_distance = 0.3  # 10 cm
        # distance_error = torch.abs(torch.norm(franka_grasp_pos - box_pos, p=2, dim=-1) - target_distance)
        # distance_reward = torch.exp(-distance_error * dist_reward_scale)

        # # 관절 안정성 유지 (이상한 자세 방지)
        # joint_deviation = torch.abs(self._robot.data.joint_pos - self.init_robot_joint_position)
        # joint_penalty = torch.sum(joint_deviation, dim=-1)
        # joint_penalty = torch.tanh(joint_penalty)

        # # 행동 패널티 (불필요한 움직임 최소화)
        # action_penalty = torch.sum(actions**2, dim=-1)
        # action_penalty = torch.tanh(action_penalty)

        # # 최종 보상 계산
        # rewards = (
        #     dist_reward_scale * distance_reward  # 목표 거리 유지 보상
        #     + alignment_reward_scale * alignment_reward  # 동적으로 결정된 잡기 축과 정렬 보상
        #     - joint_penalty_scale * joint_penalty  # 이상한 자세 방지
        #     - action_penalty_scale * action_penalty  # 불필요한 움직임 최소화
        # )
        
        # joint_penalty_scale = 0.1
        
        # # 물체와 일정한 거리 유지 보상
        # target_distance = 0.01  # 10cm 
        # distance_error = torch.abs(torch.norm(franka_grasp_pos - box_pos, p=2, dim=-1) - target_distance)
        # distance_reward = torch.exp(-distance_error * dist_reward_scale)

        # # 그리퍼 전방 축이 잡기 축과 정렬되도록 보상 적용
        # gripper_forward_vect = tf_vector(franka_grasp_rot, gripper_forward_axis)
        # grasp_axis = box_pos - self.init_robot_grasp_pos
        # grasp_axis = grasp_axis / (torch.norm(grasp_axis, p=2, dim=-1, keepdim=True) + 1e-6)
        # rot_reward = (torch.sum(gripper_forward_vect * grasp_axis, dim=-1) + 1)/2
        
        # # 관절 안정성 유지 (이상한 자세 방지)
        # joint_deviation = torch.abs(self._robot.data.joint_pos - self.init_robot_joint_position)
        # joint_penalty = torch.sum(joint_deviation, dim=-1)
        # joint_threshold = 0.5  
        # joint_penalty = torch.where(
        #     joint_penalty > joint_threshold,  
        #     torch.tanh(joint_penalty),  
        #     torch.zeros_like(joint_penalty)  
        # )

        # action_penalty = torch.sum(actions**2, dim=-1)

        # rewards = (
        #     dist_reward_scale * distance_reward + 
        #     rot_reward_scale * rot_reward + 
        #     - action_penalty_scale * action_penalty
        #     - joint_penalty_scale * joint_penalty
        # )

        return rewards

    def _compute_grasp_transforms(
        self,
        hand_rot,
        hand_pos,
        franka_local_grasp_rot,
        franka_local_grasp_pos,
        box_rot,
        box_pos,
        box_local_rot,
        box_local_pos,

    ):
        global_franka_rot, global_franka_pos = tf_combine(
            hand_rot, hand_pos, franka_local_grasp_rot, franka_local_grasp_pos
        )
        
        global_box_rot, global_box_pos = tf_combine(
            box_rot, box_pos, box_local_rot, box_local_pos
        )

        return global_franka_rot, global_franka_pos, global_box_rot, global_box_pos
        
        
        