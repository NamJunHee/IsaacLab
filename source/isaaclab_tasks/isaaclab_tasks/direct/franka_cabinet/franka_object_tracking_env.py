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
from enum import Enum
import kornia

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from vision_msgs.msg import Detection3DArray
from geometry_msgs.msg import Pose

from cv_bridge import CvBridge
import threading
import time

class ObjectMoveType(Enum):
    STATIC = "static"
    CIRCLE = "circle"
    LINEAR = "linear"

object_move = ObjectMoveType.LINEAR

training_mode = False
foundationpose_mode = True

image_publish = True
camera_enable = True

robot_action = False
robot_init_pose = False
    
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
                "panda_joint2": -0.831,
                "panda_joint3": -0.000,
                "panda_joint4": -1.796,
                "panda_joint5": -0.000,
                "panda_joint6": 2.033,
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
                # velocity_limit=2.175,
                velocity_limit=0.5,
                # stiffness=80.0,
                stiffness=200.0,
                # damping=4.0,
                damping=10.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit=12.0,
                # velocity_limit=2.61,
                velocity_limit=0.5,
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
    if camera_enable:
        camera = CameraCfg(
            prim_path="/World/envs/env_.*/Robot/panda_hand/hand_camera", 
            update_period=0.03,
            height=480,
            width=640,
            # height=800,
            # width=600,
            # height=1080,
            # width=1920,
            data_types=["rgb", "depth"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=35.0,
                # focal_length=65.0,
                focus_distance=600.0,
                horizontal_aperture=50.0,
                clipping_range=(0.1, 1.0e5),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.0, 0.0, 0.05),
                rot=(0.0, 0.707, 0.707, 0.0),
                convention="ROS",
            )
        )
    
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
                # usd_path="/home/nmail-njh/NMAIL/01_Project/Robot_Grasping/objects_usd/google_objects_usd/005_tomato_soup_can/005_tomato_soup_can.usd",
                usd_path="/home/nmail-njh/NMAIL/01_Project/Robot_Grasping/objects_usd/google_objects_usd/006_mustard_bottle/006_mustard_bottle.usd",
                # usd_path="/home/nmail-njh/NMAIL/01_Project/Robot_Grasping/objects_usd/google_objects_usd/004_sugar_box/004_sugar_box.usd",
                # usd_path="/home/nmail-njh/NMAIL/01_Project/Robot_Grasping/objects_usd/google_objects_usd/025_mug/025_mug.usd",
                # usd_path="/home/nmail-njh/NMAIL/01_Project/Robot_Grasping/objects_usd/google_objects_usd/Travel_Mate_P_series_Notebook/Travel_Mate_P_series_Notebook.usd",
                # usd_path="/home/nmail-njh/NMAIL/01_Project/Robot_Grasping/objects_usd/google_objects_usd/Mens_ASV_Billfish_Boat_Shoe_in_Dark_Brown_Leather_zdHVHXueI3w/Mens_ASV_Billfish_Boat_Shoe_in_Dark_Brown_Leather_zdHVHXueI3w.usd",
                
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
    
    # action_scale = 7.5
    # dof_velocity_scale = 0.1
    action_scale = 2.0
    dof_velocity_scale = 0.05

    # reward scales
    # dist_reward_scale = 1.5
    # rot_reward_scale = 1.5
    # open_reward_scale = 10.0
    # action_penalty_scale = 0.05
    # finger_reward_scale = 2.0
    
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
        
        self.rand_pos_range = {
            "x" : ( -0.3,   0.35),
            "y" : ( -0.45,  0.45),
            "z" : (  0.055, 0.3)
        }
        self.fixed_z = 0.055
        
        self.current_box_pos = None
        self.current_box_rot = None
        
        self.target_box_pos = torch.stack([
                torch.rand(self.num_envs, device=self.device) * (self.rand_pos_range["x"][1] - self.rand_pos_range["x"][0]) + self.rand_pos_range["x"][0],
                torch.rand(self.num_envs, device=self.device) * (self.rand_pos_range["y"][1] - self.rand_pos_range["y"][0]) + self.rand_pos_range["y"][0],
                torch.rand(self.num_envs, device=self.device) * (self.rand_pos_range["z"][1] - self.rand_pos_range["z"][0]) + self.rand_pos_range["z"][0],
            ], dim = 1)
        
        self.target_box_pos = self.target_box_pos + self.box_center
        # self.rand_pos_step = 0
        # self.new_box_pos_rand = self._box.data.body_link_pos_w[:,0,:].clone()
        
        self.speed = 0.001
        # self.speed = 0.0005
        
        rclpy.init()
        self.last_publish_time = 0.0
        self.position_error = 0.0
        self.obj_origin_distance = 0.0
        
        if image_publish:
            
            qos_profile = QoSProfile(depth=10)
            qos_profile.reliability = QoSReliabilityPolicy.BEST_EFFORT
            
            # self.node = rclpy.create_node('isaac_camera_publisher')
            # self.camera_info_publisher = self.node.create_publisher(CameraInfo, '/isaac_camera_info_rect',10)
            # self.rgb_publisher = self.node.create_publisher(Image, '/isaac_image_rect',10)
            # self.depth_publisher = self.node.create_publisher(Image, '/isaac_depth',10)

            self.node = rclpy.create_node('camera_publisher')
            self.camera_info_publisher = self.node.create_publisher(CameraInfo, '/camera_info_rect',10)
            self.rgb_publisher = self.node.create_publisher(Image, '/image_rect',10)
            self.depth_publisher = self.node.create_publisher(Image, '/depth',10)
            
            self.bridge = CvBridge()
        
        if foundationpose_mode:
            self.latest_detection_msg = None
            self.foundationpose_node = rclpy.create_node('foundationpose_receiver')
            self.foundationpose_node.create_subscription(
                Detection3DArray,
                # '/centerpose/detections',
                '/tracking/output',
                self.foundationpose_callback,
                10
            )
        
        self.init_cnt = 0
        
    def publish_camera_data(self):
        env_id = 0
        
        current_stamp = self.node.get_clock().now().to_msg() 
        current_stamp.sec = current_stamp.sec % 50000
        current_stamp.nanosec = 0
        
        # current_stamp = Time()
        # current_stamp.sec = 1
        # current_stamp.nanosec = 0
                
        if image_publish:            
            rgb_data = self._camera.data.output["rgb"]
            depth_data = self._camera.data.output["depth"]
            
            rgb_image = (rgb_data.cpu().numpy()[env_id]).astype(np.uint8)
            depth_image = (depth_data.cpu().numpy()[env_id]).astype(np.float32)

            # Publish Camera Info
            camera_info_msg = CameraInfo()
            camera_info_msg.header.stamp = current_stamp
            # camera_info_msg.header.stamp = zero_time
            
            camera_info_msg.header.frame_id = 'tf_camera'
            # camera_info_msg.header.frame_id = 'camera_frame'
        
            camera_info_msg.height = 480 
            camera_info_msg.width = 640 
            # camera_info_msg.height = 800 
            # camera_info_msg.width = 600 
            # camera_info_msg.height = 1080
            # camera_info_msg.width = 1920 
            
            camera_info_msg.distortion_model = 'plumb_bob'
        
            intrinsic_matrices = self._camera.data.intrinsic_matrices.cpu().numpy().flatten().tolist()
            camera_info_msg.k = intrinsic_matrices[:9]
            camera_info_msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
            camera_info_msg.r = [1.0, 0.0, 0.0,
                                 0.0, 1.0, 0.0,
                                 0.0, 0.0, 1.0]
            camera_info_msg.p = intrinsic_matrices[:3] + [0.0] + intrinsic_matrices[3:6] + [0.0] + [0.0, 0.0, 1.0, 0.0]

            camera_info_msg.binning_x = 0
            camera_info_msg.binning_y = 0

            camera_info_msg.roi.x_offset = 0
            camera_info_msg.roi.y_offset = 0
            camera_info_msg.roi.height = 0
            camera_info_msg.roi.width = 0
            camera_info_msg.roi.do_rectify = False
        
            self.camera_info_publisher.publish(camera_info_msg)
        
            # Publish RGB Image
            rgb_msg = self.bridge.cv2_to_imgmsg(rgb_image, encoding='rgb8')
            rgb_msg.header.stamp = current_stamp
            # rgb_msg.header.stamp = zero_time
            rgb_msg.header.frame_id = 'tf_camera'
            # rgb_msg.header.frame_id = 'camera_frame'
            self.rgb_publisher.publish(rgb_msg)
            # self.node.get_logger().info('Published RGB image')

            # Publish Depth Image
            depth_msg = self.bridge.cv2_to_imgmsg(depth_image, encoding='32FC1')
            depth_msg.header.stamp = current_stamp
            # depth_msg.header.stamp = zero_time
            depth_msg.header.frame_id = 'tf_camera'
            # depth_msg.header.frame_id = 'camera_frame'
            self.depth_publisher.publish(depth_msg)
            depth_msg.step = depth_image.shape[1] * 4
            # self.node.get_logger().info('Published Depth image')
    
    def subscribe_object_pos(self):
        msg = self.latest_detection_msg
        
        if msg is None:
            # print("No detection message received.")
            return None

        if not msg.detections:
            # print("No detections found in message.")
            return None

        if not msg.detections[0].results:
            # print("No detection results found.")
            return None

        pos = msg.detections[0].results[0].pose.pose.position
        # print(f"pos : {pos}")
        return torch.tensor([pos.x, pos.y, pos.z], device=self.device)
    
    def foundationpose_callback(self,msg):
        self.latest_detection_msg = msg
    
    def sample_target_box_pos(self):
        self.rand_pos_range_center = {
            "x" : (-0.1, 0.2),
            "y" : (-0.2, 0.2),
            "z" : ( 0.1, 0.2)
        }
        self.rand_pos_range_edge = {
            "x_R" : (-0.3,  -0.1),
            "x_L" : ( 0.25,  0.3),
            "y_R" : (-0.35, -0.2),
            "y_L" : ( 0.2,   0.35),
            "z_R" : ( 0.055, 0.3),
            "z_L" : ( 0.055, 0.1),
            "z_R" : ( 0.2,   0.3)
        }
        self.fixed_z = 0.055
        
        rand_vals = torch.rand(self.num_envs, device=self.device)
        is_edge = rand_vals < 0.6 # 0.5 
        
        x_center = torch.rand(self.num_envs, device=self.device) * (self.rand_pos_range_center["x"][1] - self.rand_pos_range_center["x"][0]) + self.rand_pos_range_center["x"][0]
        x_edge = torch.where(torch.rand(self.num_envs, device=self.device) > 0.5,  
                         torch.rand(self.num_envs, device=self.device) * (self.rand_pos_range_edge["x_R"][1] - self.rand_pos_range_edge["x_R"][0]) + self.rand_pos_range_edge["x_R"][0],  # 왼쪽 가장자리
                         torch.rand(self.num_envs, device=self.device) * (self.rand_pos_range_edge["x_L"][1] - self.rand_pos_range_edge["x_L"][0]) + self.rand_pos_range_edge["x_L"][0])  # 오른쪽 가장자리
        
        y_center = torch.rand(self.num_envs, device=self.device) * (self.rand_pos_range_center["y"][1] - self.rand_pos_range_center["y"][0]) + self.rand_pos_range_center["y"][0]
        y_edge = torch.where(torch.rand(self.num_envs, device=self.device) > 0.5,
                             torch.rand(self.num_envs, device=self.device) * (self.rand_pos_range_edge["y_R"][1] - self.rand_pos_range_edge["y_R"][0]) + self.rand_pos_range_edge["y_R"][0],  # 아래쪽 가장자리
                             torch.rand(self.num_envs, device=self.device) * (self.rand_pos_range_edge["y_L"][1] - self.rand_pos_range_edge["y_L"][0]) + self.rand_pos_range_edge["y_L"][0])  # 위쪽 가장자리
        
        z_center = torch.rand(self.num_envs, device=self.device) * (self.rand_pos_range_center["z"][1] - self.rand_pos_range_center["z"][0]) + self.rand_pos_range_center["z"][0]
        z_edge = torch.where(torch.rand(self.num_envs, device=self.device) > 0.5,
                             torch.rand(self.num_envs, device=self.device) * (self.rand_pos_range_edge["z_R"][1] - self.rand_pos_range_edge["z_R"][0]) + self.rand_pos_range_edge["z_R"][0],  # 아래쪽 가장자리
                             torch.rand(self.num_envs, device=self.device) * (self.rand_pos_range_edge["z_L"][1] - self.rand_pos_range_edge["z_L"][0]) + self.rand_pos_range_edge["z_L"][0])  # 위쪽 가장자리
        
        x_final = torch.where(is_edge, x_edge, x_center)
        y_final = torch.where(is_edge, y_edge, y_center)
        z_final = torch.where(is_edge, z_edge, z_center)
        
        target_box_pos = torch.stack([x_final, y_final, z_final], dim=1)
        target_box_pos = target_box_pos + self.box_center
        
        self.rand_pos_step = 0
        self.new_box_pos_rand = self._box.data.body_link_pos_w[:,0,:].clone()
        self.speed = 0.8
        
        return target_box_pos
    
    def quat_mul(self, q, r):
        x1, y1, z1, w1 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        x2, y2, z2, w2 = r[:, 0], r[:, 1], r[:, 2], r[:, 3]

        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2

        quat = torch.stack((x, y, z, w), dim=-1)
        return kornia.geometry.quaternion.normalize_quaternion(quat)
    
    def quat_conjugate(self, q):
        """
        쿼터니언 켤레(Conjugate)를 반환
        Args:
        - q: (batch_size, 4)
        Returns:
        - conjugate_q: (batch_size, 4)
        """
        q_conj = torch.cat([-q[:, :3], q[:, 3:4]], dim=-1)
        return q_conj
    
    def compute_camera_world_pose(self, hand_pos, hand_rot):

        cam_offset_pos = torch.tensor([0.0, 0.0, 0.05], device=hand_pos.device).repeat(self.num_envs, 1)

        hand_rot_matrix = kornia.geometry.quaternion.quaternion_to_rotation_matrix(hand_rot)
        cam_offset_pos_world = torch.bmm(hand_rot_matrix, cam_offset_pos.unsqueeze(-1)).squeeze(-1)

        camera_pos_w = hand_pos + cam_offset_pos_world
        camera_pos_w = camera_pos_w - self.scene.env_origins

        return camera_pos_w

    def world_to_camera_pose(self, camera_pos_w, camera_rot_w, obj_pos_w, obj_rot_w):
        rel_pos = obj_pos_w - camera_pos_w

        cam_rot_matrix = kornia.geometry.quaternion.quaternion_to_rotation_matrix(camera_rot_w)
        
        obj_pos_cam = torch.bmm(cam_rot_matrix.transpose(1, 2), rel_pos.unsqueeze(-1)).squeeze(-1)

        cam_rot_inv = self.quat_conjugate(camera_rot_w)
        obj_rot_cam = self.quat_mul(cam_rot_inv, obj_rot_w)

        return obj_pos_cam, obj_rot_cam
    
    def camera_to_world_pose(self, camera_pos_w, camera_rot_w, obj_pos_cam, obj_rot_cam):
        cam_rot_matrix = kornia.geometry.quaternion.quaternion_to_rotation_matrix(camera_rot_w)
        
        obj_pos_world = torch.bmm(cam_rot_matrix, obj_pos_cam.unsqueeze(-1)).squeeze(-1) + camera_pos_w
        obj_rot_world = self.quat_mul(camera_rot_w, obj_rot_cam)
        
        return obj_pos_world, obj_rot_world
        
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
        if camera_enable:
            self._camera = Camera(self.cfg.camera)
            self.scene.sensors["hand_camera"] = self._camera
        
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
        
        # if not hasattr(self, "frame_start_time"):
        #     self.frame_start_time = time.time()
        #     self.frame_count = 0
        # else:
        #     self.frame_count += 1
        #     if self.frame_count % 60 == 0:
        #         elapsed = time.time() - self.frame_start_time
        #         # print(f"Simulated FPS: {self.frame_count / elapsed:.2f}")
        #         self.frame_start_time = time.time()
        #         self.frame_count = 0
        
        # 카메라 ros2 publish----------------------------------------------------------------------------------------------
        if image_publish:   
            self.last_publish_time += self.dt
            if self.last_publish_time >= (1.0 / 15.0):  # 정확히 30fps 기준
                self.publish_camera_data()
                rclpy.spin_once(self.node, timeout_sec=0.001)
                self.last_publish_time = 0.0

        # 물체 원 운동 (실제 운동 제어 코드)-------------------------------------------------------------------------------------------
        if object_move == ObjectMoveType.CIRCLE:
            R = 0.10
            omega = 0.7 # Speed

            offset_x = R * torch.cos(omega * current_time) - 0.35
            offset_y = R * torch.sin(omega * current_time) 
            offset_z = 0.055

            offset_pos = torch.tensor([offset_x, offset_y, offset_z], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)

            new_box_pos_circle = self.box_center + offset_pos
            new_box_rot_circle = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device, dtype=torch.float32).unsqueeze(0).repeat(self.num_envs, 1)

            new_box_pose_circle = torch.cat([new_box_pos_circle, new_box_rot_circle], dim = -1)

            self._box.write_root_pose_to_sim(new_box_pose_circle)
        
        # 물체 위치 랜덤 선형 이동 --------------------------------------------------------------------------------------------------
        if object_move == ObjectMoveType.LINEAR:
            distance_to_target = torch.norm(self.target_box_pos - self.new_box_pos_rand, p=2, dim = -1)
            if torch.any(distance_to_target < 0.01):
                self.target_box_pos = torch.stack([
                torch.rand(self.num_envs, device=self.device) * (self.rand_pos_range["x"][1] - self.rand_pos_range["x"][0]) + self.rand_pos_range["x"][0],
                torch.rand(self.num_envs, device=self.device) * (self.rand_pos_range["y"][1] - self.rand_pos_range["y"][0]) + self.rand_pos_range["y"][0],
                torch.rand(self.num_envs, device=self.device) * (self.rand_pos_range["z"][1] - self.rand_pos_range["z"][0]) + self.rand_pos_range["z"][0],
                ], dim = 1)

                self.target_box_pos = self.target_box_pos + self.box_center

                self.current_box_pos = self._box.data.body_link_pos_w[:, 0, :].clone()
                self.current_box_rot = self._box.data.body_link_quat_w[:, 0, :].clone()

                self.new_box_pos_rand = self.current_box_pos

                direction = self.target_box_pos - self.current_box_pos
                direction_norm = torch.norm(direction, p=2, dim=-1, keepdim=True) + 1e-6
                self.rand_pos_step = (direction / direction_norm * self.speed)

            self.new_box_pos_rand = self.new_box_pos_rand + self.rand_pos_step
            new_box_rot_rand = self.current_box_rot 

            new_box_pose_rand = torch.cat([self.new_box_pos_rand, new_box_rot_rand], dim = -1)
            self._box.write_root_pose_to_sim(new_box_pose_rand)
        
    def _apply_action(self):
        
        global robot_action
        global robot_init_pose
        
        # print(f"robot_action : {robot_action}")
        # print(f"robot_init_pose : {robot_init_pose}")

        if training_mode == False:
            
            if robot_action and robot_init_pose:
                target_pos = self.robot_dof_targets.clone()

                joint7_index = self._robot.find_joints(["panda_joint7"])[0]
                target_pos[:, joint7_index] = 0.

                self._robot.set_joint_position_target(target_pos)

            elif (robot_action == False) and (robot_init_pose == False):
                init_pos = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

                joint_names = [
                "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
                "panda_joint5", "panda_joint6", "panda_joint7",
                "panda_finger_joint1", "panda_finger_joint2"
                ]
                # joint_values = [0.000, -0.831, 0.000, -1.796, 0.000, 2.033, 0.707, 0.035, 0.035]
                joint_values = [0.000, -0.831, 0.000, -1.796, 0.000, 1.600, 0.707, 0.035, 0.035]

                for name, val in zip(joint_names, joint_values):
                    index = self._robot.find_joints(name)[0]
                    init_pos[:, index] = val

                self._robot.set_joint_position_target(init_pos)

                joint_err = torch.abs(self._robot.data.joint_pos - init_pos)
                max_err = torch.max(joint_err).item()

                if foundationpose_mode:
                    pos = self.subscribe_object_pos()
                    if (max_err < 0.1) and (pos is not None):
                        self.init_cnt += 1
                        if self.init_cnt > 300 :#and self.position_error < 0.06:
                            # robot_action = True
                            robot_init_pose = True
                            self.robot_dof_targets[:] = init_pos 
                            
                elif foundationpose_mode == False and max_err < 0.1:
                    robot_init_pose = True
                    robot_action = True
                    
        else:            
            target_pos = self.robot_dof_targets.clone()

            joint7_index = self._robot.find_joints(["panda_joint7"])[0]
            target_pos[:, joint7_index] = 0.707

            self._robot.set_joint_position_target(self.robot_dof_targets)
        
    # post-physics step calls

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        
        terminated = self._box.data.body_link_pos_w[:, 0,2] > 0.3
        if {training_mode} | {object_move == ObjectMoveType.CIRCLE}:
            # truncated = self.episode_length_buf >= self.max_episode_length # 물체 원운동 환경 초기화 주기
            truncated = self.episode_length_buf >= self.max_episode_length - 400 # 물체 램덤 생성 환경 초기화 주기 (초기 움직임 제한 학습)

        else:
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

        camera_pos_w = self.compute_camera_world_pose(self.robot_grasp_pos, self.robot_grasp_rot)
        camera_rot_w = self.robot_grasp_rot
        
        box_pos_cam, box_rot_cam = self.world_to_camera_pose(
            camera_pos_w, camera_rot_w,
            self.box_grasp_pos - self.scene.env_origins, self.box_grasp_rot,
        )
        
        return self._compute_rewards(
            self.actions,
            self.robot_grasp_pos,
            self.box_grasp_pos,
            self.robot_grasp_rot,
            self.box_grasp_rot,
            box_pos_cam,
            box_rot_cam,
            self.gripper_forward_axis,
            self.gripper_up_axis,
        )

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        
        # robot state ---------------------------------------------------------------------------------
        if training_mode:
            joint_pos = self._robot.data.default_joint_pos[env_ids] + sample_uniform(
                -0.125,
                0.125,
                (len(env_ids), self._robot.num_joints),
                self.device,
            )
            joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
            joint_vel = torch.zeros_like(joint_pos)
            self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
            self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        
        # 물체 원 운동 (원 운동 시 환경 초기화 코드)------------------------------------------------------------------------------------------------------------
        reset_pos = self.box_center
        reset_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device, dtype=torch.float32).unsqueeze(0).repeat(self.num_envs, 1)
        reset_box_pose = torch.cat([reset_pos, reset_rot], dim = -1)
        
        if object_move == ObjectMoveType.CIRCLE:
            self._box.write_root_pose_to_sim(reset_box_pose)
        
        # 물체 위치 랜덤 생성 (실제 물체 생성 코드) -----------------------------------------------------------------------------------------------------------
        self.rand_pos = torch.stack([
            torch.rand(self.num_envs, device=self.device) * (self.rand_pos_range["x"][1] - self.rand_pos_range["x"][0]) + self.rand_pos_range["x"][0],
            torch.rand(self.num_envs, device=self.device) * (self.rand_pos_range["y"][1] - self.rand_pos_range["y"][0]) + self.rand_pos_range["y"][0],
            torch.rand(self.num_envs, device=self.device) * (self.rand_pos_range["z"][1] - self.rand_pos_range["z"][0]) + self.rand_pos_range["z"][0],
        ], dim = 1)
        
        rand_reset_pos = self.rand_pos + self.box_center
        
        random_angles = torch.rand(self.num_envs, device=self.device) * 2 * torch.pi  # 0 ~ 2π 랜덤 값
        rand_reset_rot = torch.stack([
            torch.cos(random_angles / 2),  # w
            torch.zeros(self.num_envs, device=self.device),  # x
            torch.zeros(self.num_envs, device=self.device),  # y
            torch.sin(random_angles / 2)  # z (z축 회전)
        ], dim=1)
        
        rand_reset_box_pose = torch.cat([rand_reset_pos, rand_reset_rot], dim=-1)
        zero_root_velocity = torch.zeros((self.num_envs, 6), device=self.device)

        if object_move == ObjectMoveType.STATIC:
            self._box.write_root_pose_to_sim(rand_reset_box_pose)
            self._box.write_root_velocity_to_sim(zero_root_velocity)
        
        # 물체 위치 선형 랜덤 이동----------------------------------------------------------------
        if object_move == ObjectMoveType.LINEAR:
            self.new_box_pos_rand = self._box.data.body_link_pos_w[:, 0, :].clone()
            self.current_box_rot = self._box.data.body_link_quat_w[:, 0, :].clone()

            # self.new_box_pos_rand = self.current_box_pos
            # self.target_box_pos = self.rand_pos

            direction = self.target_box_pos - self.new_box_pos_rand
            direction_norm = torch.norm(direction, p=2, dim=-1, keepdim=True) + 1e-6
            self.rand_pos_step = (direction / direction_norm * self.speed)
        #--------------------------------------------------------------------------------
               
        self.cfg.current_time = 0
        self._compute_intermediate_values(env_ids)

    def _get_observations(self) -> dict:
        
        dof_pos_scaled = (
            2.0
            * (self._robot.data.joint_pos - self.robot_dof_lower_limits)
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
            - 1.0
        )
        
        global robot_action
        
        camera_pos_w = self.compute_camera_world_pose(self.robot_grasp_pos, self.robot_grasp_rot)
        camera_rot_w = self.robot_grasp_rot
                
        if foundationpose_mode:
            rclpy.spin_once(self.foundationpose_node, timeout_sec=0.01)
            pos = self.subscribe_object_pos()
        
            if (pos is not None) and robot_init_pose:
            
                camera_pos_w = self.compute_camera_world_pose(self.robot_grasp_pos, self.robot_grasp_rot)
                camera_rot_w = self.robot_grasp_rot
                
                # box_pos_cam, box_rot_cam = self.world_to_camera_pose(camera_pos_w, camera_rot_w, self.box_grasp_pos - self.scene.env_origins, self.box_grasp_rot,)
                
                foundationpose_pos = pos.repeat(self.num_envs, 1)
                
                foundationpose_pos_converted = torch.zeros_like(foundationpose_pos)
                foundationpose_pos_converted[:, 0] = -foundationpose_pos[:, 1]  # x = -y_fp
                foundationpose_pos_converted[:, 1] =  foundationpose_pos[:, 0]  # y = x_fp
                foundationpose_pos_converted[:, 2] =  foundationpose_pos[:, 2]  # z = z_fp
                
                fp_world_pos, _ = self.camera_to_world_pose(camera_pos_w, camera_rot_w, foundationpose_pos_converted, self.box_grasp_rot,)

                # print(f"isaac_cam_pos : {box_pos_cam}")
                print(f"fp_cam_pos : {foundationpose_pos_converted}")
                # print(f"isaac_world_pos : {self.box_grasp_pos}")
                print(f"fp_world_pos : {fp_world_pos}")
                
                origin = torch.zeros_like(self.box_grasp_pos)
                self.position_error = torch.norm(self.box_grasp_pos - fp_world_pos, dim=-1)
                self.obj_origin_distance = torch.norm(origin - fp_world_pos, dim=-1)
                
                print(f"Position error : {self.position_error.mean().item()}")
                print(f"obj_origin_distance : {self.obj_origin_distance.mean().item()}")
                
                if self.position_error < 0.06:
                    robot_action = True
                elif self.position_error is None or self.position_error >= 0.06:
                    robot_action = False
                    
                to_target = fp_world_pos - self.robot_grasp_pos
                # to_target = self.box_grasp_pos - self.robot_grasp_pos 
                
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
                # return {"policy": torch.clamp(obs, -5.0, 5.0),}
                    
            else:
                robot_action = False
                to_target = self.box_grasp_pos - self.robot_grasp_pos 
        else:
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
        
        box_pos_world = self._box.data.body_link_pos_w[env_ids, self.box_idx]
        box_rot_world = self._box.data.body_link_quat_w[env_ids, self.box_idx]
        
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
            box_rot_world,
            box_pos_world,
            self.box_local_rot[env_ids],
            self.box_local_pos[env_ids],
        )
        
    def _compute_rewards(
        self,
        actions,
        franka_grasp_pos, 
        box_pos_w,    
        franka_grasp_rot,
        box_rot_w,
        box_pos_cam,
        box_rot_cam,
        gripper_forward_axis,
        gripper_up_axis,
    ):
        distance_reward_scale = 8.0
        alignment_reward_scale = 12.0
        pview_reward_scale = 10.0  
        # cam_up_alignment_reward_scale = 12.0
        joint_penalty_scale = 6.0
        action_penalty_scale = 12.0
        
        # tracking은 잘되는 보상 함수(단, 카메라 시야 수평 고정 x)
        if not hasattr(self, "init_robot_grasp_pos"):
            self.init_robot_grasp_pos = franka_grasp_pos.clone()  
            
        if not hasattr(self, "init_robot_joint_position"):
            self.init_robot_joint_position = self._robot.data.joint_pos.clone()
            
        if not hasattr(self, "init_robot_grasp_rot"):
            self.init_robot_grasp_rot = franka_grasp_rot.clone()
        
        eps = 1e-6  # NaN 방지용 작은 값
        
        # 거리 유지 보상 (그리퍼와 물체 간 거리 일정 유지)
        target_distance = 0.15  # 목표 거리 (예: 20cm)
        distance_error = torch.abs(torch.norm(franka_grasp_pos - box_pos_w, p=2, dim=-1) - target_distance)
        distance_reward = torch.exp(-distance_error * 1.5)

        # 잡기축 정의 (그리퍼 초기 위치 → 물체 위치 벡터)
        grasp_axis = box_pos_w - self.init_robot_grasp_pos
        grasp_axis = grasp_axis / (torch.norm(grasp_axis, p=2, dim=-1, keepdim=True) + eps)  # 정규화

        # 그리퍼 전방축과 잡기축 정렬 보상
        gripper_forward = tf_vector(franka_grasp_rot, gripper_forward_axis)
        alignment_score = torch.sum(gripper_forward * grasp_axis, dim=-1)  # 내적 계산
        alignment_reward = (alignment_score + 1) / 2  # [-1,1] → [0,1] 변환
        
        # 그리퍼 위치가 잡기축 위에 있는지 확인
        gripper_proj_dist = torch.norm(torch.cross(franka_grasp_pos - self.init_robot_grasp_pos, grasp_axis, dim=-1), p=2, dim=-1)
        position_alignment_reward = torch.exp(-gripper_proj_dist * alignment_reward_scale)  # 잡기축 벗어나면 패널티

        # 최종 정렬 보상
        total_alignment_reward = 0.7 * alignment_reward + 0.3 * position_alignment_reward

        # 그리퍼가 초기 자세에서 많이 벗어날수록 패널티 적용 (이상한 자세 방지)
        joint_deviation = torch.abs(self._robot.data.joint_pos - self.init_robot_joint_position)
        joint_penalty = torch.sum(joint_deviation, dim=-1)
        joint_penalty = torch.tanh(joint_penalty)

        # 행동 크기가 클수록 패널티 적용 (이상한 행동 방지)
        action_penalty = 0.15 * torch.sum(actions**2, dim=-1)
        
        # 카메라 veiw 중심으로부터 거리 (XY 평면 기준)
        center_offset = torch.norm(box_pos_cam[:, :2], dim=-1)  # XY 거리 (Z는 depth)
        center_wdight = 15.0
        pview_reward = torch.exp(-center_offset * center_wdight)  # scale 조정 가능 (10.0은 예시)
        
        fov_threshold = 1.0
        out_of_fov_mask = (box_pos_cam[:, 2] < 0.05) | (center_offset > fov_threshold) # Z가 너무 작으면 벗어남
        pview_reward[out_of_fov_mask] = -5.0  # 큰 패널티
        
        # 카메라 veiw 기준 물체의 회전 고정
        up_vec = torch.tensor([0, 1, 0], dtype=torch.float32, device=box_rot_cam.device).expand(box_rot_cam.shape[0], 3)        
        box_up_cam = tf_vector(box_rot_cam, up_vec)  # (N, 3)       

        cam_up = torch.tensor([0, -1, 0], dtype=torch.float32, device=box_rot_cam.device).unsqueeze(0)    

        up_alignment = torch.sum(box_up_cam * cam_up, dim=-1)
        up_alignment_reward = (up_alignment + 1) / 2

        # 5. 최종 보상 계산
        rewards = (
            distance_reward_scale * distance_reward  # 거리 유지 보상
            + alignment_reward_scale * total_alignment_reward  # 정렬 보상
            + pview_reward_scale * pview_reward
            # + cam_up_alignment_reward_scale * up_alignment_reward
            - joint_penalty_scale * joint_penalty  # 자세 안정성 패널티
            - action_penalty_scale * action_penalty  # 행동 크기 패널티
        )

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
        
        
        