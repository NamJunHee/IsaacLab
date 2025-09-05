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
from geometry_msgs.msg import Point
import math

from cv_bridge import CvBridge
import threading
import time

class RobotType(Enum):
    FRANKA = "franka"
    UF = "ufactory"
    DOOSAN = "doosan"
robot_type = RobotType.UF

class ObjectMoveType(Enum):
    STATIC = "static"
    CIRCLE = "circle"
    LINEAR = "linear"
    CURRICULAR = "curricular"
object_move = ObjectMoveType.STATIC
# object_move = ObjectMoveType.LINEAR
# object_move = ObjectMoveType.CURRICULAR

training_mode = True

foundationpose_mode = False

camera_enable = False
image_publish = False

robot_action = False
robot_init_pose = False
robot_fix = False

init_reward = True

add_episode_length = 200
# add_episode_length = 400
# add_episode_length = 800
# add_episode_length = -930

vel_ratio = 0.10 # max 2.61s

obj_speed = 0.0005
# obj_speed = 0.001
# obj_speed = 0.0015
# obj_speed = 0.002

# 물체 움직임 범위
rand_pos_range = {
    "x" : (  0.35, 0.80),
    "y" : ( -0.1, 0.1),
    "z" : (  0.00, 0.75),
}

reward_curriculum_levels = [
    # Level 0: pview(시야 확보)에 압도적인 가중치를 부여하는 초기 단계
    {
        "reward_scales": {"distance": 10.0, "vector_align": 6.0, "position_align": 6.0, "pview": 10.0, "joint_penalty": 0.5},
        "success_threshold": 20.0, "failure_threshold": 10.0
    },
    # Level 1: pview에 익숙해졌으므로 다른 보상들의 가중치를 높이는 중간 단계
    {
        "reward_scales": {"distance": 10.0, "vector_align": 9.0, "position_align": 8.0, "pview": 10.0, "joint_penalty": 1.0},
        "success_threshold": 25.0, "failure_threshold": 20.0
    },
    # Level 2: 모든 과업을 균형 있게 잘해야 하는 최종 단계
    {
        "reward_scales": {"distance": 10.0, "vector_align": 8.0, "position_align": 8.0, "pview": 10.0, "joint_penalty": 2.5},
        "success_threshold": 25.0, "failure_threshold": 20.0
    },
]

# vector_align_margin = 0.85
vector_align_margin = math.radians(10.0)
# vector_align_margin = 0.95

position_align_margin = 0.05
# position_align_margin = 0.10
# position_align_margin = 0.05

pview_margin = 0.20 # 학습 초기
# pview_margin = 0.15 # 학습 중기
# # pview_margin = 0.10 # 학습 후기

pose_candidate = {
    "top_close":   {"joint1": math.radians(0.0), 
                      "joint2": math.radians(-50.0), 
                      "joint3": math.radians(-30.0), 
                      "joint4": math.radians(0.0), 
                      "joint5": math.radians(-30.0), 
                      "joint6": math.radians(0.0)},
    
    # "top_close_2":   {"joint1": math.radians(0.0), 
    #                   "joint2": math.radians(-110.0), 
    #                   "joint3": math.radians(5.0), 
    #                   "joint4": math.radians(0.0), 
    #                   "joint5": math.radians(-5.0), 
    #                   "joint6": math.radians(0.0)},
    
    "top_middle":   {"joint1": math.radians(  0.0), 
                      "joint2": math.radians(-30.0), 
                      "joint3": math.radians(-10.0), 
                      "joint4": math.radians(  0.0), 
                      "joint5": math.radians( -45.0), 
                      "joint6": math.radians(  0.0)},
    
    # "top_middle_2":   {"joint1": math.radians(  0.0), 
    #                   "joint2": math.radians(-5.0), 
    #                   "joint3": math.radians(-60.0), 
    #                   "joint4": math.radians(  0.0), 
    #                   "joint5": math.radians( -35.0), 
    #                   "joint6": math.radians(  0.0)},
    
    "top_far":     {"joint1": math.radians(  0.0), 
                      "joint2": math.radians( -20.0),  
                      "joint3": math.radians(-45.0), 
                      "joint4": math.radians(  0.0), 
                      "joint5": math.radians(  -35.0), 
                      "joint6": math.radians(  0.0)},
    
    # "top_far_2":     {"joint1": math.radians(  0.0), 
    #                   "joint2": math.radians(  0.0),  
    #                   "joint3": math.radians(-65.0), 
    #                   "joint4": math.radians(  0.0), 
    #                   "joint5": math.radians(-35.0), 
    #                   "joint6": math.radians(  0.0)},
    
    
    "middle_close":  {"joint1": math.radians(0.0), 
                      "joint2": math.radians(-110.0),
                      "joint3": math.radians( -5.0),  
                      "joint4": math.radians(0.0), 
                      "joint5": math.radians(45.0), 
                      "joint6": math.radians(0.0)},
    
    "middle_middle": {"joint1": math.radians(0.0), 
                      "joint2": math.radians(-40.0), 
                      "joint3": math.radians(0.0), 
                      "joint4": math.radians(0.0), 
                      "joint5": math.radians(-45.0),  
                      "joint6": math.radians(0.0)},
    
    "middle_far":    {"joint1": math.radians(0.0),  
                      "joint2": math.radians(15.0),  
                      "joint3": math.radians(-50.0),
                      "joint4": math.radians(0.0), 
                      "joint5": math.radians(-50.0), 
                      "joint6": math.radians(0.0)},



    "bottom_close":  {"joint1": math.radians(0.0), 
                      "joint2": math.radians(-95.0),   
                      "joint3": math.radians(-5.0),   
                      "joint4": math.radians(0.0), 
                      "joint5": math.radians( 50.0),
                      "joint6": math.radians(0.0)},
    
    # "bottom_close_2":  {"joint1": math.radians(0.0), 
    #                   "joint2": math.radians(-70.0),   
    #                   "joint3": math.radians( 0.0),   
    #                   "joint4": math.radians(0.0), 
    #                   "joint5": math.radians( 35.0),
    #                   "joint6": math.radians(0.0)},
    
    
    "bottom_middle": {"joint1": math.radians(0.0), 
                      "joint2": math.radians(-60.0),  
                      "joint3": math.radians(-0.0), 
                      "joint4": math.radians(0.0), 
                      "joint5": math.radians(10.0),
                      "joint6": math.radians(0.0)},
    
    # "bottom_middle_2": {"joint1": math.radians(0.0), 
    #                   "joint2": math.radians(-30.0),  
    #                   "joint3": math.radians(-0.0), 
    #                   "joint4": math.radians(0.0), 
    #                   "joint5": math.radians(-10.0),
    #                   "joint6": math.radians(0.0)},
    
    "bottom_far":    {"joint1": math.radians(0.0), 
                      "joint2": math.radians(-25.0),  
                      "joint3": math.radians(-15.0),
                      "joint4": math.radians(0.0), 
                      "joint5": math.radians(-5.0), 
                      "joint6": math.radians(0.0)},
    
    # "bottom_far_2":    {"joint1": math.radians(0.0), 
    #                   "joint2": math.radians(15.0),  
    #                   "joint3": math.radians(-45.0),
    #                   "joint4": math.radians(0.0), 
    #                   "joint5": math.radians(-5.0), 
    #                   "joint6": math.radians(0.0)},
}

initial_pose = pose_candidate["middle_middle"]

workspace_zones = {
    "x": {"far": 0.65, "middle": 0.50},
    "z": {"bottom": 0.25, "middle": 0.50}
}

@configclass
class FrankaObjectTrackingEnvCfg(DirectRLEnvCfg):
    ## env
    episode_length_s = 8.3333  # 500 timesteps
    decimation = 2
    
    if robot_type == RobotType.FRANKA:
        action_space = 9
        observation_space = 23
        
    elif robot_type == RobotType.UF:
        # action_space = 12
        # observation_space = 29
        
        action_space = 6
        observation_space = 17
        
    elif robot_type == RobotType.DOOSAN:
        action_space = 8
        observation_space = 21
    
    state_space = 0

    ## simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 240,
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

    ## scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=3.0, replicate_physics=True)

    ## robot
    Franka_robot = ArticulationCfg(
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
                "panda_joint1":  0.000,
                "panda_joint2": -0.831,
                "panda_joint3": -0.000,
                "panda_joint4": -1.796,
                "panda_joint5": -0.000,
                "panda_joint6":  1.733,
                "panda_joint7":  0.707,
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
                velocity_limit=0.22,
                stiffness=80.0,
                # stiffness=200.0,
                # damping=4.0,
                damping=25.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit=12.0,
                # velocity_limit=2.61,
                velocity_limit=0.22,
                stiffness=80.0,
                # stiffness=200.0,
                # damping=4.0,
                damping=25.0,
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
    
    UF_robot = ArticulationCfg(
        # prim_path="/World/envs/env_.*/xarm6_with_gripper",
        prim_path="/World/envs/env_.*/xarm6",
        spawn=sim_utils.UsdFileCfg(
            # usd_path="/home/nmail-njh/NMAIL/01_Project/Robot_Grasping/IsaacLab/ROBOT/xarm6_with_gripper/xarm6_with_gripper.usd",
            usd_path="/home/nmail-njh/NMAIL/01_Project/Robot_Grasping/IsaacLab/ROBOT/xarm6_robot_white/xarm6_robot_white.usd",
            
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, solver_position_iteration_count=24, solver_velocity_iteration_count=1
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            # joint_pos={
            #     # "joint1" : math.radians(  0.0),
            #     # "joint2" : math.radians(-66.0),
            #     # "joint3" : math.radians(  8.0),
            #     # "joint4" : math.radians(  0.0),
            #     # "joint5" : math.radians( 15.0),
            #     # "joint6" : math.radians(  0.0),
            # },
            joint_pos = initial_pose,
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "ufactory_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["joint1", "joint2", "joint3"],
                effort_limit=87.0,
                
                velocity_limit=2.61 * vel_ratio,
                stiffness=2000.0,
                damping=100.0,
                
                # velocity_limit=0.8,
                # stiffness=80.0,
                # damping=18.0,
            ),
            "ufactory_forearm": ImplicitActuatorCfg(
                joint_names_expr=["joint4", "joint5", "joint6"],
                effort_limit=87.0,
                
                velocity_limit=2.61 * vel_ratio,
                stiffness=2000.0,
                damping=100.0,
                
                # velocity_limit=0.8,
                # stiffness=80.0,
                # damping=18.0,
            ),
            # "ufactory_hand": ImplicitActuatorCfg(
            #     joint_names_expr=["left_finger_joint", "right_finger_joint"],
            #     effort_limit=200.0,
            #     velocity_limit=0.2,
            #     stiffness=2e3,
            #     damping=1e2,
            # ),
        },
    )

    Doosan_robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Doosan_M1013",
        # prim_path="/World/envs/env_.*/m1013",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/nmail-njh/NMAIL/01_Project/Robot_Grasping/IsaacLab/ROBOT/Doosan_M1013/M1013_onrobot_with_gripper/M1013_onrobot.usda",
            # usd_path="/home/nmail-njh/NMAIL/01_Project/Robot_Grasping/IsaacLab/ROBOT/Doosan_M1013/m1013_with_gripper/m1013_with_gripper.usd",
            # usd_path="/home/nmail-njh/NMAIL/01_Project/Robot_Grasping/IsaacLab/ROBOT/Doosan_M1013/m1013/m1013.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=12, solver_velocity_iteration_count=1
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "J1_joint":  0.00,
                "J2_joint": -0.60,
                "J3_joint":  1.80,
                "J4_joint":  0.00,
                "J5_joint":  1.25,
                "J6_joint":  0.00,
                "left_joint" : 0.0,
                "right_joint": 0.0
                
                # "joint1":  0.00,
                # "joint2": -0.60,
                # "joint3":  1.80,
                # "joint4":  0.00,
                # "joint5":  1.25,
                # "joint6" : 0.00,
                # "left_joint" : 0.0,
                # "right_joint": 0.0
            },
            pos=(1.0, 0.0, 0.05),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        actuators={
            "doosan_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["J1_joint", "J2_joint", "J3_joint"],
                # joint_names_expr=["joint1", "joint2", "joint3"],
                effort_limit=87.0,
                # velocity_limit=2.175,
                velocity_limit=0.25,
                stiffness=20.0,
                # stiffness=200.0,
                # damping=4.0,
                damping=30.0,
            ),
            "doosan_forearm": ImplicitActuatorCfg(
                joint_names_expr=["J4_joint", "J5_joint", "J6_joint"],
                # joint_names_expr=["joint4", "joint5", "joint6"],
                effort_limit=12.0,
                # velocity_limit=2.61,
                velocity_limit=0.25,
                stiffness=20.0,
                # stiffness=200.0,
                # damping=4.0,
                damping=30.0,
            ),
            "doosan_hand": ImplicitActuatorCfg(
                joint_names_expr=["left_joint", "right_joint"],
                effort_limit=200.0,
                velocity_limit=0.3,
                stiffness=2e3,
                damping=1e2,
            ),
        },
    )

    ## camera
    if camera_enable:
        if robot_type == RobotType.FRANKA:
            camera = CameraCfg(
                prim_path="/World/envs/env_.*/Robot/panda_hand/hand_camera", 
                update_period=0.03,
                height=480,
                width=640,
                data_types=["rgb", "depth"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=25.0, # 값이 클수록 확대
                    focus_distance=60.0,
                    horizontal_aperture=50.0,
                    clipping_range=(0.1, 1.0e5),
                ),
                offset=CameraCfg.OffsetCfg(
                    pos=(0.0, 0.0, 0.05),
                    rot=(0.0, 0.707, 0.707, 0.0),
                    convention="ROS",
                )
            )
            
        elif robot_type == RobotType.UF:
            camera = CameraCfg(
                # prim_path="/World/envs/env_.*/xarm6_with_gripper/link6/hand_camera",
                prim_path="/World/envs/env_.*/xarm6/link6/hand_camera",
                update_period=0.03,
                height=480,
                width=640,
                data_types=["rgb", "depth"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=15.0, # 값이 클수록 확대
                    focus_distance=60.0,
                    horizontal_aperture=50.0,
                    clipping_range=(0.1, 1.0e5),
                ),
                offset=CameraCfg.OffsetCfg(
                    pos=(0.0, 0.0, 0.1),
                    rot=(0.0, 0.707, 0.707, 0.0),
                    convention="ROS",
                )
            )
            
        elif robot_type == RobotType.DOOSAN:
            camera = CameraCfg(
                # prim_path="/World/envs/env_.*/Doosan_M1013/gripper/onrobot_2fg_14/base/hand_camera", 
                prim_path="/World/envs/env_.*/Doosan_M1013/J6/hand_camera", 
                # prim_path="/World/envs/env_.*/m1013/link6/hand_camera", 
                update_period=0.03,
                height=480,
                width=640,
                data_types=["rgb", "depth"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=15.0, # 값이 클수록 확대
                    focus_distance=60.0,
                    horizontal_aperture=50.0,
                    clipping_range=(0.1, 1.0e5),
                ),
                offset=CameraCfg.OffsetCfg(
                    pos=(0.0, 0.0, 1.5),
                    # rot=(-0.5, 0.5, -0.5, -0.5), #ROS
                    # rot=(-0.5, -0.5, -0.5, 0.5), #ros
                    rot=(0.0, -0.707, 0.707, 0.0),
                    convention="ROS",
                )
            )
    
    ## cabinet
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

    ## ground plane
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
    
    ## cube
    cube = RigidObjectCfg(
        prim_path="/World/envs/env_.*/cube",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.1, 0, 0.055), rot=(1, 0, 0, 0)),
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

    ## mustard
    box = RigidObjectCfg(
        prim_path="/World/envs/env_.*/base_link",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0, 0.25), rot=(0.923, 0, 0, -0.382)),
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
    current_time = 0.0

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
        
        self.log_counter = 0
        self.LOG_INTERVAL = 2  # 1번의 리셋 묶음마다 한 번씩 로그 출력

        # 가까운 물체 추적에 유리한 시작 자세
        close_range_pose = {"joint1" : math.radians( 0.0),
                            "joint2" : math.radians(-100.0),
                            "joint3" : math.radians(-0.0),
                            "joint4" : math.radians( 0.0),
                            "joint5" : math.radians( 70.0),
                            "joint6" : math.radians( 0.0), }
        # 멀리 있는 물체 추적에 유리한 시작 자세
        far_range_pose = {"joint1" : math.radians( 0.0),
                          "joint2" : math.radians(-80.0),
                          "joint3" : math.radians(-12.0),
                          "joint4" : math.radians( 0.0),
                          "joint5" : math.radians( 60.0),
                          "joint6" : math.radians( 0.0), }

        
        self.curriculum_levels = [
            # =================================================================
            #  1단계: 정적 물체 마스터 (기본기 학습)
            # =================================================================
            # Level 0: 가장 관대한 마진과 기본 정렬(경로/방향)에 집중된 가중치
            {
                "obj_speed": 0.0, "init_joint_pos": far_range_pose,
                "rand_pos_range": {"x": (0.40, 0.50), "y": (-0.15, 0.15), "z": (0.15, 0.25)},
                "sampling_weights": {"sweet_spot": 1.0, "medium": 0.0, "edge": 0.0},
                "reward_scales": {"distance": 6.0, 
                                  "vector_align": 9.5, 
                                  "position_align": 10.0, 
                                  "pview": 8.0, 
                                  "joint_penalty": 1.0},
                "margins": {"vector_align": 0.85, "position_align": 0.15, "pview": 0.20},
                "success_threshold": 20.0, "failure_threshold": 15.0
            },
            # Level 1: 마진을 약간 좁히고, 거리 유지의 중요도를 조금씩 높임
            {
                "obj_speed": 0.0, "init_joint_pos": far_range_pose,
                "rand_pos_range": {"x": (0.35, 0.55), "y": (-0.20, 0.20), "z": (0.10, 0.35)},
                "sampling_weights": {"sweet_spot": 0.6, "medium": 0.4, "edge": 0.0},
                "reward_scales": {"distance": 7.0, 
                                  "vector_align": 9.0, 
                                  "position_align": 9.5, 
                                  "pview": 8.0, 
                                  "joint_penalty": 1.5},
                "margins": {"vector_align": 0.88, "position_align": 0.12, "pview": 0.18},
                "success_threshold": 18.0, "failure_threshold": 10.0
            },
            # Level 2: 넓은 범위 학습 시작. 먼 자세를 도입하고 거리 유지와 자세 효율성(페널티)의 중요도 상승
            {
                "obj_speed": 0.0, "init_joint_pos": far_range_pose,
                "rand_pos_range": {"x": (0.30, 0.60), "y": (-0.25, 0.25), "z": (0.07, 0.50)},
                "sampling_weights": {"sweet_spot": 0.2, "medium": 0.4, "edge": 0.4},
                "reward_scales": {"distance": 8.0, 
                                  "vector_align": 8.0, 
                                  "position_align": 8.5, 
                                  "pview": 7.5, 
                                  "joint_penalty": 2.0},
                "margins": {"vector_align": 0.90, "position_align": 0.10, "pview": 0.15},
                "success_threshold": 15.0, "failure_threshold": 8.0
            },

            # =================================================================
            #  2단계: 동적 물체 추적 (응용 학습)
            # =================================================================
            # Level 3: 움직임 첫 도입. 마진을 다시 약간 완화하여 적응을 돕고, 정렬 능력에 집중
            {
                "obj_speed": 0.0005, "init_joint_pos": far_range_pose,
                "rand_pos_range": {"x": (0.40, 0.50), "y": (-0.15, 0.15), "z": (0.15, 0.25)},
                "sampling_weights": {"sweet_spot": 1.0, "medium": 0.0, "edge": 0.0},
                "reward_scales": {"distance": 7.0, 
                                  "vector_align": 9.0, 
                                  "position_align": 9.5, 
                                  "pview": 8.0, 
                                  "joint_penalty": 2.0},
                "margins": {"vector_align": 0.90, "position_align": 0.12, "pview": 0.18},
                "success_threshold": 15.0, "failure_threshold": 5.0
            },
            # Level 4: 더 빠른 속도와 넓은 범위. 거리 유지의 중요도를 다시 높임
            {
                "obj_speed": 0.0010, "init_joint_pos": far_range_pose,
                "rand_pos_range": {"x": (0.30, 0.60), "y": (-0.25, 0.25), "z": (0.07, 0.50)},
                "sampling_weights": {"sweet_spot": 0.2, "medium": 0.4, "edge": 0.4},
                "reward_scales": {"distance": 9.0, 
                                  "vector_align": 7.5, 
                                  "position_align": 7.5, 
                                  "pview": 7.0, 
                                  "joint_penalty": 2.5},
                "margins": {"vector_align": 0.92, "position_align": 0.08, "pview": 0.13},
                "success_threshold": 12.0, "failure_threshold": 2.0
            },
            # Level 5: 최종 단계. 가장 엄격한 마진, 거리 유지와 자세 효율성에 최고 가중치 부여
            {
                "obj_speed": 0.0015, "init_joint_pos": far_range_pose,
                "rand_pos_range": {"x": (0.30, 0.60), "y": (-0.25, 0.25), "z": (0.07, 0.50)},
                "sampling_weights": {"sweet_spot": 0.1, "medium": 0.3, "edge": 0.6},
                "reward_scales": {"distance": 10.0, 
                                  "vector_align": 7.0, 
                                  "position_align": 7.0, 
                                  "pview": 7.0, 
                                  "joint_penalty": 3.0},
                "margins": {"vector_align": 0.95, "position_align": 0.07, "pview": 0.12},
                "success_threshold": 10.0, "failure_threshold": 0.0
            },
        ]
        self.max_curriculum_level = len(self.curriculum_levels) - 1

        # 각 환경의 현재 커리큘럼 레벨 추적 (모두 0에서 시작)
        self.current_curriculum_level = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
        # 성능 모니터링을 위한 버퍼
        self.episode_reward_buf = torch.zeros(self.num_envs, device=self.device)
        
        # 승급/강등 조건을 위한 카운터
        self.consecutive_successes = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.consecutive_failures = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.PROMOTION_COUNT = 15  # 20번 연속 성공 시 승급
        self.DEMOTION_COUNT = 10   # 15번 연속 실패 시 강등
        
        # 1. 보상 스케일만 조절하는 새로운 커리큘럼 레벨 정의
        self.max_reward_level = len(reward_curriculum_levels) - 1

        # 2. 보상 커리큘럼을 위한 독립적인 상태 변수들
        self.current_reward_level = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.consecutive_successes_reward = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.consecutive_failures_reward = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.PROMOTION_COUNT_REWARD = 15
        self.DEMOTION_COUNT_REWARD = 10
        
        if robot_type == RobotType.FRANKA:
            self.joint_names = [
            "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
            "panda_joint5", "panda_joint6", "panda_joint7",
            "panda_finger_joint1", "panda_finger_joint2"
            ]
            self.joint_init_values = [0.000, -0.831, 0.000, -1.796, 0.000, 2.033, 0.707, 0.035, 0.035]
        elif robot_type == RobotType.UF:
            self.joint_names = [
            "joint1", "joint2", "joint3", "joint4","joint5", "joint6", ]
            
            # self.joint_init_values = [0.000, -1.220, -0.50, -0.000, 1.300, 0.000]
            # self.joint_init_values = [0.000, -1.220, -0.50, -0.000, 0.300, 0.000]
            # self.joint_init_values = [0.000, -1.220, -0.50, -0.000, 0.800, 0.000]
            
            # self.joint_init_values = [math.radians(0.0), 
            #                           math.radians(-100.0),
            #                           math.radians(0.0), 
            #                           math.radians(0.0), 
            #                           math.radians(70.0), 
            #                           math.radians(0.0)]
            
            # self.joint_init_values = [math.radians(0.0), 
            #                           math.radians(-66.0),
            #                           math.radians(  8.0), 
            #                           math.radians(  0.0), 
            #                           math.radians( 15.0), 
            #                           math.radians(  0.0)]
            
            self.joint_init_values = [initial_pose[name] for name in self.joint_names]
            
        elif robot_type == RobotType.DOOSAN:
            self.joint_names = [
            "J1_joint", "J2_joint", "J3_joint", "J4_joint","J5_joint", "J6_joint" ]
            self.joint_init_values = [0.000, -0.600, 1.800, 0.000, 1.250, 0.000] 

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
        
        stage = get_current_stage()
        
        if robot_type == RobotType.FRANKA:
            self.robot_dof_speed_scales[self._robot.find_joints("panda_finger_joint1")[0]] = 0.1
            self.robot_dof_speed_scales[self._robot.find_joints("panda_finger_joint2")[0]] = 0.1
            
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
            self.hand_link_idx = self._robot.find_bodies("panda_link7")[0][0]
            self.left_finger_link_idx = self._robot.find_bodies("panda_leftfinger")[0][0]
            self.right_finger_link_idx = self._robot.find_bodies("panda_rightfinger")[0][0]
            
        elif robot_type == RobotType.UF:
            # self.robot_dof_speed_scales[self._robot.find_joints("left_finger_joint")[0]] = 0.1
            # self.robot_dof_speed_scales[self._robot.find_joints("right_finger_joint")[0]] = 0.1
            
            hand_pose = get_env_local_pose(
                self.scene.env_origins[0],
                # UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/xarm6_with_gripper/link6")),
                UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/xarm6/link6")),
                self.device,
            )
            lfinger_pose = get_env_local_pose(
                self.scene.env_origins[0],
                # UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/xarm6_with_gripper/left_finger")),
                UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/xarm6/link6")),
                self.device,
            )
            rfinger_pose = get_env_local_pose(
                self.scene.env_origins[0],
                # UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/xarm6_with_gripper/right_finger")),
                UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/xarm6/link6")),
                self.device,
            )
            self.hand_link_idx = self._robot.find_bodies("link6")[0][0]
            # self.left_finger_link_idx = self._robot.find_bodies("left_finger")[0][0]
            # self.right_finger_link_idx = self._robot.find_bodies("right_finger")[0][0]
             
        elif robot_type == RobotType.DOOSAN:
            
            self.robot_dof_speed_scales[self._robot.find_joints("left_joint")[0]] = 0.1
            self.robot_dof_speed_scales[self._robot.find_joints("right_joint")[0]] = 0.1
            
            hand_pose = get_env_local_pose(
                self.scene.env_origins[0],
                # UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Doosan_M1013/gripper/onrobot_2fg_14/base")),
                UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Doosan_M1013/J6")),
                self.device,
            )
            lfinger_pose = get_env_local_pose(
                self.scene.env_origins[0],
                UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Doosan_M1013/gripper/onrobot_2fg_14/Left")),
                self.device,
            )
            rfinger_pose = get_env_local_pose(
                self.scene.env_origins[0],
                UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Doosan_M1013/gripper/onrobot_2fg_14/Right")),
                self.device,
            )
            # self.hand_link_idx = self._robot.find_bodies("base")[0][0]
            self.hand_link_idx = self._robot.find_bodies("J6")[0][0]
            self.left_finger_link_idx = self._robot.find_bodies("Left")[0][0]
            self.right_finger_link_idx = self._robot.find_bodies("Right")[0][0]
        
        self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

        finger_pose = torch.zeros(7, device=self.device)
        finger_pose[0:3] = (lfinger_pose[0:3] + rfinger_pose[0:3]) / 2.0
        finger_pose[3:7] = lfinger_pose[3:7]
        hand_pose_inv_rot, hand_pose_inv_pos = tf_inverse(hand_pose[3:7], hand_pose[0:3])

        robot_local_grasp_pose_rot, robot_local_pose_pos = tf_combine(
            hand_pose_inv_rot, hand_pose_inv_pos, finger_pose[3:7], finger_pose[0:3]
        )
        robot_local_pose_pos += torch.tensor([0, 0.00, 0], device=self.device)
        self.robot_local_grasp_pos = robot_local_pose_pos.repeat((self.num_envs, 1))
        self.robot_local_grasp_rot = robot_local_grasp_pose_rot.repeat((self.num_envs, 1))
        
        box_local_pose = torch.tensor([0.0, 0.0,0.0, 1.0, 0.0, 0.0, 0.0], device=self.device)
        self.box_local_pos = box_local_pose[0:3].repeat((self.num_envs, 1))
        self.box_local_rot = box_local_pose[3:7].repeat((self.num_envs, 1))

        if robot_type == RobotType.FRANKA or robot_type == RobotType.UF:
            self.gripper_forward_axis = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).repeat(
                (self.num_envs, 1)
            )
        elif robot_type == RobotType.DOOSAN:
            self.gripper_forward_axis = torch.tensor([0, 0, -1], device=self.device, dtype=torch.float32).repeat(
                (self.num_envs, 1)
            )
            
        self.gripper_up_axis = torch.tensor([1, 0, 0], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        
        # self.cube_z_axis = torch.tensor([0,0,1], device=self.device, dtype=torch.float32).repeat(
        #     (self.num_envs,1)
        # )
        self.box_z_axis = torch.tensor([0,0,1], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs,1)
        )
        
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
        
        self.box_pos_cam = torch.zeros((self.num_envs, 4), device=self.device)
        
        self.fixed_z = 0.055
        
        self.current_box_pos = None
        self.current_box_rot = None
        
        self.target_box_pos = torch.stack([
                torch.rand(self.num_envs, device=self.device) * (rand_pos_range["x"][1] - rand_pos_range["x"][0]) + rand_pos_range["x"][0],
                torch.rand(self.num_envs, device=self.device) * (rand_pos_range["y"][1] - rand_pos_range["y"][0]) + rand_pos_range["y"][0],
                torch.rand(self.num_envs, device=self.device) * (rand_pos_range["z"][1] - rand_pos_range["z"][0]) + rand_pos_range["z"][0],
            ], dim = 1)
        
        self.target_box_pos = self.target_box_pos + self.scene.env_origins
        # self.rand_pos_step = 0
        self.new_box_pos_rand = torch.zeros((self.num_envs, 3), device=self.device)
        self.current_box_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.rand_pos_step = torch.zeros((self.num_envs, 3), device=self.device)
        
        rclpy.init()
        self.last_publish_time = 0.0
        self.position_error = 0.0
        self.obj_origin_distance = 0.0
        self.out_of_fov_cnt = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        
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
                Point,
                '/object_position',
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
            
            camera_info_msg.header.frame_id = 'tf_camera'
        
            camera_info_msg.height = 480 
            camera_info_msg.width = 640 
            
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
            rgb_msg.header.frame_id = 'tf_camera'
            self.rgb_publisher.publish(rgb_msg)

            # Publish Depth Image
            depth_msg = self.bridge.cv2_to_imgmsg(depth_image, encoding='32FC1')
            depth_msg.header.stamp = current_stamp
            depth_msg.header.frame_id = 'tf_camera'
            self.depth_publisher.publish(depth_msg)
            depth_msg.step = depth_image.shape[1] * 4
    
    def subscribe_object_pos(self):
        msg = self.latest_detection_msg
        
        if msg is None:
            return None

        return torch.tensor([msg.x, msg.y, msg.z], device=self.device)
        
    def foundationpose_callback(self,msg):
        self.latest_detection_msg = msg
    
    def quat_mul(self, q, r):
        x1, y1, z1, w1 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        x2, y2, z2, w2 = r[:, 0], r[:, 1], r[:, 2], r[:, 3]

        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2

        quat = torch.stack((x, y, z, w), dim=-1)
        # return kornia.geometry.quaternion.normalize_quaternion(quat)
        return kornia.geometry.conversions.normalize_quaternion(quat)
    
    def quat_conjugate(self, q):
        q_conj = torch.cat([-q[:, :3], q[:, 3:4]], dim=-1)
        return q_conj
    
    def compute_camera_world_pose(self, hand_pos, hand_rot):
        if robot_type == RobotType.FRANKA:
            cam_offset_pos = torch.tensor([0.0, 0.0, 0.05], device=hand_pos.device).repeat(self.num_envs, 1)
            q_cam_in_hand = torch.tensor([0.0, 0.707, 0.707, 0.0], device=hand_pos.device).repeat(self.num_envs, 1)
        elif robot_type == RobotType.UF:    
            cam_offset_pos = torch.tensor([0.0, 0.0, 0.1], device=hand_pos.device).repeat(self.num_envs, 1)
            q_cam_in_hand = torch.tensor([0.0, 0.707, 0.707, 0.0], device=hand_pos.device).repeat(self.num_envs, 1)
        elif robot_type == RobotType.DOOSAN:
            cam_offset_pos = torch.tensor([0.0, 0.0, 0.0], device=hand_pos.device).repeat(self.num_envs, 1)
            # q_cam_in_hand = torch.tensor([-0.5, 0.5, -0.5, -0.5], device=hand_pos.device).repeat(self.num_envs, 1)
            q_cam_in_hand = torch.tensor([0.0, -0.707, 0.707, 0.0], device=hand_pos.device).repeat(self.num_envs, 1)

        hand_rot_matrix = kornia.geometry.conversions.quaternion_to_rotation_matrix(hand_rot)
        cam_offset_pos_world = torch.bmm(hand_rot_matrix, cam_offset_pos.unsqueeze(-1)).squeeze(-1)

        camera_pos_w = hand_pos + cam_offset_pos_world
        camera_pos_w = camera_pos_w - self.scene.env_origins
        
        # camera_rot_w = self.quat_mul(hand_rot, q_cam_in_hand)
        camera_rot_w = self.robot_grasp_rot
        
        # if robot_type == RobotType.DOOSAN:
        #     camera_rot_w = self.quat_mul(self.robot_grasp_rot, q_cam_in_hand)

        return camera_pos_w, camera_rot_w

    def world_to_camera_pose(self, camera_pos_w, camera_rot_w, obj_pos_w, obj_rot_w):
        rel_pos = obj_pos_w - camera_pos_w

        cam_rot_matrix = kornia.geometry.conversions.quaternion_to_rotation_matrix(camera_rot_w)
        
        obj_pos_cam = torch.bmm(cam_rot_matrix.transpose(1, 2), rel_pos.unsqueeze(-1)).squeeze(-1)

        cam_rot_inv = self.quat_conjugate(camera_rot_w)
        obj_rot_cam = self.quat_mul(cam_rot_inv, obj_rot_w)

        return obj_pos_cam, obj_rot_cam
    
    def camera_to_world_pose(self, camera_pos_w, camera_rot_w, obj_pos_cam, obj_rot_cam):
        cam_rot_matrix = kornia.geometry.conversions.quaternion_to_rotation_matrix(camera_rot_w)
        
        obj_pos_world = torch.bmm(cam_rot_matrix, obj_pos_cam.unsqueeze(-1)).squeeze(-1) + camera_pos_w
        obj_rot_world = self.quat_mul(camera_rot_w, obj_rot_cam)
        
        return obj_pos_world, obj_rot_world
        
    def rand_obj_coordinate(self, env_ids: torch.Tensor):
        num_resets = len(env_ids)
        if num_resets == 0:
            return

        # 작업 공간 영역 정의 (단위: 미터)
        max_reach = 0.700  # 최대 도달 반경 (이미지의 762mm)
        min_reach = 0.350  # 데드 스페이스 반경 (조정 가능)

        u = torch.rand(num_resets, device=self.device)
        r = (u * (max_reach**3 - min_reach**3) + min_reach**3) ** (1.0 / 3.0)

        theta = torch.rand(num_resets, device=self.device) * torch.pi - (torch.pi / 2)
        v = torch.rand(num_resets, device=self.device)
        phi = torch.acos(v)

        x = r * torch.sin(phi) * torch.cos(theta)
        y = r * torch.sin(phi) * torch.sin(theta)
        z = r * torch.cos(phi) 
        
        # x_clamped = torch.clamp(x, min=x_limits[0], max=x_limits[1])
        # y_clamped = torch.clamp(y, min=y_limits[0], max=y_limits[1])
        # z_clamped = torch.clamp(z, min=z_limits[0], max=z_limits[1])

        local_pos = torch.stack([x, y, z], dim=-1)
        reset_pos = self.scene.env_origins[env_ids] + local_pos
            
        identity_rot = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).repeat(num_resets, 1)

        # print("x,y,z:", reset_pos[0])
        reset_pose = torch.cat([reset_pos, identity_rot], dim=-1)
        zero_velocity = torch.zeros((num_resets, 6), device=self.device)

        self._box.write_root_pose_to_sim(reset_pose, env_ids=env_ids)
        self._box.write_root_velocity_to_sim(zero_velocity, env_ids=env_ids)
        
    def _generate_positions_for_levels(self, levels_to_sample: torch.Tensor, num_to_sample: int) -> torch.Tensor:
        """주어진 레벨에 따라 가중치 샘플링으로 새로운 위치를 생성하는 헬퍼 함수"""
        final_x_pos = torch.zeros(num_to_sample, device=self.device)
        final_y_pos = torch.zeros(num_to_sample, device=self.device)
        final_z_pos = torch.zeros(num_to_sample, device=self.device)

        # 각 레벨별로 순회하며 해당 레벨의 환경들에 대해 위치 샘플링
        for level_idx in torch.unique(levels_to_sample):
            level_mask = (levels_to_sample == level_idx.item())
            num_in_level = torch.sum(level_mask)
            if num_in_level == 0:
                continue

            level_cfg = self.curriculum_levels[level_idx.item()]
            weights = level_cfg["sampling_weights"]
            r_range = level_cfg["rand_pos_range"]

            # X축 위치 샘플링
            sweet_spot_range = (0.40, 0.50)
            medium_range = [(0.35, 0.40), (0.50, 0.55)]
            edge_range = [(0.30, 0.35), (0.55, 0.60)]
            
            probs = torch.rand(num_in_level, device=self.device)
            w_sweet = weights["sweet_spot"]
            w_medium = weights["medium"]

            # 각 구역에 해당하는 환경들의 인덱스를 가져옴
            level_indices = torch.where(level_mask)[0]

            # Sweet Spot 샘플링
            sweet_mask = probs < w_sweet
            num_sweet = torch.sum(sweet_mask)
            if num_sweet > 0:
                final_x_pos[level_indices[sweet_mask]] = torch.rand(num_sweet, device=self.device) * (sweet_spot_range[1] - sweet_spot_range[0]) + sweet_spot_range[0]

            # Medium 샘플링
            medium_mask = (probs >= w_sweet) & (probs < w_sweet + w_medium)
            num_medium = torch.sum(medium_mask)
            if num_medium > 0:
                med_sub_probs = torch.rand(num_medium, device=self.device)
                med_pos_1 = torch.rand(num_medium, device=self.device) * (medium_range[0][1] - medium_range[0][0]) + medium_range[0][0]
                med_pos_2 = torch.rand(num_medium, device=self.device) * (medium_range[1][1] - medium_range[1][0]) + medium_range[1][0]
                final_x_pos[level_indices[medium_mask]] = torch.where(med_sub_probs < 0.5, med_pos_1, med_pos_2)
            
            # Edge 샘플링
            edge_mask = probs >= w_sweet + w_medium
            num_edge = torch.sum(edge_mask)
            if num_edge > 0:
                edge_sub_probs = torch.rand(num_edge, device=self.device)
                edge_pos_1 = torch.rand(num_edge, device=self.device) * (edge_range[0][1] - edge_range[0][0]) + edge_range[0][0]
                edge_pos_2 = torch.rand(num_edge, device=self.device) * (edge_range[1][1] - edge_range[1][0]) + edge_range[1][0]
                final_x_pos[level_indices[edge_mask]] = torch.where(edge_sub_probs < 0.5, edge_pos_1, edge_pos_2)
            
            # Y, Z 위치 샘플링 (해당 레벨의 전체 범위 내에서 균등 샘플링)
            final_y_pos[level_indices] = torch.rand(num_in_level, device=self.device) * (r_range["y"][1] - r_range["y"][0]) + r_range["y"][0]
            final_z_pos[level_indices] = torch.rand(num_in_level, device=self.device) * (r_range["z"][1] - r_range["z"][0]) + r_range["z"][0]

        return torch.stack([final_x_pos, final_y_pos, final_z_pos], dim=1)
    
    def _setup_scene(self):
        
        if robot_type == RobotType.FRANKA:
            self._robot = Articulation(self.cfg.Franka_robot)
        elif robot_type == RobotType.UF:
            self._robot = Articulation(self.cfg.UF_robot)
        elif robot_type == RobotType.DOOSAN:
            self._robot = Articulation(self.cfg.Doosan_robot)
    
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
        
        # 카메라 ros2 publish----------------------------------------------------------------------------------------------
        if image_publish:   
            self.last_publish_time += self.dt
            if self.last_publish_time >= (1.0 / 15.0):  # 30fps 기준
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
                torch.rand(self.num_envs, device=self.device) * (rand_pos_range["x"][1] - rand_pos_range["x"][0]) + rand_pos_range["x"][0],
                torch.rand(self.num_envs, device=self.device) * (rand_pos_range["y"][1] - rand_pos_range["y"][0]) + rand_pos_range["y"][0],
                torch.rand(self.num_envs, device=self.device) * (rand_pos_range["z"][1] - rand_pos_range["z"][0]) + rand_pos_range["z"][0],
                ], dim = 1)

                self.target_box_pos = self.target_box_pos + self.scene.env_origins

                self.current_box_pos = self._box.data.body_link_pos_w[:, 0, :].clone()
                self.current_box_rot = self._box.data.body_link_quat_w[:, 0, :].clone()

                self.new_box_pos_rand = self.current_box_pos

                direction = self.target_box_pos - self.current_box_pos
                direction_norm = torch.norm(direction, p=2, dim=-1, keepdim=True) + 1e-6
                self.rand_pos_step = (direction / direction_norm * obj_speed)

            self.new_box_pos_rand = self.new_box_pos_rand + self.rand_pos_step
            new_box_rot_rand = self.current_box_rot 

            if self.new_box_pos_rand is not None and new_box_rot_rand is not None:
                new_box_pose_rand = torch.cat([self.new_box_pos_rand, new_box_rot_rand], dim=-1)
            else:
                raise ValueError("self.new_box_pos_rand or new_box_rot_rand is None")
            
            self._box.write_root_pose_to_sim(new_box_pose_rand)
        
        if object_move == ObjectMoveType.CURRICULAR:
                        
            env_speeds = torch.tensor([self.curriculum_levels[level]["obj_speed"] for level in self.current_curriculum_level], device=self.device).unsqueeze(1)
            moving_envs_mask = (env_speeds > 0).squeeze()

            if torch.any(moving_envs_mask):
                
                distance_to_target = torch.norm(self.target_box_pos - self.new_box_pos_rand, p=2, dim=-1)
                reset_target_mask = moving_envs_mask & (distance_to_target < 0.02)
                
                if torch.any(reset_target_mask):
                    num_reset_targets = torch.sum(reset_target_mask)
                    reset_levels = self.current_curriculum_level[reset_target_mask]
                    
                    new_target_pos = self._generate_positions_for_levels(reset_levels, num_reset_targets)
                    self.target_box_pos[reset_target_mask] = new_target_pos + self.scene.env_origins[reset_target_mask]

                    self.current_box_pos[reset_target_mask] = self._box.data.body_link_pos_w[reset_target_mask, 0, :].clone()
                
                direction = self.target_box_pos - self.new_box_pos_rand
                direction_norm = torch.norm(direction, p=2, dim=-1, keepdim=True) + 1e-6
                self.rand_pos_step = (direction / direction_norm) * env_speeds # env별 속도 적용

                self.new_box_pos_rand[moving_envs_mask] += self.rand_pos_step[moving_envs_mask]
                
                new_box_pose_rand = torch.cat([self.new_box_pos_rand, self.current_box_rot], dim=-1)
                self._box.write_root_pose_to_sim(new_box_pose_rand)
            
        
    def _apply_action(self):
        
        global robot_action
        global robot_init_pose
        
        target_pos = self.robot_dof_targets.clone()
        # print(f"target_pos: {target_pos}")
        
        if robot_type == RobotType.FRANKA:
            joint3_index = self._robot.find_joints(["panda_joint3"])[0]
            joint5_index = self._robot.find_joints(["panda_joint5"])[0]
            joint7_index = self._robot.find_joints(["panda_joint7"])[0]
            target_pos[:, joint3_index] = 0.0
            target_pos[:, joint5_index] = 0.0
            target_pos[:, joint7_index] = 0.0
        elif robot_type == RobotType.UF:
            joint4_index = self._robot.find_joints(["joint4"])[0]
            joint6_index = self._robot.find_joints(["joint6"])[0]
            target_pos[:, joint4_index] = 0.0
            target_pos[:, joint6_index] = 0.0
            target_pos[:, 7:] = 0.0
        elif robot_type == RobotType.DOOSAN:
            joint4_index = self._robot.find_joints(["J4_joint"])[0]
            joint6_index = self._robot.find_joints(["J6_joint"])[0]
            # joint4_index = self._robot.find_joints(["joint4"])[0]
            # joint6_index = self._robot.find_joints(["joint6"])[0]
            target_pos[:, joint4_index] = 0.0
            target_pos[:, joint6_index] = 0.0
        
        if training_mode == False and robot_fix == False:
            if robot_action and robot_init_pose:
                self._robot.set_joint_position_target(target_pos)
                # self._robot.set_joint_position_target(self.robot_dof_targets)
            
            elif robot_action == False and robot_init_pose == False:
                init_pos = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

                for name, val in zip(self.joint_names, self.joint_init_values):
                    index = self._robot.find_joints(name)[0]
                    init_pos[:, index] = val

                self._robot.set_joint_position_target(init_pos)

                joint_err = torch.abs(self._robot.data.joint_pos - init_pos)
                max_err = torch.max(joint_err).item()
                
                if foundationpose_mode:
                    pos = self.subscribe_object_pos()
                    if (max_err < 0.3) and (pos is not None):
                        self.init_cnt += 1
                        print(f"init_cnt : {self.init_cnt}")
                        
                        if self.init_cnt > 300: 
                            robot_action = True
                            robot_init_pose = True
                            
                elif foundationpose_mode == False and max_err < 0.3:
                    self.init_cnt += 1
                    print(f"init_cnt : {self.init_cnt}")
                    if self.init_cnt > 300:
                        robot_init_pose = True
                        robot_action = True
                               
        elif training_mode == True and robot_fix == False:
            if robot_init_pose == False:
                init_pos = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)
                for name, val in zip(self.joint_names, self.joint_init_values):
                    index = self._robot.find_joints(name)[0]
                    init_pos[:, index] = val

                self._robot.set_joint_position_target(init_pos)

                joint_err = torch.abs(self._robot.data.joint_pos - init_pos)
                max_err = torch.max(joint_err).item()
                
                # print(f"max_err : {max_err}")
                if max_err < 0.3:
                    robot_init_pose = True
                    robot_action = True
                
            elif robot_init_pose:
                self._robot.set_joint_position_target(target_pos)
        
    # post-physics step calls

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
                
        if training_mode or object_move == ObjectMoveType.CIRCLE :
            terminated = 0
            truncated = self.episode_length_buf >= self.max_episode_length + add_episode_length
        else:
            terminated = 0
            truncated = self.episode_length_buf >= self.max_episode_length + add_episode_length #- 400 # 물체 램덤 생성 환경 초기화 주기
        
        #환경 고정
        # terminated = 0
        # truncated = 0
        
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        # Refresh the intermediate values after the physics steps
        self._compute_intermediate_values()
        
        # robot_left_finger_pos = self._robot.data.body_link_pos_w[:, self.left_finger_link_idx]
        # robot_right_finger_pos = self._robot.data.body_link_pos_w[:, self.right_finger_link_idx]

        # camera_pos_w = self.compute_camera_world_pose(self.robot_grasp_pos, self.robot_grasp_rot)
        # camera_rot_w = self.robot_grasp_rot
        
        camera_pos_w, camera_rot_w = self.compute_camera_world_pose(self.robot_grasp_pos, self.robot_grasp_rot)

        self.box_pos_cam, box_rot_cam = self.world_to_camera_pose(
            camera_pos_w, camera_rot_w,
            self.box_grasp_pos - self.scene.env_origins, self.box_grasp_rot,
        )
        # print("box_grasp_pos : ", self.box_grasp_pos)
        # print("box_pos_cam : ", self.box_pos_cam) 
        
        reward = self._compute_rewards(
            self.actions,
            self.robot_grasp_pos,
            self.box_grasp_pos,
            self.robot_grasp_rot,
            self.box_grasp_rot,
            self.box_pos_cam,
            box_rot_cam,
            self.gripper_forward_axis,
            self.gripper_up_axis,
        )
        self.episode_reward_buf += reward
    
        return reward

    def _reset_idx(self, env_ids: torch.Tensor | None):

        if object_move == ObjectMoveType.CURRICULAR:
            avg_reward = self.episode_reward_buf[env_ids] / self.episode_length_buf[env_ids]
            current_levels = self.current_curriculum_level[env_ids]
            success_thresholds = torch.tensor([self.curriculum_levels[level.item()]["success_threshold"] for level in current_levels], device=self.device)
            failure_thresholds = torch.tensor([self.curriculum_levels[level.item()]["failure_threshold"] for level in current_levels], device=self.device)

            success_mask = avg_reward >= success_thresholds
            failure_mask = avg_reward < failure_thresholds

            self.consecutive_successes[env_ids] += success_mask.long()
            self.consecutive_successes[env_ids] *= (1 - failure_mask.long())
            self.consecutive_failures[env_ids] += failure_mask.long()
            self.consecutive_failures[env_ids] *= (1 - success_mask.long())

            promotion_candidate_mask = self.consecutive_successes[env_ids] >= self.PROMOTION_COUNT
            if torch.any(promotion_candidate_mask):
                promotion_env_ids = env_ids[promotion_candidate_mask]
                self.current_curriculum_level[promotion_env_ids] = (self.current_curriculum_level[promotion_env_ids] + 1).clamp(max=self.max_curriculum_level)
                self.consecutive_successes[promotion_env_ids] = 0

            demotion_candidate_mask = self.consecutive_failures[env_ids] >= self.DEMOTION_COUNT
            if torch.any(demotion_candidate_mask):
                demotion_env_ids = env_ids[demotion_candidate_mask]
                self.current_curriculum_level[demotion_env_ids] = (self.current_curriculum_level[demotion_env_ids] - 1).clamp(min=0)
                self.consecutive_failures[demotion_env_ids] = 0
        
        avg_reward = self.episode_reward_buf[env_ids] / self.episode_length_buf[env_ids]
        current_reward_levels = self.current_reward_level[env_ids]

        # 2. 새로운 보상 커리큘럼의 임계값 가져오기
        success_thresholds_reward = torch.tensor([reward_curriculum_levels[l.item()]["success_threshold"] for l in current_reward_levels], device=self.device)
        failure_thresholds_reward = torch.tensor([reward_curriculum_levels[l.item()]["failure_threshold"] for l in current_reward_levels], device=self.device)     

        success_mask_reward = avg_reward >= success_thresholds_reward
        failure_mask_reward = avg_reward < failure_thresholds_reward        

        # 3. 보상 커리큘럼의 연속 성공/실패 카운터 업데이트
        self.consecutive_successes_reward[env_ids] += success_mask_reward.long()
        self.consecutive_successes_reward[env_ids] *= (1 - failure_mask_reward.long())
        self.consecutive_failures_reward[env_ids] += failure_mask_reward.long()
        self.consecutive_failures_reward[env_ids] *= (1 - success_mask_reward.long())       

        # 4. 보상 커리큘럼 레벨 승급/강등 처리
        promotion_candidate_mask_reward = self.consecutive_successes_reward[env_ids] >= self.PROMOTION_COUNT_REWARD
        if torch.any(promotion_candidate_mask_reward):
            promotion_env_ids = env_ids[promotion_candidate_mask_reward]
            self.current_reward_level[promotion_env_ids] = (self.current_reward_level[promotion_env_ids] + 1).clamp(max=self.max_reward_level)
            self.consecutive_successes_reward[promotion_env_ids] = 0        

        demotion_candidate_mask_reward = self.consecutive_failures_reward[env_ids] >= self.DEMOTION_COUNT_REWARD
        if torch.any(demotion_candidate_mask_reward):
            demotion_env_ids = env_ids[demotion_candidate_mask_reward]
            self.current_reward_level[demotion_env_ids] = (self.current_reward_level[demotion_env_ids] - 1).clamp(min=0)
            self.consecutive_failures_reward[demotion_env_ids] = 0
        
        # 에피소드 보상 버퍼 초기화
        self.episode_reward_buf[env_ids] = 0.0
                
        # robot state ---------------------------------------------------------------------------------
        if training_mode:
            joint_pos = self._robot.data.default_joint_pos[env_ids]
            
            # joint_pos = self._robot.data.default_joint_pos[env_ids] + sample_uniform(
            #     -0.125,
            #     0.125,
            #     (len(env_ids), self._robot.num_joints),
            #     self.device,
            # )
            
            # joint_pos = torch.zeros((len(env_ids), self._robot.num_joints), device=self.device)
            
            for i, name in enumerate(self.joint_names):
                joint_idx = self._robot.find_joints([name])[0]
                joint_pos[:, joint_idx] = self.joint_init_values[i]
            
            joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
            joint_vel = torch.zeros_like(joint_pos)
            self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
            self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        else:
            # 최초 한 번만 실행
            if not hasattr(self, "_initialized"):
                self._initialized = False

            if not self._initialized:
                joint_pos = self._robot.data.default_joint_pos[env_ids] 
                
                # joint_pos = self._robot.data.default_joint_pos[env_ids] + sample_uniform(
                # -0.125,
                # 0.125,
                # (len(env_ids), self._robot.num_joints),
                # self.device,
                # )
                
                joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
                joint_vel = torch.zeros_like(joint_pos)
                self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
                self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
                self._initialized = True
        
        # 물체 원 운동 (원 운동 시 환경 초기화 코드)------------------------------------------------------------------------------------------------------------  
        if object_move == ObjectMoveType.CIRCLE:
            reset_pos = self.box_center
            reset_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device, dtype=torch.float32).unsqueeze(0).repeat(self.num_envs, 1)
            reset_box_pose = torch.cat([reset_pos, reset_rot], dim = -1)
            self._box.write_root_pose_to_sim(reset_box_pose)
        
        # 물체 위치 랜덤 생성 (Static) (실제 물체 생성 코드) -----------------------------------------------------------------------------------------------------------
        if object_move == ObjectMoveType.STATIC:
            self.rand_pos = torch.stack([
                torch.rand(self.num_envs, device=self.device) * (rand_pos_range["x"][1] - rand_pos_range["x"][0]) + rand_pos_range["x"][0],
                torch.rand(self.num_envs, device=self.device) * (rand_pos_range["y"][1] - rand_pos_range["y"][0]) + rand_pos_range["y"][0],
                torch.rand(self.num_envs, device=self.device) * (rand_pos_range["z"][1] - rand_pos_range["z"][0]) + rand_pos_range["z"][0],
            ], dim = 1)
            
            rand_reset_pos = self.rand_pos + self.scene.env_origins
            
            random_angles = torch.rand(self.num_envs, device=self.device) * 2 * torch.pi  # 0 ~ 2π 랜덤 값
            rand_reset_rot = torch.stack([
                torch.cos(random_angles / 2),  # w
                torch.zeros(self.num_envs, device=self.device),  # x
                torch.zeros(self.num_envs, device=self.device),  # y
                torch.sin(random_angles / 2)  # z (z축 회전)
            ], dim=1)
            
            rand_reset_box_pose = torch.cat([rand_reset_pos, rand_reset_rot], dim=-1)
            zero_root_velocity = torch.zeros((self.num_envs, 6), device=self.device)
            # self.rand_obj_coordinate(env_ids)
            self._box.write_root_pose_to_sim(rand_reset_box_pose)
            self._box.write_root_velocity_to_sim(zero_root_velocity)
            
            
            ## 최소값 기준 구역 판단 및 로봇 초기 자세 설정
            if training_mode == True:
                joint_pos = self._robot.data.default_joint_pos[env_ids].clone()

                for i, env_id in enumerate(env_ids):
                    # 물체의 로컬 좌표 (x, z) 가져오기
                    object_pos_local = rand_reset_pos[i] - self.scene.env_origins[env_id]
                    obj_x, obj_z = object_pos_local[0], object_pos_local[2]

                    if obj_x >= workspace_zones["x"]["far"]: # [구간 A] x >= 0.65
                        x_zone = "far"
                    elif obj_x >= workspace_zones["x"]["middle"]: # [구간 B] 0.5 <= x < 0.65
                        x_zone = "middle"
                    else:
                        x_zone = "close"

                    if obj_z <= workspace_zones["z"]["bottom"]:
                        z_zone = "bottom"
                    elif obj_z <= workspace_zones["z"]["middle"]:
                        z_zone = "middle"
                    else: # obj_z >= workspace_zones["z"]["bottom"]:
                        z_zone = "top"

                    zone_key = f"{z_zone}_{x_zone}"
                    target_pose_dict = pose_candidate[zone_key]
                    
                    # print("zone_key : ", zone_key)

                    for joint_name, pos in target_pose_dict.items():
                        joint_idx = self._robot.find_joints(joint_name)[0]
                        joint_pos[i, joint_idx] = pos

                # 계산된 초기 자세들을 시뮬레이터에 적용
                joint_vel = torch.zeros_like(joint_pos)
                self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
                self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        
        # 물체 위치 선형 랜덤 이동 (Linear) ---------------------------------------------------------------
        if object_move == ObjectMoveType.LINEAR:
            self.new_box_pos_rand = self._box.data.body_link_pos_w[:, 0, :].clone()
            self.current_box_rot = self._box.data.body_link_quat_w[:, 0, :].clone()

            # self.new_box_pos_rand = self.current_box_pos
            # self.target_box_pos = self.rand_pos

            direction = self.target_box_pos - self.new_box_pos_rand
            direction_norm = torch.norm(direction, p=2, dim=-1, keepdim=True) + 1e-6
            self.rand_pos_step = (direction / direction_norm * obj_speed)
        
        # 물체 위치 커리큘럼 (Curricular) (커리큘럼 샘플링 시 환경 초기화 코드) -----------------------------------------------------------------------------------------------------------
        if object_move == ObjectMoveType.CURRICULAR:
            num_resets = len(env_ids)
            reset_levels = self.current_curriculum_level[env_ids]
            
            # 헬퍼 함수(_generate_positions_for_levels)를 사용하거나, 여기에 직접 가중치 샘플링 로직을 구현합니다.
            # (이전 답변의 가중치 샘플링 로직 전체를 여기에 복사)
            rand_pos = self._generate_positions_for_levels(reset_levels, num_resets)
            rand_reset_pos = rand_pos + self.scene.env_origins[env_ids]
            
            random_angles = torch.rand(num_resets, device=self.device) * 2 * torch.pi
            rand_reset_rot = torch.stack([torch.cos(random_angles / 2), torch.zeros_like(random_angles), torch.zeros_like(random_angles), torch.sin(random_angles / 2)], dim=1)
            rand_reset_box_pose = torch.cat([rand_reset_pos, rand_reset_rot], dim=-1)
            self._box.write_root_pose_to_sim(rand_reset_box_pose, env_ids=env_ids)
            zero_root_velocity = torch.zeros((num_resets, 6), device=self.device)
            self._box.write_root_velocity_to_sim(zero_root_velocity, env_ids=env_ids)
            
            # 동적 이동을 위한 변수 초기화
            self.new_box_pos_rand[env_ids] = self._box.data.body_link_pos_w[env_ids, 0, :].clone()
            self.current_box_rot[env_ids] = self._box.data.body_link_quat_w[env_ids, 0, :].clone()
        
        self.cfg.current_time = 0
        self._compute_intermediate_values(env_ids)
        
        # # -- [추가] 커리큘럼 상태 주기적 출력 --
        # self.log_counter += 1
        # # print(self.log_counter)
        # if self.log_counter % self.LOG_INTERVAL == 0:
        #     # 평균 레벨 계산
        #     avg_level = self.current_curriculum_level.float().mean().item()
        #     # 평균 레벨에 가장 가까운 정수 레벨의 임계값 가져오기
        #     avg_level_idx = round(avg_level)
            
        #     # 레벨 인덱스가 유효한 범위 내에 있는지 확인
        #     if 0 <= avg_level_idx < len(self.curriculum_levels):
        #         success_thresh = self.curriculum_levels[avg_level_idx]["success_threshold"]
        #         failure_thresh = self.curriculum_levels[avg_level_idx]["failure_threshold"]
        #     else:
        #         success_thresh = -1 # 유효하지 않은 경우
        #         failure_thresh = -1

        #     # 레벨 분포 계산 (min, max, median)
        #     min_level = self.current_curriculum_level.min().item()
        #     max_level = self.current_curriculum_level.max().item()
        #     median_level = self.current_curriculum_level.median().item()

        #     # print("---")
        #     # # print(f"[{self.episode_num}] Curriculum Status:")
        #     # print(f"    Avg Level : {avg_level:.2f} (Min: {min_level}, Max: {max_level}, Median: {median_level})")
        #     # print(f"    Thresholds for Avg Level ({avg_level_idx}): Success > {success_thresh:.2f}, Failure < {failure_thresh:.2f}")
        #     # print("---")
        
        
        super()._reset_idx(env_ids)


    def _get_observations(self) -> dict:
        
        dof_pos_scaled = (
            2.0
            * (self._robot.data.joint_pos - self.robot_dof_lower_limits)
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
            - 1.0
        )
        
        global robot_action
        
        camera_pos_w, camera_rot_w = self.compute_camera_world_pose(self.robot_grasp_pos, self.robot_grasp_rot)
        # print(f"isaac_camera_pos : {camera_pos_w}")
        
        box_pos_cam, box_rot_cam = self.world_to_camera_pose(camera_pos_w, camera_rot_w, self.box_grasp_pos - self.scene.env_origins, self.box_grasp_rot,)
        
        # print(f"isaac_box_cam_pos : {box_pos_cam}")
        # print(f"isaac_box_world_pos : {self.box_grasp_pos}")
        
        if foundationpose_mode:
            rclpy.spin_once(self.foundationpose_node, timeout_sec=0.01)
            pos = self.subscribe_object_pos()
            
            if (pos is not None): #and robot_init_pose:

                # camera_pos_w = self.compute_camera_world_pose(self.robot_grasp_pos, self.robot_grasp_rot)
                # camera_rot_w = self.robot_grasp_rot
                
                foundationpose_pos = pos.repeat(self.num_envs, 1)
                foundationpose_pos_converted = torch.zeros_like(foundationpose_pos)
                
                if robot_type == RobotType.FRANKA:
                    foundationpose_pos_converted[:, 0] = -foundationpose_pos[:, 1]  # x = -y_fp
                    foundationpose_pos_converted[:, 1] =  foundationpose_pos[:, 0]  # y = x_fp
                    foundationpose_pos_converted[:, 2] =  foundationpose_pos[:, 2]  # z = z_fp
                
                elif robot_type == RobotType.UF:
                    foundationpose_pos_converted[:, 0] =  foundationpose_pos[:, 0]
                    foundationpose_pos_converted[:, 1] =  foundationpose_pos[:, 1] 
                    foundationpose_pos_converted[:, 2] =  foundationpose_pos[:, 2] 
                    
                elif robot_type == RobotType.DOOSAN :
                    foundationpose_pos_converted[:, 0] =  foundationpose_pos[:, 0]
                    foundationpose_pos_converted[:, 1] =  foundationpose_pos[:, 1] 
                    foundationpose_pos_converted[:, 2] =  foundationpose_pos[:, 2] 
                
                fp_world_pos, _ = self.camera_to_world_pose(camera_pos_w, camera_rot_w, foundationpose_pos_converted, self.box_grasp_rot,)

                # print(f"isaac_cam_pos : {box_pos_cam}")
                # print(f"fp_cam_pos : {foundationpose_pos_converted}")
                # print(f"isaac_world_pos : {self.box_grasp_pos}")
                # print(f"fp_world_pos : {fp_world_pos}")

                to_target = fp_world_pos - self.robot_grasp_pos
                
                # obs = torch.cat(
                #     (
                #         dof_pos_scaled,
                #         self._robot.data.joint_vel * self.cfg.dof_velocity_scale,
                #         to_target,
                #         self._box.data.body_link_pos_w[:, 0, 2].unsqueeze(-1),
                #         self._box.data.body_link_vel_w[:, 0, 2].unsqueeze(-1),
                #     ),
                #     dim=-1,
                # )
                    
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
        # distance_reward_scale = 8.0
        # vector_align_reward_scale = 6.0
        # position_align_reward_scale = 4.0
        # pview_reward_scale = 10.0
        # veloity_align_reward_scale = 0.0
        # joint_penalty_scale = 2.0
        
        levels = self.current_reward_level

        # 새로운 보상 커리큘럼의 'reward_scales'를 참조
        distance_reward_scale = torch.tensor([reward_curriculum_levels[l.item()]["reward_scales"]["distance"] for l in levels], device=self.device)
        vector_align_reward_scale = torch.tensor([reward_curriculum_levels[l.item()]["reward_scales"]["vector_align"] for l in levels], device=self.device)
        position_align_reward_scale = torch.tensor([reward_curriculum_levels[l.item()]["reward_scales"]["position_align"] for l in levels], device=self.device)
        pview_reward_scale = torch.tensor([reward_curriculum_levels[l.item()]["reward_scales"]["pview"] for l in levels], device=self.device)
        joint_penalty_scale = torch.tensor([reward_curriculum_levels[l.item()]["reward_scales"]["joint_penalty"] for l in levels], device=self.device)

        # print("distance_reward_scale : ", distance_reward_scale)
        # print("vector_align_reward_scale : ", vector_align_reward_scale)
        # print("position_align_reward_scale : ", position_align_reward_scale)
        # print("pview_reward_scale : ", pview_reward_scale)
        # print("joint_penalty_scale : ", joint_penalty_scale)
        
        eps = 1e-6
        
        ## 거리 유지 보상 --------------------------------------------------------------------------
        if robot_type == RobotType.FRANKA or robot_type == RobotType.UF:
            min_dist = 0.25
            max_dist = 0.35
            target_distance = 0.30
            
        elif robot_type == RobotType.DOOSAN:
            min_dist = 0.30
            max_dist = 0.40
            target_distance = 0.35
        
        gripper_to_box_dist = torch.norm(franka_grasp_pos - box_pos_w, p=2, dim=-1)
        distance_error = torch.abs(gripper_to_box_dist - target_distance)
        
        within_range = (gripper_to_box_dist >= min_dist) & (gripper_to_box_dist <= max_dist)
        too_close = gripper_to_box_dist < min_dist
        too_far = gripper_to_box_dist > max_dist
        
        distance_reward = torch.zeros_like(gripper_to_box_dist)
       
        k = 2.0 
        distance_reward[within_range] = 1.0 - k * distance_error[within_range]
        
        distance_reward[too_close] = -1.0 * torch.tanh(1.0 * distance_error[too_close])
        distance_reward[too_far] = -3.0 * torch.tanh(5.0 * distance_error[too_far])
        
        ## 각도 정렬 보상--------------------------------------------------------------------------
        # if not hasattr(self, "init_grasp_pos"):
        #     self.init_grasp_pos = franka_grasp_pos.clone().detach()
        
        # robot_base_offset = torch.tensor([0.0, 0.0, 0.0], device=self.device)
        # robot_base_pos_xy = self.scene.env_origins + robot_base_offset
        
        # grasp_axis_origin = torch.zeros_like(self.init_grasp_pos)
        # grasp_axis_origin[:, 0] = robot_base_pos_xy[:, 0]
        # grasp_axis_origin[:, 1] = robot_base_pos_xy[:, 1]
        # grasp_axis_origin[:, 2] = robot_base_pos_xy[:, 2]
            
        # grasp_vec = box_pos_w - grasp_axis_origin 
        # grasp_axis = grasp_vec / (torch.norm(grasp_vec, dim=-1, keepdim=True) + eps)
        
        # gripper_forward = tf_vector(franka_grasp_rot, gripper_forward_axis)
        
        # alignment_cos = torch.sum(gripper_forward * grasp_axis, dim=-1).clamp(-1.0, 1.0)

        # vector_alignment_reward = torch.where(
        #     alignment_cos >= vector_align_margin,
        #     alignment_cos,
        #     -1.0 * (1.0 - alignment_cos)
        # )
        
        box_z = box_pos_w[:, 2]
        top_mask = box_z > workspace_zones["z"]["middle"]    # z >= 0.50 인 환경들
        middle_mask = (box_z > workspace_zones["z"]["bottom"]) & ~top_mask # 0.25 <= z < 0.50 인 환경들
        bottom_mask = ~top_mask & ~middle_mask                     # z < 0.25 인 환경들

        target_angle_rad = torch.zeros_like(box_z)
        target_angle_rad[top_mask] = math.radians(20.0)    # Top 구역 목표
        target_angle_rad[middle_mask] = math.radians(0.0)   # Middle 구역 목표
        target_angle_rad[bottom_mask] = math.radians(-40.0)  # Bottom 구역 목표
        
        gripper_forward = tf_vector(franka_grasp_rot, gripper_forward_axis)
        actual_angle_rad = torch.asin(gripper_forward[:, 2].clamp(-1.0, 1.0))
        
        angle_error_rad = torch.abs(actual_angle_rad - target_angle_rad)

        within_tolerance = angle_error_rad <= vector_align_margin

        cos_reward = torch.cos((angle_error_rad / vector_align_margin) * (math.pi / 2))

        outside_error = angle_error_rad - vector_align_margin
        max_penalty = -2.0
        progressive_penalty = max_penalty * torch.tanh(5.0 * torch.clamp(outside_error, min=0.0))

        vector_alignment_reward = torch.where(
            within_tolerance,
            cos_reward,
            progressive_penalty
        )
        
        ## 그리퍼 위치 유지 보상--------------------------------------------------------------------------
        # grasp_axis = box_pos_w - self.scene.env_origins
        # grasp_axis[..., 2] = 0.0
        # print("self.scene.env_origins : ", self.scene.env_origins)
        
        # epsilon = 1e-6
        # grasp_axis = grasp_axis / (torch.norm(grasp_axis, dim=-1, keepdim=True) + epsilon)

        # vec_from_box_to_gripper_xy = franka_grasp_pos - box_pos_w
        # vec_from_box_to_gripper_xy[..., 2] = 0.0
        # print("franka_grasp_pos : ", franka_grasp_pos)
        # print("box_pos_w : ", box_pos_w)
        
        # gripper_proj_dist = torch.norm(torch.cross(vec_from_box_to_gripper_xy, grasp_axis, dim=-1),dim=-1)        
        
        # start_reward = 0.0        
        # slope = -10.0

        # position_alignment_reward = slope * (gripper_proj_dist - position_align_margin) + start_reward
        # position_alignment_reward = torch.clamp(position_alignment_reward, min=-3.0)
        
        # positive_mask = position_alignment_reward > 0.0
        # position_alignment_reward = torch.where(
        #     positive_mask,
        #     position_alignment_reward + 1,
        #     position_alignment_reward
        # )
        
        # sensitivity = 20.0
        
        # exp_reward = torch.exp(-sensitivity * gripper_proj_dist)
        # penalty_threshold = position_align_margin # 기존 마진을 임계값으로 사용
        # max_penalty = -3.0

        # error_dist = gripper_proj_dist - penalty_threshold
        # error_dist_clamped = torch.clamp(error_dist, min=0.0)

        # progressive_penalty = max_penalty * torch.tanh(10.0 * error_dist_clamped)

        # is_far_away = gripper_proj_dist > penalty_threshold
        # position_alignment_reward = torch.where(
        #     is_far_away,
        #     progressive_penalty, 
        #     exp_reward           
        # )
        robot_origin = self.scene.env_origins + torch.tensor([0.0, 0.0, 0.0], device=self.scene.env_origins.device)

        grasp_axis = box_pos_w - robot_origin
        grasp_axis[..., 2] = 0.0
        grasp_axis = torch.nn.functional.normalize(grasp_axis, p=2, dim=-1)

        box_to_gripper_vec_xy = franka_grasp_pos - box_pos_w
        box_to_gripper_vec_xy[..., 2] = 0.0

        gripper_proj_dist = torch.norm(torch.cross(box_to_gripper_vec_xy, grasp_axis, dim=-1), dim=-1)
        # is_within_margin = gripper_proj_dist <= position_align_margin

        # exp_reward = torch.exp(-20 * gripper_proj_dist)

        # error_dist = gripper_proj_dist - position_align_margin
        # progressive_penalty = -3 * torch.tanh(10 * torch.clamp(error_dist, min=0.0))

        # position_alignment_reward = torch.where(
        #     is_within_margin,
        #     exp_reward,
        #     progressive_penalty
        # )
        
        is_within_margin = gripper_proj_dist <= position_align_margin

        margin_val_tensor = torch.tensor(-20 * position_align_margin, device=gripper_proj_dist.device)
        min_val_at_margin = torch.exp(margin_val_tensor)
        positive_reward = torch.exp(-20 * gripper_proj_dist) - min_val_at_margin

        max_val = 1.0 - min_val_at_margin
        positive_reward = positive_reward / max_val

        error_dist = gripper_proj_dist - position_align_margin
        progressive_penalty = -3 * torch.tanh(10 * torch.clamp(error_dist, min=0.0))

        position_alignment_reward = torch.where(
            is_within_margin,
            positive_reward,      # 스케일링된 새로운 보상 적용
            progressive_penalty
        )
        
        # print("=====================================")
        # print("robot_origin : ", robot_origin)
        # print("box_pos_w : ", box_pos_w)
        # print("franka_grasp_pos : ", franka_grasp_pos)
        # print("box_to_gripper_vec_xy : ", box_to_gripper_vec_xy)
        # print("gripper_proj_dist : ", gripper_proj_dist)
        # print("position_alignment_reward : ", position_alignment_reward)

        # gripper_pos_xy = franka_grasp_pos.clone()
        # gripper_pos_xy[:, 2] = 0.0
        # box_pos_xy = box_pos_w.clone()
        # box_pos_xy[:, 2] = 0.0
        # base_pos_xy = self.scene.env_origins.clone() 
        # base_pos_xy[:, 2] = 0.0

        # ideal_horizontal_path_vec = box_pos_xy - base_pos_xy

        # base_to_gripper_vec = gripper_pos_xy - base_pos_xy
        # cross_product_mag = torch.norm(torch.cross(ideal_horizontal_path_vec, base_to_gripper_vec, dim=-1), dim=-1)
        # path_length = torch.norm(ideal_horizontal_path_vec, dim=-1) + eps
        # horizontal_deviation_dist = cross_product_mag / path_length

        # sensitivity = 20.0

        # exp_reward = torch.exp(-sensitivity * horizontal_deviation_dist)
        # is_far_away = horizontal_deviation_dist > position_align_margin

        # error_dist = horizontal_deviation_dist - position_align_margin
        # error_dist_clamped = torch.clamp(error_dist, min=0.0)
        # max_penalty = -3.0
        # progressive_penalty = max_penalty * torch.tanh(10.0 * error_dist_clamped)

        # position_alignment_reward = torch.where(
        #     is_far_away,
        #     progressive_penalty, 
        #     exp_reward           
        # )
        
        # gripper_pos_xy = franka_grasp_pos.clone()
        # gripper_pos_xy[:, 2] = 0.0
        # print("gripper_pos_xy :", gripper_pos_xy)
        # box_pos_xy = box_pos_w.clone()
        # box_pos_xy[:, 2] = 0.0
        # print("box_pos_xy :", box_pos_xy)
        # base_pos_xy = self.scene.env_origins.clone()
        # base_pos_xy[:, 2] = 0.0
        # print("base_pos_xy :", base_pos_xy)


        # ideal_horizontal_path_vec = box_pos_xy - base_pos_xy
        # base_to_gripper_vec = gripper_pos_xy - base_pos_xy
        
        # cross_product_mag = torch.norm(torch.cross(ideal_horizontal_path_vec, base_to_gripper_vec, dim=-1), dim=-1)
        # path_length = torch.norm(ideal_horizontal_path_vec, dim=-1) + eps
        # horizontal_deviation_dist = cross_product_mag / path_length

        # sensitivity = 20.0

        # exp_reward = torch.exp(-sensitivity * horizontal_deviation_dist)
        # penalty_threshold = position_align_margin
        # max_penalty = -3.0

        # error_dist = horizontal_deviation_dist - penalty_threshold
        # error_dist_clamped = torch.clamp(error_dist, min=0.0)

        # progressive_penalty = max_penalty * torch.tanh(10.0 * error_dist_clamped)

        # is_far_away = horizontal_deviation_dist > penalty_threshold
        # position_alignment_reward = torch.where(
        #     is_far_away,
        #     progressive_penalty, 
        #     exp_reward           
        # )
        
        ## 시야 유지 보상
        is_in_front_mask = box_pos_cam[:, 2] > 0
        center_offset = torch.norm(box_pos_cam[:, :2], dim=-1)
        
        out_of_fov_mask = center_offset > pview_margin

        pview_reward_candidate = torch.where(
            out_of_fov_mask,
            torch.full_like(center_offset, -5.0),
            1.0 * torch.exp(-10.0 * center_offset)
        )

        pview_reward = torch.where(
            is_in_front_mask,
            pview_reward_candidate,
            torch.full_like(center_offset, -10.0) 
        )
        
        ## 자세 안정성 유지 패널티
        
        if not hasattr(self, "init_robot_joint_position"):
            self.init_robot_joint_position = self._robot.data.joint_pos.clone()
        
        joint_deviation = torch.abs(self._robot.data.joint_pos - self.init_robot_joint_position)
        joint_weights = torch.ones_like(joint_deviation)
        
        if robot_type == RobotType.FRANKA:
            joint_weights[:, 2] = 0.0
            joint_weights[:, 4] = 0.0 
        elif robot_type == RobotType.UF:
            joint4_idx = self._robot.find_joints(["joint4"])[0]
            joint6_idx = self._robot.find_joints(["joint6"])[0]
            joint_weights[:, joint4_idx] = 0.0
            joint_weights[:, joint6_idx] = 0.0
        elif robot_type == RobotType.DOOSAN:
            joint4_idx = self._robot.find_joints(["J4_joint"])[0]
            joint6_idx = self._robot.find_joints(["J6_joint"])[0]
            joint_weights[:, joint4_idx] = 0.0
            joint_weights[:, joint6_idx] = 0.0
            
        weighted_joint_deviation = joint_deviation * joint_weights
        joint_penalty = torch.sum(weighted_joint_deviation, dim=-1)
        joint_penalty = torch.tanh(joint_penalty)
        
        ## 최종 보상 계산
        rewards = (
            distance_reward_scale * distance_reward  
            + vector_align_reward_scale * vector_alignment_reward
            + position_align_reward_scale * position_alignment_reward
            + pview_reward_scale * pview_reward
            - joint_penalty_scale * joint_penalty 
        )
        
        # print("=====================================")
        # print("gripper_to_box_dist : ", gripper_to_box_dist)
        # print("distance_reward : ", distance_reward)
        # # # print("alignment_cos : ", alignment_cos)
        # # # print("grasp_axis_origin : ", {grasp_axis_origin})
        # print("vector_alignment_reward:", vector_alignment_reward)
        # # print("angle_error_rad:", angle_error_rad)
        # print("position_alignment_reward:", position_alignment_reward)
        # # # print("center_offset:", center_offset)
        # print("pview_reward:", pview_reward)
        # # print(f"ee_motion_penalty : {ee_motion_penalty}")

        #2025.08.19
        
        self.last_step_reward = rewards.detach()
        
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
        
        
        