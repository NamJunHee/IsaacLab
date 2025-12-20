# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
torch.set_printoptions(precision=4, sci_mode=False)

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
# from isaaclab.utils.math import sample_uniform

from isaaclab.sensors import CameraCfg, Camera
from isaaclab.assets import RigidObjectCfg, RigidObject
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg #, CollisionPropertiesCfg
from isaaclab.sim.schemas import modify_collision_properties, CollisionPropertiesCfg
from isaaclab.sensors import CameraCfg, Camera, ContactSensor, ContactSensorCfg # [수정] ContactSensor, ContactSensorCfg 추가

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
import time

import matplotlib.pyplot as plt
import numpy as np
import csv
import os

from pxr import Usd, UsdPhysics, PhysxSchema, UsdGeom
import omni.usd
from isaaclab.sim import spawn_from_usd

import multiprocessing as mp
from multiprocessing import Process, Queue
import queue # Full 예외 처리를 위해 필요

import atexit

class RobotType(Enum):
    FRANKA = "franka"
    UF = "ufactory"
    DOOSAN = "doosan"
robot_type = RobotType.UF

class ObjectMoveType(Enum):
    STATIC = "static"
    LINEAR = "linear"
# object_move = ObjectMoveType.STATIC
object_move = ObjectMoveType.LINEAR

training_mode = False

approach_mode = False
grasp_mode = False

camera_enable = True
image_publish = True
test_graph_mode = True

robot_action = False
robot_init_pose = False
robot_fix = False

init_reward = True
reset_flag = True

add_episode_length = 200
# add_episode_length = -800
# add_episode_length = -900

rand_pos_range = {
    "x" : (  0.35, 0.75),
    "y" : ( -0.50, 0.50),
    "z" : (  0.08, 0.75),
    
    # "x" : (  0.25, 0.45),
    # "y" : ( -0.00, 0.00),
    # "z" : (  0.65, 0.65),
}

reward_curriculum_levels = [
    # Level 0: (Static) - 기초 단계
    {
        "reward_scales": {
            "distance": 6.0,      # [핵심] 1.0 -> 6.0 (접근이 깡패다)
            "pview": 0.5,         # [하향] 1.0 -> 0.5 (시야는 Gating용)
            "vector_align": 0.5,  # 0.6 -> 0.5
            "position_align": 0.5,# 0.8 -> 0.5
            "joint_penalty": 1.0, # [유지] 손목 보호를 위해 1.0 유지
            "blind_penalty": 1.0  # [상향] 0.1 -> 1.0 (놓치면 점수 다 뱉어내라)
        },
        "success_multiplier": 1.2, "failure_multiplier": 0.8, 
        "y_range" : ( -0.50, 0.50),

        "distance_margin" : 0.15,
        "vector_align_margin" : math.radians(20.0),
        "position_align_margin" : 0.20,
        "pview_margin" : 0.25,
        "fail_margin" : 0.35,
    },
    # Level 1: (Moving Slow) - 추적 시작
    {
        "reward_scales": {
            "distance": 6.0,      # [핵심] 공격적 접근 유도
            "pview": 0.5,
            "vector_align": 0.5,
            "position_align": 0.5,
            "joint_penalty": 1.0, # [유지]
            "blind_penalty": 1.0  # [상향]
        },
        "success_multiplier": 1.0, "failure_multiplier": 1.2, 
        "y_range" : ( -0.50, 0.50),

        "distance_margin" : 0.15, 
        "vector_align_margin" : math.radians(25.0),
        "position_align_margin" : 0.25,
        "pview_margin" : 0.25,
        "fail_margin" : 0.35,
    },
    # Level 2: (Moving Planar) - 여기가 고비였던 구간
    {
        "reward_scales": {
            "distance": 6.0,      # [핵심] 멀어지는 물체 쫓아가려면 보상이 커야 함
            "pview": 0.5,
            "vector_align": 0.5,
            "position_align": 0.5,
            "joint_penalty": 1.0, # [유지]
            "blind_penalty": 1.0  # [상향]
        },
        "success_multiplier": 0.9, "failure_multiplier": 1.0, 
        "y_range": (-0.50, 0.50),

        "distance_margin" : 0.10,
        "vector_align_margin" : math.radians(20.0),
        "position_align_margin" : 0.20,
        "pview_margin" : 0.20,
        "fail_margin" : 0.30
    },
    # Level 3: (Moving Fast)
    {
        "reward_scales": {
            "distance": 6.0, 
            "pview": 0.5, 
            "vector_align": 0.5, 
            "position_align": 0.5, 
            "joint_penalty": 1.0, 
            "blind_penalty": 1.0 
        },
        "success_multiplier": 0.8, "failure_multiplier": 1.0, 
        "y_range": (-0.50, 0.50),

        "distance_margin" : 0.10,
        "vector_align_margin" : math.radians(15.0),
        "position_align_margin" : 0.15,
        "pview_margin" : 0.20,
        "fail_margin" : 0.30
    },
    # Level 4: (Moving Very Fast)
    {
        "reward_scales": {
            "distance": 6.0, 
            "pview": 0.5, 
            "vector_align": 0.5, 
            "position_align": 0.5, 
            "joint_penalty": 1.0, 
            "blind_penalty": 1.5 # [최상위] 더 엄격하게
        },
        "success_multiplier": 1.0, "failure_multiplier": 1.2, 
        "y_range": (-0.50, 0.50),

        "distance_margin" : 0.05,
        "vector_align_margin" : math.radians(10.0),
        "position_align_margin" : 0.10,
        "pview_margin" : 0.15,
        "fail_margin" : 0.25,
    },
    {
        "reward_scales": {
            "distance": 6.0, 
            "pview": 0.5, 
            "vector_align": 0.5, 
            "position_align": 0.5, 
            "joint_penalty": 1.0, 
            "blind_penalty": 1.5 # [최상위] 더 엄격하게
        },
        "success_multiplier": 1.0, "failure_multiplier": 1.2, 
        "y_range": (-0.50, 0.50),

        "distance_margin" : 0.05,
        "vector_align_margin" : math.radians(10.0),
        "position_align_margin" : 0.10,
        "pview_margin" : 0.15,
        "fail_margin" : 0.25,
    },
]

pose_candidate = {
    # "zero" : {"joint1": math.radians(0.0), 
    #                   "joint2": math.radians(0.0), 
    #                   "joint3": math.radians(0.0), 
    #                   "joint4": math.radians(0.0), 
    #                   "joint5": math.radians(0.0), 
    #                   "joint6": math.radians(0.0)},
    
    ## top------------------------------------------------
    # "top_close":   {"joint1": math.radians(0.0), 
    #                   "joint2": math.radians(-50.0), 
    #                   "joint3": math.radians(-30.0), 
    #                   "joint4": math.radians(0.0), 
    #                   "joint5": math.radians(-30.0), 
    #                   "joint6": math.radians(0.0)},
        
    "top_close":   {"joint1": math.radians(0.0), 
                      "joint2": math.radians(-75.0), 
                      "joint3": math.radians(-40.0), 
                      "joint4": math.radians(0.0), 
                      "joint5": math.radians(0.0), 
                      "joint6": math.radians(0.0)},

    # "top_close_2":   {"joint1": math.radians(0.0), 
    #                   "joint2": math.radians(-110.0), 
    #                   "joint3": math.radians(5.0), 
    #                   "joint4": math.radians(0.0), 
    #                   "joint5": math.radians(-5.0), 
    #                   "joint6": math.radians(0.0)},
    
    # "top_middle":   {"joint1": math.radians(  0.0), 
    #                   "joint2": math.radians(-30.0), 
    #                   "joint3": math.radians(-30.0), 
    #                   "joint4": math.radians(  0.0), 
    #                   "joint5": math.radians( -45.0), 
    #                   "joint6": math.radians(  0.0)},
    
    "top_middle":   {"joint1": math.radians(  0.0), 
                      "joint2": math.radians(-25.0), 
                      "joint3": math.radians(-60.0), 
                      "joint4": math.radians(  0.0), 
                      "joint5": math.radians( -30.0), 
                      "joint6": math.radians(  0.0)},
    
    # "top_middle_2":   {"joint1": math.radians(  0.0), 
    #                   "joint2": math.radians(-5.0), 
    #                   "joint3": math.radians(-60.0), 
    #                   "joint4": math.radians(  0.0), 
    #                   "joint5": math.radians( -35.0), 
    #                   "joint6": math.radians(  0.0)},
    
    # "top_far":     {"joint1": math.radians(  0.0), 
    #                   "joint2": math.radians( -20.0),  
    #                   "joint3": math.radians(-45.0), 
    #                   "joint4": math.radians(  0.0), 
    #                   "joint5": math.radians(  -35.0), 
    #                   "joint6": math.radians(  0.0)},
    
    "top_far":     {"joint1": math.radians(  0.0), 
                    "joint2": math.radians(  0.0),  
                    "joint3": math.radians(-90.0), 
                    "joint4": math.radians(  0.0), 
                    "joint5": math.radians(-20.0), 
                    "joint6": math.radians(  0.0)},
    
    # "top_far_2":     {"joint1": math.radians(  0.0), 
    #                   "joint2": math.radians(  0.0),  
    #                   "joint3": math.radians(-65.0), 
    #                   "joint4": math.radians(  0.0), 
    #                   "joint5": math.radians(-35.0), 
    #                   "joint6": math.radians(  0.0)},
    
    ##middle------------------------------------------------
    # "middle_close":  {"joint1": math.radians(0.0), 
    #                   "joint2": math.radians(-110.0),
    #                   "joint3": math.radians( -5.0),  
    #                   "joint4": math.radians(0.0), 
    #                   "joint5": math.radians(45.0), 
    #                   "joint6": math.radians(0.0)},
    
    # "middle_middle": {"joint1": math.radians(0.0), 
    #                   "joint2": math.radians(-40.0), 
    #                   "joint3": math.radians(0.0), 
    #                   "joint4": math.radians(0.0), 
    #                   "joint5": math.radians(-45.0),  
    #                   "joint6": math.radians(0.0)},
    
    # "middle_far":    {"joint1": math.radians(0.0),  
    #                   "joint2": math.radians(15.0),  
    #                   "joint3": math.radians(-50.0),
    #                   "joint4": math.radians(0.0), 
    #                   "joint5": math.radians(-50.0), 
    #                   "joint6": math.radians(0.0)},
    
    "middle_close":  {"joint1": math.radians(  0.0), 
                      "joint2": math.radians(-90.0),
                      "joint3": math.radians(-25.0),  
                      "joint4": math.radians(  0.0), 
                      "joint5": math.radians( 25.0), 
                      "joint6": math.radians( 0.0)},
    
    "middle_middle": {"joint1": math.radians(  0.0), 
                      "joint2": math.radians(-45.0), 
                      "joint3": math.radians(-40.0), 
                      "joint4": math.radians(  0.0), 
                      "joint5": math.radians( -5.0),  
                      "joint6": math.radians(  0.0)},
    
    "middle_far":    {"joint1": math.radians(  0.0),  
                      "joint2": math.radians(  5.0),  
                      "joint3": math.radians(-80.0),
                      "joint4": math.radians(  0.0), 
                      "joint5": math.radians(-15.0), 
                      "joint6": math.radians(  0.0)},

    ##bottom------------------------------------------------
    # "bottom_close":  {"joint1": math.radians(0.0), 
    #                   "joint2": math.radians(-95.0),   
    #                   "joint3": math.radians(-5.0),   
    #                   "joint4": math.radians(0.0), 
    #                   "joint5": math.radians( 50.0),
    #                   "joint6": math.radians(0.0)},
    
    # "bottom2_close2":  {"joint1": math.radians(0.0), 
    #                   "joint2": math.radians(-70.0),   
    #                   "joint3": math.radians( 0.0),   
    #                   "joint4": math.radians(0.0), 
    #                   "joint5": math.radians( 35.0),
    #                   "joint6": math.radians(0.0)},
    
    # "bottom_middle": {"joint1": math.radians(0.0), 
    #                   "joint2": math.radians(-60.0),  
    #                   "joint3": math.radians(-0.0), 
    #                   "joint4": math.radians(0.0), 
    #                   "joint5": math.radians(10.0),
    #                   "joint6": math.radians(0.0)},
    
    # "bottom_middle_2": {"joint1": math.radians(0.0), 
    #                   "joint2": math.radians(-30.0),  
    #                   "joint3": math.radians(-0.0), 
    #                   "joint4": math.radians(0.0), 
    #                   "joint5": math.radians(-10.0),
    #                   "joint6": math.radians(0.0)},
    
    # "bottom_far":    {"joint1": math.radians(0.0), 
    #                   "joint2": math.radians(-25.0),  
    #                   "joint3": math.radians(-15.0),
    #                   "joint4": math.radians(0.0), 
    #                   "joint5": math.radians(-5.0), 
    #                   "joint6": math.radians(0.0)},
    
    # "bottom_far_2":    {"joint1": math.radians(0.0), 
    #                   "joint2": math.radians(15.0),  
    #                   "joint3": math.radians(-45.0),
    #                   "joint4": math.radians(0.0), 
    #                   "joint5": math.radians(-5.0), 
    #                   "joint6": math.radians(0.0)},
    
    "bottom_close":    {"joint1": math.radians(  0.0), 
                        "joint2": math.radians(-95.0),  
                        "joint3": math.radians(-10.0),
                        "joint4": math.radians(  0.0), 
                        "joint5": math.radians( 60.0), 
                        "joint6": math.radians(  0.0)},
    
    "bottom_middle":   {"joint1": math.radians(  0.0), 
                        "joint2": math.radians(-40.0),  
                        "joint3": math.radians(-25.0),
                        "joint4": math.radians(  0.0), 
                        "joint5": math.radians( 20.0), 
                        "joint6": math.radians(  0.0)},
    
    "bottom_far":      {"joint1": math.radians(  0.0), 
                        "joint2": math.radians(  5.0),  
                        "joint3": math.radians(-55.0),
                        "joint4": math.radians(  0.0), 
                        "joint5": math.radians(  5.0), 
                        "joint6": math.radians(  0.0)},
}

# initial_pose = pose_candidate["bottom_close"]
initial_pose = pose_candidate["middle_close"]
# initial_pose = pose_candidate["top_close"]
# initial_pose = pose_candidate["zero"]

workspace_zones = {
    "x": {"close" : 0.35, "middle": 0.50,"far": 0.65},
    "z": {"bottom": 0.30, "middle": 0.50,"top": 0.65}
}

x_weights = {"far": 5.0, "middle": 1.0, "close" : 4.0}
z_weights = {"top": 1.0, "middle": 1.0, "bottom": 10.0}

zone_activation = {
    "top_close":    True,
    "top_middle":   True,
    "top_far":      True, # << 이 값을 False로 바꾸면 제외됩니다.
    "middle_close": True,
    "middle_middle":True,
    "middle_far":   True,
    "bottom_close": True,
    "bottom_middle":True,
    "bottom_far":   True,
}

zone_definitions = {
    "top_close":    {"x": (workspace_zones["x"]["middle"], workspace_zones["x"]["far"]),   "z": (workspace_zones["z"]["middle"], rand_pos_range["z"][1])},
    "top_middle":   {"x": (workspace_zones["x"]["middle"], workspace_zones["x"]["far"]),   "z": (workspace_zones["z"]["middle"], rand_pos_range["z"][1])},
    "top_far":      {"x": (workspace_zones["x"]["far"],   rand_pos_range["x"][1]),         "z": (workspace_zones["z"]["middle"], rand_pos_range["z"][1])},
    "middle_close": {"x": (rand_pos_range["x"][0], workspace_zones["x"]["middle"]), "z": (workspace_zones["z"]["bottom"], workspace_zones["z"]["middle"])},
    "middle_middle":{"x": (workspace_zones["x"]["middle"], workspace_zones["x"]["far"]),   "z": (workspace_zones["z"]["bottom"], workspace_zones["z"]["middle"])},
    "middle_far":   {"x": (workspace_zones["x"]["far"],   rand_pos_range["x"][1]),         "z": (workspace_zones["z"]["bottom"], workspace_zones["z"]["middle"])},
    "bottom_close": {"x": (rand_pos_range["x"][0], workspace_zones["x"]["middle"]), "z": (rand_pos_range["z"][0], workspace_zones["z"]["bottom"])},
    "bottom_middle":{"x": (workspace_zones["x"]["middle"], workspace_zones["x"]["far"]),   "z": (rand_pos_range["z"][0], workspace_zones["z"]["bottom"])},
    "bottom_far":   {"x": (workspace_zones["x"]["far"],   rand_pos_range["x"][1]),         "z": (rand_pos_range["z"][0], workspace_zones["z"]["bottom"])},
    # "bottom2_close2": {"x": (rand_pos_range["x"][0], workspace_zones["x"]["close"]), "z": (rand_pos_range["z"][0], workspace_zones["z"]["bottom"])}
}
zone_keys = list(pose_candidate.keys())

CSV_FILEPATH = "/home/nmail-njh/NMAIL/01_Project/Robot_Grasping/IsaacLab/tracking_data.csv"

@configclass
class FrankaObjectTrackingEnvCfg(DirectRLEnvCfg):
    ## env
    episode_length_s = 8.3333  # 500 timesteps
    decimation = 2
    action_space = 6
    # observation_space = 21
    # observation_space = 24
    observation_space = 23
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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, 
        env_spacing=3.0, 
        replicate_physics=True,
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
    
    ## robot
    UF_robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/xarm6",
        spawn=sim_utils.UsdFileCfg(
            # usd_path="/home/nmail-njh/NMAIL/01_Project/Robot_Grasping/IsaacLab/ROBOT/xarm6_robot_white/xarm6_robot_white.usd",
            usd_path="/home/nmail-njh/NMAIL/01_Project/Robot_Grasping/IsaacLab/ROBOT/Ufactory/xarm6/xarm6.usd",
            
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=24, solver_velocity_iteration_count=1
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos = initial_pose,
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "ufactory_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["joint1", "joint2", "joint3"],
                effort_limit = 87.0,
                velocity_limit = 3.5,
                stiffness = 2000.0,
                damping = 100.0,
            ),
            "ufactory_forearm": ImplicitActuatorCfg(
                joint_names_expr=["joint4", "joint5", "joint6"],
                effort_limit = 87.0,
                velocity_limit = 3.5,
                stiffness = 2000.0,
                damping = 100.0,
            ),
            "ufactory_hand": ImplicitActuatorCfg(
                # xArm6 그리퍼의 메인 관절 이름이 "drive_joint" 입니다.
                joint_names_expr=[".*drive_joint"], 
                effort_limit=100.0,   # 충분한 힘
                velocity_limit=2.0,   # 적절한 속도
                stiffness=5000.0,     # [중요] 위치 고정을 위해 높은 강성(Stiffness) 필요
                damping=100.0,        # 진동 방지
            ),
        },
    )
    
    contact_forces = ContactSensorCfg(
        prim_path="/World/envs/env_.*/xarm6/gripper/.*(finger|knuckle|base_link)",
        update_period=0.0, 
        history_length=6, 
        debug_vis=True
    )

    ## camera
    if camera_enable:
        camera = CameraCfg(
            # prim_path="/World/envs/env_.*/xarm6/gripper/xarm_gripper_base_link/hand_camera",
            prim_path="/World/envs/env_.*/xarm6/link5/hand_camera",
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
                # pos=(0.07, 0.03, -0.13), # 위/아래, 좌/우, 앞/뒤
                # pos=(0.07, 0.03, 0.055), # 위/아래, 좌/우, 앞/뒤
                # rot=(0.7071, 0.0, 0.0, 0.7071),
                
                pos=(0.2, 0.03, 0.00), # 위/아래, 좌/우, 앞/뒤
                rot=(0.5, -0.5, 0.5, 0.5),
            )
        )
            
    
    ## mustard
    box = RigidObjectCfg(
        prim_path="/World/envs/env_.*/base_link",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0, 0.08), rot=(0, 0, 0, 0)),
        spawn=UsdFileCfg(
            # usd_path="/home/nmail-njh/NMAIL/01_Project/Robot_Grasping/objects_usd/google_objects_usd/006_mustard_bottle/006_mustard_bottle.usd",
            usd_path="/home/nmail-njh/NMAIL/01_Project/Robot_Grasping/objects_usd/google_objects_usd/010_potted_meat_can/010_potted_meat_can.usd",
            scale=(0.7, 0.7, 0.7),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity = True,
                kinematic_enabled = False,
            ),
            collision_props=CollisionPropertiesCfg(
                collision_enabled = True,  # [핵심] 0으로 설정하면 "아무것과도 충돌하지 않음"
            ),
        ),    
    )
    
    action_scale = 4.0
    dof_velocity_scale = 0.07
    blind_penalty_scale = 0.5

    #time
    current_time = 0.0

class FrankaObjectTrackingEnv(DirectRLEnv):
    cfg: FrankaObjectTrackingEnvCfg

    def __init__(self, cfg: FrankaObjectTrackingEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        self.dt = self.cfg.sim.dt * self.cfg.decimation

        if not training_mode and test_graph_mode:
            # 1. CSV 파일 경로 설정 (이전 에러 해결)
            self.csv_filepath = CSV_FILEPATH 
            
            if os.path.exists(self.csv_filepath):
                try:
                    os.remove(self.csv_filepath)
                except OSError:
                    pass
            
            with open(self.csv_filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['grip_x', 'grip_y', 'grip_z', 
                                 'obj_x', 'obj_y', 'obj_z', 
                                 'cam_x', 'cam_y', 
                                 'distance',
                                 'grasp_angle', 'grasp_width'])
                
        self.log_counter = 0
        self.LOG_INTERVAL = 6 
        
        # ----------------------------------------------------------------
        # 3. 커리큘럼 및 평가 기준 설정 (Curriculum & Evaluation)
        # ----------------------------------------------------------------
        self.max_reward_level = len(reward_curriculum_levels) - 1
        self.current_reward_level = torch.full((self.num_envs,), 0, dtype=torch.long, device=self.device)
        self.baseline_avg_reward = 0.05 # 계산된 기준 보상값
        
        # 커리큘럼 계수 (Teacher-Student Learning 관련)
        self.curriculum_factor_k0 = 0.25  # k_c의 초기값 (논문 권장값)
        self.curriculum_factor_kd = 0.997 # k_c의 진전 속도
        self.curriculum_factor_k_c = torch.full((self.num_envs, 1), self.curriculum_factor_k0, device=self.device)
        
        # 에피소드 성공 판단 기준
        self.EVAL_BATCH_SIZE = 10     # 20판마다 성적 평가
        self.PROMOTION_RATE = 0.80     # 승률 90% 이상이면 레벨업 (20판 중 18승)
        self.DEMOTION_RATE = 0.10      # 승률 40% 이하면 레벨다운 (20판 중 8승 이하)

        # 세부 성공 조건
        self.MIN_PVIEW_RATIO = 0.90  # 에피소드 길이의 90% 이상 유지
        self.MAX_DISTANCE_ERROR = 0.05  # 평균 거리 오차 5cm 미만
        self.MIN_TOTAL_REWARD = 200.0    # 에피소드 총 보상 최소 60.0 이상
        
        # 성적 기록용 버퍼
        self.success_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.failure_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.episode_reward_buf = torch.zeros(self.num_envs, device=self.device)
        self.avg_distance_error_buf = torch.zeros(self.num_envs, device=self.device)
        self.success_steps_buf = torch.zeros(self.num_envs, device=self.device)
        
        # ----------------------------------------------------------------
        # 4. 로봇 설정 (Robot Config)
        # ----------------------------------------------------------------
        self.joint_names = ["joint1", "joint2", "joint3", "joint4","joint5", "joint6"]
        self.joint_init_values = [initial_pose[name] for name in self.joint_names]
        
        # create auxiliary variables for computing applied action, observations and rewards
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)
        
        self.episode_init_joint_pos = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)
        self.current_joint_pos_buffer = self._robot.data.joint_pos.clone()

        self.action_scale_tensor = torch.full(
            (self.num_envs,), 0.5, device=self.device, dtype=torch.float32
        )
        
        # ----------------------------------------------------------------
        # 5. 물체(Box) 이동 및 상태 설정 (Object State)
        # ----------------------------------------------------------------
        self.MOVE_STATE_STATIC = 0
        self.MOVE_STATE_LINEAR = 1

        self.object_move_state = torch.full(
            (self.num_envs,), self.MOVE_STATE_STATIC, dtype=torch.long, device=self.device
        )
        self.obj_speed = torch.zeros(
            (self.num_envs,), device=self.device, dtype=torch.float32
        )
        
        self.speed_change_timer = torch.zeros(self.num_envs, device=self.device)
        self.current_speed_factor = torch.ones(self.num_envs, device=self.device)        
        
        self.level1_axis_mode = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.level2_plane_mode = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # ----------------------------------------------------------------
        # 6. 작업 공간 및 초기 자세 설정 (Workspace & Reference Poses)
        # ----------------------------------------------------------------
        self.zone_names_ordered = [
            "top_close", "top_middle", "top_far",
            "middle_close", "middle_middle", "middle_far",
            "bottom_close", "bottom_middle", "bottom_far"
        ]
        
        ref_poses = []
        for name in self.zone_names_ordered:
            pose_dict = pose_candidate[name]
            joint_vals = [pose_dict[jn] for jn in self.joint_names if jn in pose_dict]
            
            if len(joint_vals) < len(self.joint_names):
                joint_vals = [0.0] + joint_vals 
            ref_poses.append(joint_vals)
        self.ref_pose_tensor = torch.tensor(ref_poses, device=self.device, dtype=torch.float32)
        
        # ----------------------------------------------------------------
        # 7. 기하학적 계산 및 링크 인덱싱 (Geometry & Kinematics)
        # ----------------------------------------------------------------
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
        
        stage = get_current_stage()
        
        self.hand_link_idx = self._robot.find_bodies("link6")[0][0]
        self.box_idx = self._box.find_bodies("base_link")[0][0]
        
        hand_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/xarm6/link6")),
            self.device,
        )
        lfinger_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/xarm6/link6")),
            self.device,
        )
        rfinger_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/xarm6/link6")),
            self.device,
        )
        
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
        
        self.gripper_forward_axis = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).repeat((self.num_envs, 1))
        self.gripper_up_axis = torch.tensor([1, 0, 0], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        self.box_z_axis = torch.tensor([0,0,1], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs,1)
        )
        
        # ----------------------------------------------------------------
        # 8. 런타임 상태 버퍼 할당 (Runtime Buffers)
        # ----------------------------------------------------------------
        self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)
        self.hand_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.hand_rot = torch.zeros((self.num_envs, 4), device=self.device)
        
        self.box_center = self._box.data.body_link_pos_w[:,0,:].clone()
        self.target_box_pos = torch.stack([
                torch.rand(self.num_envs, device=self.device) * (rand_pos_range["x"][1] - rand_pos_range["x"][0]) + rand_pos_range["x"][0],
                torch.rand(self.num_envs, device=self.device) * (rand_pos_range["y"][1] - rand_pos_range["y"][0]) + rand_pos_range["y"][0],
                torch.rand(self.num_envs, device=self.device) * (rand_pos_range["z"][1] - rand_pos_range["z"][0]) + rand_pos_range["z"][0],
            ], dim = 1) + self.scene.env_origins
        
        self.new_box_pos_rand = torch.zeros((self.num_envs, 3), device=self.device)
        self.rand_pos = torch.zeros((self.num_envs, 3), device=self.device)
        # self.rand_pos_step = torch.zeros((self.num_envs, 3), device=self.device)
        
        self.current_box_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.current_box_pos = torch.zeros((self.num_envs, 3), device=self.device)
        
        self.robot_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.robot_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        
        self.box_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.box_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        
        self.box_pos_cam = torch.zeros((self.num_envs, 4), device=self.device)  
        
        # 상태 플래그
        self.init_cnt = 0
        self.out_of_fov_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.is_object_visible_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_publish_time = 0.0
        self.position_error = 0.0
        self.obj_origin_distance = 0.0
                
        rclpy.init()
        
        if image_publish:      
            qos_profile = QoSProfile(depth=10)
            qos_profile.reliability = QoSReliabilityPolicy.BEST_EFFORT

            self.node = rclpy.create_node('camera_publisher')
            self.camera_info_publisher = self.node.create_publisher(CameraInfo, '/camera_info_rect',10)
            self.rgb_publisher = self.node.create_publisher(Image, '/image_rect',10)
            self.depth_publisher = self.node.create_publisher(Image, '/depth',10)
            
            self.bridge = CvBridge()
        
        self.prev_box_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.prev_box_pos_c = torch.zeros((self.num_envs, 3), device=self.device)
        
        self.last_action_filter = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self.smoothing_alpha = 1.0
        
        self.last_grasp_angle = 0.0
        self.last_grasp_width = 0.0
        

        self.gripper_drive_idx = self._robot.find_joints(".*drive_joint")[0][0]

        # [추가 2] 파지 실험용 상태 머신 변수 초기화
        # 0: Tracking (추적), 1: Approach (접근), 2: Grasping (파지), 3: Success (성공)
        self.grasp_phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.stable_timer = torch.zeros(self.num_envs, device=self.device)
        self.holding_timer = torch.zeros(self.num_envs, device=self.device)
        
        self.is_collision = 0
        
        self.target_grasp_width = torch.zeros(self.num_envs, device=self.device)
        self.target_grasp_angle = torch.zeros(self.num_envs, device=self.device)
        
    def publish_camera_data(self):
        env_id = 0
        
        current_stamp = self.node.get_clock().now().to_msg() 
        current_stamp.sec = current_stamp.sec % 50000
        current_stamp.nanosec = 0
                
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
         
    def quat_mul(self, q, r):
        x1, y1, z1, w1 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        x2, y2, z2, w2 = r[:, 0], r[:, 1], r[:, 2], r[:, 3]

        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2

        quat = torch.stack((x, y, z, w), dim=-1)
        return kornia.geometry.conversions.normalize_quaternion(quat)
    
    def quat_conjugate(self, q):
        q_conj = torch.cat([-q[:, :3], q[:, 3:4]], dim=-1)
        return q_conj
    
    def compute_camera_world_pose(self, hand_pos, hand_rot):
        batch_size = hand_pos.shape[0]
        
        cam_offset_pos = torch.tensor([0.07, 0.03, -0.13], device=hand_pos.device).repeat(batch_size, 1)
        q_cam_in_hand = torch.tensor([0.7071, 0.0, 0.0, 0.7071], device=hand_pos.device).repeat(batch_size, 1)
        
        # cam_offset_pos = torch.tensor([0.07, 0.03, -0.13], device=hand_pos.device).repeat(self.num_envs, 1)
        # q_cam_in_hand = torch.tensor([0.7071, 0.0, 0.0, 0.7071], device=hand_pos.device).repeat(self.num_envs, 1)

        camera_rot_w, camera_pos_w_abs = tf_combine(
            hand_rot,
            hand_pos,
            q_cam_in_hand,
            cam_offset_pos      
        )
        
        camera_pos_w = camera_pos_w_abs - self.scene.env_origins
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
    
    def _initialize_realtime_plots(self):
        plt.ion()
        
        # Figure 1: 3D Trajectory (필요 없다면 최소화하거나 무시)
        # 3D 그래프가 에러를 유발하지 않게 빈 껍데기만 유지하거나, 원하시면 아예 삭제해도 됩니다.
        self.fig1 = plt.figure(figsize=(5, 5))
        self.ax1 = self.fig1.add_subplot(111, projection='3d')
        self.traj_obj_line, = self.ax1.plot([], [], [], label='Object', color='blue')
        self.traj_grip_line, = self.ax1.plot([], [], [], label='Gripper', color='red', linestyle='--')
        self.ax1.set_visible(False) # 3D 그래프 숨기기 (속도 향상)

        # Figure 2: Grasping Gauge (Main)
        self.fig2, self.ax2 = plt.subplots(figsize=(6, 6))
        
        # [핵심] 중앙에 고정된 Grasp Line (빨간색 굵은 선)
        self.grasp_line, = self.ax2.plot([], [], color='red', linewidth=5, label='Grasp Width')
        
        # 중심점 표시 (검은 점)
        self.ax2.scatter([0], [0], color='black', s=50, marker='+')

        # 십자선 (기준선)
        self.ax2.axhline(0, color='gray', linestyle='--', linewidth=0.5)
        self.ax2.axvline(0, color='gray', linestyle='--', linewidth=0.5)
        
        # 텍스트 정보 표시 (좌측 상단)
        self.info_text = self.ax2.text(0.05, 0.95, '', transform=self.ax2.transAxes, 
                                       verticalalignment='top', fontsize=12, 
                                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        self.ax2.set_title('Grasping Angle & Width (Centered)')
        self.ax2.set_xlabel('Width (m)')
        self.ax2.set_ylabel('Width (m)')
        
        # [중요] 축 범위를 고정하여 막대가 잘 보이게 함 (예: +/- 10cm)
        self.ax2.set_xlim(-0.10, 0.10)
        self.ax2.set_ylim(-0.10, 0.10)
        self.ax2.set_aspect('equal', adjustable='box')
        self.ax2.grid(True, linestyle=':', alpha=0.6)
        
        plt.show(block=False)
        
    def _update_realtime_plots(self):
        """계산된 데이터를 시각화 프로세스로 전송 (Non-blocking)"""
        
        # PCA 정보가 없으면 패스
        if not hasattr(self, 'debug_grasp_info') or self.debug_grasp_info is None:
            return

        angle_deg = self.debug_grasp_info["angle"]
        width_m = self.debug_grasp_info["width"]
        
        # [핵심 성능 최적화]
        # 큐가 꽉 차있으면(시각화 프로세스가 바쁘면) 기다리지 않고 이번 프레임은 버림.
        # 이렇게 해야 시뮬레이션 속도가 느려지지 않음.
        try:
            self.plot_queue.put_nowait((angle_deg, width_m))
        except queue.Full:
            pass # 큐가 찼으면 그냥 넘어감 (Drop frame)
    
    def _setup_scene(self):
        self._robot = Articulation(self.cfg.UF_robot)

        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        
        if camera_enable:
            self._camera = Camera(self.cfg.camera)
            self.scene.sensors["hand_camera"] = self._camera
            
        self._contact_sensor = ContactSensor(self.cfg.contact_forces)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        
        self._box = RigidObject(self.cfg.box)
        self.scene.rigid_objects["base_link"] = self._box     
    
    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone().clamp(-1.0, 1.0)
        
        # ======================================================================
        # CASE 1: 학습 모드 (Training Mode)
        # ======================================================================
        if training_mode:
            current_action_scale = self.action_scale_tensor.unsqueeze(-1)
            arm_targets = self.robot_dof_targets[:, :6]
            arm_speed_scales = self.robot_dof_speed_scales[:6]
            new_arm_targets = arm_targets + arm_speed_scales * self.dt * self.actions * current_action_scale
            
            lower_limits = self.robot_dof_lower_limits[:6]
            upper_limits = self.robot_dof_upper_limits[:6]
            self.robot_dof_targets[:, :6] = torch.clamp(new_arm_targets, lower_limits, upper_limits)
            
            if hasattr(self, 'gripper_drive_idx'):
                g_idx = self.gripper_drive_idx
                self.robot_dof_targets[:, g_idx] = self.robot_dof_upper_limits[g_idx]

        # ======================================================================
        # CASE 2: 테스트 모드 (Test Mode)
        # ======================================================================
        else:            
            # [A] 거리 및 벡터 오차 계산
            dist_cam_to_obj = torch.norm(self.box_pos_cam[:, :3], p=2, dim=-1)
            xy_error_cam = torch.norm(self.box_pos_cam[:, :2], p=2, dim=-1)

            forward_local = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1)
            gripper_forward_w = tf_vector(self.robot_grasp_rot, forward_local)
            
            rel_vec = self.box_grasp_pos - self.robot_grasp_pos 
            axial_dist = torch.sum(rel_vec * gripper_forward_w, dim=-1)
            lateral_vec = rel_vec - (axial_dist.unsqueeze(-1) * gripper_forward_w)
            lateral_error = torch.norm(lateral_vec, p=2, dim=-1)

            # [B] 상태 머신 (Phase Transition)
            
            # Phase 0: Tracking
            track_mask = (self.grasp_phase == 0)
            if torch.any(track_mask):
                if hasattr(self, 'debug_grasp_info') and self.debug_grasp_info is not None:
                    pca_angle_deg = self.debug_grasp_info["angle"]
                    pca_width_m = self.debug_grasp_info["width"]
                    current_j6 = self._robot.data.joint_pos[track_mask, 5]
                    
                    self.target_grasp_angle[track_mask] = current_j6 + math.radians(pca_angle_deg)
                    self.target_grasp_width[track_mask] = max(0.0, pca_width_m - 0.01)
            
            # Track -> Approach
            is_stable = (dist_cam_to_obj < 0.50) & (dist_cam_to_obj > 0.30) & (xy_error_cam < 0.10) & approach_mode
            self.stable_timer = torch.where(track_mask & is_stable, self.stable_timer + self.dt, torch.zeros_like(self.stable_timer))
            self.grasp_phase[self.stable_timer > 0.7] = 1
            
            # Approach -> Grasping
            is_approaching = (self.grasp_phase == 1)
            ready_to_grasp = is_approaching & (lateral_error < 0.04) & (axial_dist < 0.16) & grasp_mode
            self.grasp_phase[ready_to_grasp] = 2

            # [C] 로봇 제어 (Arm + Gripper)
            
            # 1. Arm 이동
            current_action_scale = self.action_scale_tensor.unsqueeze(-1)
            arm_targets = self.robot_dof_targets[:, :6]
            arm_speed_scales = self.robot_dof_speed_scales[:6]
            new_arm_targets = arm_targets + arm_speed_scales * self.dt * self.actions * current_action_scale
            
            lower_limits = self.robot_dof_lower_limits[:6]
            upper_limits = self.robot_dof_upper_limits[:6]
            self.robot_dof_targets[:, :6] = torch.clamp(new_arm_targets, lower_limits, upper_limits)

            # [Arm Freeze] Phase 2(Grasping)에서는 팔 위치 고정
            grasp_mask = (self.grasp_phase >= 2)
            if torch.any(grasp_mask):
                # 현재 관절 각도를 목표값으로 덮어씌워 제자리에 멈추게 함
                current_joints = self._robot.data.joint_pos[grasp_mask, :6]
                self.robot_dof_targets[grasp_mask, :6] = current_joints

            # 2. 손목 회전 제어 (Phase 1 이상)
            align_mask = (self.grasp_phase < 2) 
            if torch.any(align_mask):
                self.robot_dof_targets[align_mask, 6] = self.target_grasp_angle[align_mask]

            # 3. 그리퍼 너비 제어
            if hasattr(self, 'gripper_drive_idx'):
                g_idx = self.gripper_drive_idx
                open_mask = (self.grasp_phase < 2)
                if torch.any(open_mask):
                    self.robot_dof_targets[open_mask, g_idx] = 0.0
                if torch.any(grasp_mask):
                    print("self.target_grasp_width[grasp_mask] :", self.target_grasp_width[grasp_mask])
                    self.robot_dof_targets[grasp_mask, g_idx] = self.target_grasp_width[grasp_mask]

            # ==================================================================
            # [D] 실시간 상태 모니터링 출력
            # ==================================================================
            if self.log_counter % 20 == 0:
                env_idx = 0
                phase = self.grasp_phase[env_idx].item()
                
                if phase == 0:
                    s_flag = "✅ STABLE" if is_stable[env_idx].item() else "❌ UNSTABLE"
                    cur_timer = self.stable_timer[env_idx].item()
                    cur_cam_dist = dist_cam_to_obj[env_idx].item()
                    cur_err = xy_error_cam[env_idx].item()
                    print(f"[Phase 0: Track] {s_flag} | Time: {cur_timer:.2f}/0.7s | Cam Dist: {cur_cam_dist:.3f}m | Err: {cur_err:.3f}m")
                    
                elif phase == 1:
                    lat_err = lateral_error[env_idx].item()
                    ax_dist = axial_dist[env_idx].item()
                    latched_ang = math.degrees(self.target_grasp_angle[env_idx].item())
                    print(f"[Phase 1: Appr ] 🚀 GOING! | Lat: {lat_err:.3f}m | Axial: {ax_dist:.3f}m (Target < 0.16) | Ang: {latched_ang:.1f}°")
                    
                elif phase >= 2:
                    latched_ang = math.degrees(self.target_grasp_angle[env_idx].item())
                    curr_width = self._robot.data.joint_pos[env_idx, self.gripper_drive_idx].item()
                    target_w = self.target_grasp_width[env_idx].item()
                    # 0.0이 아니라 PCA 타겟 너비로 출력
                    print(f"[Phase 2: Grasp] ✊ HOLDING (Freezed) | Angle: {latched_ang:.1f}° | Width: {curr_width*100:.1f}cm (Target: {target_w*100:.1f})")

        # ======================================================================
        # [E] 공통 업데이트
        # ======================================================================
        self.cfg.current_time += self.dt
        if image_publish:   
            self.last_publish_time += self.dt
            if self.last_publish_time >= (1.0 / 15.0):
                self.publish_camera_data()
                rclpy.spin_once(self.node, timeout_sec=0.001)
                self.last_publish_time = 0.0

        # [F] 물체 이동 로직 (기존 코드 유지)
        static_mask = (self.object_move_state == self.MOVE_STATE_STATIC)
        static_env_ids = torch.where(static_mask)[0]
        if len(static_env_ids) > 0:
            pos = self.current_box_pos[static_env_ids]; rot = self.current_box_rot[static_env_ids]
            self._box.write_root_pose_to_sim(torch.cat([pos, rot], dim=-1), env_ids=static_env_ids)
            self._box.write_root_velocity_to_sim(torch.zeros((len(static_env_ids), 6), device=self.device), env_ids=static_env_ids)
        
        linear_move_mask = (self.object_move_state == self.MOVE_STATE_LINEAR)
        linear_env_ids = torch.where(linear_move_mask)[0]
        if len(linear_env_ids) > 0:
            current_pos_world = self._box.data.root_pos_w[linear_env_ids]
            target_pos = self.target_box_pos[linear_env_ids]
            distance_to_target = torch.norm(target_pos - current_pos_world, p=2, dim=-1)
            reached_target_mask = (distance_to_target < 0.01)

            if torch.any(reached_target_mask):
                env_ids_to_update = linear_env_ids[reached_target_mask]
                num_to_update = len(env_ids_to_update)
                current_levels = self.current_reward_level[env_ids_to_update]
                
                rand_pos_new = torch.rand(num_to_update, 3, device=self.device)
                for i, axis in enumerate(['x', 'y', 'z']):
                    rand_pos_new[:, i] = rand_pos_new[:, i] * (rand_pos_range[axis][1] - rand_pos_range[axis][0]) + rand_pos_range[axis][0]
                
                curr_target_world = self.target_box_pos[env_ids_to_update]
                curr_target_local = curr_target_world - self.scene.env_origins[env_ids_to_update]
                final_target = rand_pos_new.clone()

                mask_lv1 = (current_levels == 1); mask_lv2 = (current_levels == 2)
                
                if torch.any(mask_lv1): # Level 1 Logic
                    ids_l1 = env_ids_to_update[mask_lv1]
                    axes = self.level1_axis_mode[ids_l1]
                    
                    for ax_idx in range(3):
                        cond = (axes == ax_idx)
                        cur_val = curr_target_local[mask_lv1, ax_idx] 
                        min_v, max_v = list(rand_pos_range.values())[ax_idx]
                        
                        next_val = torch.where(
                            torch.abs(cur_val - max_v) < torch.abs(cur_val - min_v), 
                            torch.tensor(min_v, device=self.device), 
                            torch.tensor(max_v, device=self.device)
                        )
                        
                        final_target[mask_lv1, ax_idx] = torch.where(cond, next_val, cur_val)
                        
                        for other_ax in range(3):
                            if other_ax != ax_idx: 
                                cur_val_other = curr_target_local[mask_lv1, other_ax]
                                final_target[mask_lv1, other_ax] = torch.where(cond, cur_val_other, final_target[mask_lv1, other_ax])

                if torch.any(mask_lv2): # Level 2 Logic
                    ids_l2 = env_ids_to_update[mask_lv2]
                    planes = self.level2_plane_mode[ids_l2]
                    dim_map = {0:2, 1:1, 2:0}
                    
                    for p_idx, dim in dim_map.items():
                        cond = (planes == p_idx)
                        cur_val = curr_target_local[mask_lv2, dim]
                        final_target[mask_lv2, dim] = torch.where(cond, cur_val, final_target[mask_lv2, dim])

                self.target_box_pos[env_ids_to_update] = final_target + self.scene.env_origins[env_ids_to_update]

            self.speed_change_timer[linear_env_ids] -= self.dt
            change_speed_mask = (self.speed_change_timer <= 0.0) & linear_move_mask
            ids_change = torch.where(change_speed_mask)[0]
            if len(ids_change) > 0:
                self.current_speed_factor[ids_change] = (torch.rand(len(ids_change), device=self.device) * 1.0) + 0.5
                self.speed_change_timer[ids_change] = (torch.rand(len(ids_change), device=self.device) * 1.0) + 0.5
            
            target_updated = self.target_box_pos[linear_env_ids]
            direction = target_updated - current_pos_world
            unit_dir = direction / (torch.norm(direction, p=2, dim=-1, keepdim=True) + 1e-6)
            vel = unit_dir * self.obj_speed[linear_env_ids].unsqueeze(-1) * self.current_speed_factor[linear_env_ids].unsqueeze(-1)
            
            vel_cmd = torch.zeros((len(linear_env_ids), 6), device=self.device)
            vel_cmd[:, 0:3] = vel
            self._box.write_root_velocity_to_sim(vel_cmd, env_ids=linear_env_ids)
    
    def _apply_action(self):
        global robot_action
        global robot_init_pose
        
        target_pos = self.robot_dof_targets.clone()
    
        joint4_index = self._robot.find_joints(["joint4"])[0]
        joint6_index = self._robot.find_joints(["joint6"])[0]
        gripper_joint_idex = self._robot.find_joints(["drive_joint"])[0]
        target_pos[:, joint4_index] = 0.0
        
        if training_mode == False and approach_mode == True:
            non_grasping_mask = (self.grasp_phase == 0)
            if torch.any(non_grasping_mask):
                target_pos[non_grasping_mask, joint6_index] = 0.0
                target_pos[non_grasping_mask, gripper_joint_idex] =0.0
        else:
            target_pos[:, joint6_index] = 0.0
            target_pos[:, gripper_joint_idex] = 0.0
        
        if training_mode == False and robot_fix == False:
            if robot_action and robot_init_pose:
                self._robot.set_joint_position_target(target_pos)
            
            elif robot_action == False and robot_init_pose == False:
                init_pos = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

                for name, val in zip(self.joint_names, self.joint_init_values):
                    index = self._robot.find_joints(name)[0]
                    init_pos[:, index] = val

                self._robot.set_joint_position_target(init_pos)

                joint_err = torch.abs(self._robot.data.joint_pos - init_pos)
                max_err = torch.max(joint_err).item()
                            
                if max_err < 2.0:
                    self.init_cnt += 1
                    print(f"init_cnt : {self.init_cnt}")
                    if self.init_cnt > 50:
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
                                
                if max_err < 2.0:
                    robot_init_pose = True
                    robot_action = True
                
            elif robot_init_pose:
                self._robot.set_joint_position_target(target_pos)
        
        if training_mode == False:
            rgb_data = self._camera.data.output["rgb"]
            depth_data = self._camera.data.output["depth"]

            # 1. 결과를 하나의 변수로 받습니다.
            result = self.calculate_pca_grasping(rgb_data, depth_data)

            # 2. 결과가 유효한지(None이 아닌지) 확인합니다.
            if result is not None:
                gripper_angle, gripper_width, center = result
                if gripper_angle is not None:
                    self.debug_grasp_info = {
                        "angle": gripper_angle,
                        "width": gripper_width
                    }

    def calculate_pca_grasping(self, rgb_image, depth_image, env_id=0):
        try:
            # --- 1. 데이터 전처리 ---
            if isinstance(rgb_image, torch.Tensor): rgb_image = rgb_image.cpu().numpy()
            if isinstance(depth_image, torch.Tensor): depth_image = depth_image.cpu().numpy()

            if rgb_image.ndim == 4: rgb_image = rgb_image[env_id]
            
            # Depth 차원 정리
            if depth_image.ndim == 4: depth_image = depth_image[env_id]
            elif depth_image.ndim == 3 and depth_image.shape[-1] != 1: depth_image = depth_image[env_id]
            if depth_image.ndim == 3 and depth_image.shape[-1] == 1: depth_image = depth_image.squeeze(-1)

            # 포맷 변환
            if rgb_image.shape[-1] == 4: rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGBA2RGB)
            if rgb_image.dtype != np.uint8:
                rgb_image = (rgb_image * 255).astype(np.uint8) if rgb_image.max() <= 1.1 else rgb_image.astype(np.uint8)

            # --- 2. 물체 마스크 추출 ---
            hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
            
            # [색상 범위] (물체에 맞춰 조정 필요)
            # 캔(Potted Meat)의 경우 붉은색 계열이므로 두 범위 합쳐야 할 수 있음
            # 머스타드(노란색) 예시:
            lower_color = np.array([20, 100, 100]); upper_color = np.array([35, 255, 255])
            mask = cv2.inRange(hsv, lower_color, upper_color)

            # [핵심 1] 구멍 메우기 (Morphology Close)
            # 물체 내부의 Depth 결측으로 인해 구멍이 뚫리면 PCA가 모서리를 잡습니다.
            # 커널 크기를 키워서(5x5 -> 7x7) 확실하게 메워줍니다.
            kernel = np.ones((7, 7), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) 
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) # 노이즈 제거

            # --- 3. 윤곽선 검출 ---
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours: return None
            
            max_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(max_contour) < 100: return None

            # --- 4. PCA (각도 계산용) ---
            pts = max_contour.reshape(-1, 2).astype(np.float64)
            mean, eigenvectors, _ = cv2.PCACompute2(pts, mean=np.array([]))
            cx, cy = int(mean[0][0]), int(mean[0][1])
            center = (cx, cy)

            # 각도는 PCA가 안정적임
            major_axis = eigenvectors[0]
            angle_rad = np.arctan2(major_axis[1], major_axis[0])
            grasp_angle_deg = np.degrees(angle_rad)

            # --- 5. MinAreaRect (너비 계산용) ---
            # [핵심 2] PCA 분산 대신 '최소 외접 직사각형' 사용
            rect = cv2.minAreaRect(max_contour)
            (rect_center, (w, h), rect_angle) = rect
            
            # 물체의 '짧은 변'이 우리가 잡아야 할 너비입니다.
            pixel_width = min(w, h) 

            # --- 6. 실제 거리(Depth) 계산 ---
            valid_depths = depth_image[(mask > 0) & (depth_image > 0)]
            
            if len(valid_depths) > 10:
                d_val = np.median(valid_depths)
            else:
                # 마스크 내 유효값이 없으면 중심점 사용 (예외 처리)
                if 0 <= cy < depth_image.shape[0] and 0 <= cx < depth_image.shape[1]:
                    d_val = depth_image[cy, cx]
                else:
                    d_val = 0.0

            if d_val <= 0 or np.isnan(d_val): d_val = 0.4 # 기본값

            # --- 7. 픽셀 -> 미터 변환 ---
            intrinsics = self._camera.data.intrinsic_matrices[env_id]
            fx = intrinsics[0, 0].item()
            
            real_width_m = d_val * (pixel_width / fx)

            return grasp_angle_deg, real_width_m, center

        except Exception as e:
            return None
            
    # post-physics step calls
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        if hasattr(self, 'is_pview_fail'):
            is_high_level = (self.current_reward_level >= 7)
            current_fail = self.is_pview_fail
            
            self.out_of_fov_counter = torch.where(
                is_high_level & current_fail,
                self.out_of_fov_counter + 1,
                torch.zeros_like(self.out_of_fov_counter)
            )
        
            MAX_CONSECUTIVE_FAIL_STEPS = 45
            terminated = (self.out_of_fov_counter >= MAX_CONSECUTIVE_FAIL_STEPS)
    
        else:
            terminated = torch.zeros_like(self.episode_length_buf, dtype=torch.bool)
        
        truncated = self.episode_length_buf >= self.max_episode_length + add_episode_length

        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        self._compute_intermediate_values()
    
        camera_pos_w, camera_rot_w = self.compute_camera_world_pose(self.hand_pos, self.hand_rot)

        self.box_pos_cam, box_rot_cam = self.world_to_camera_pose(
            camera_pos_w, camera_rot_w,
            self.box_grasp_pos - self.scene.env_origins, self.box_grasp_rot,
        )
        
        # gripper_to_box_dist = torch.norm(self.robot_grasp_pos - self.box_grasp_pos, p=2, dim=-1)

        if not training_mode and test_graph_mode: 
            gripper_pos = self.robot_grasp_pos[0].cpu().numpy()
            object_pos = self.box_grasp_pos[0].cpu().numpy()
            
            cam_pos = np.zeros(2) 
            cam_pos[0] = self.box_pos_cam[0,0].cpu().numpy() #x축
            cam_pos[1] = self.box_pos_cam[0,1].cpu().numpy() #y축
            
            camera_real_dist = torch.norm(self.box_pos_cam[0], p=2, dim=-1).item()
            distance_val = camera_real_dist

            # with open(self.csv_filepath, 'a', newline='') as f:
            #     writer = csv.writer(f)
            #     writer.writerow([gripper_pos[0], gripper_pos[1], gripper_pos[2],
            #                      object_pos[0], object_pos[1], object_pos[2],
            #                      cam_pos[0], cam_pos[1],
            #                     distance_val])
            #     f.flush() 
            #     os.fsync(f.fileno()) 
            
            if hasattr(self, 'debug_grasp_info') and self.debug_grasp_info is not None:
                self.last_grasp_angle = self.debug_grasp_info["angle"]
                self.last_grasp_width = self.debug_grasp_info["width"]
            
            # CSV에는 항상 self.last_... 값을 기록
            save_angle = self.last_grasp_angle
            save_width = self.last_grasp_width

            with open(self.csv_filepath, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([gripper_pos[0], gripper_pos[1], gripper_pos[2],
                                 object_pos[0], object_pos[1], object_pos[2],
                                 cam_pos[0], cam_pos[1],
                                 distance_val,
                                 save_angle, save_width]) # 저장
                f.flush() 
                os.fsync(f.fileno())
            
        
        # --- 하드 종료 조건 계산 및 저장 ---
        levels = self.current_reward_level
        fail_margin = torch.tensor([reward_curriculum_levels[l.item()]["fail_margin"] for l in levels], device=self.device)
        
        depth_val = torch.abs(self.box_pos_cam[:, 2]) + 1e-6
        physical_offset = torch.norm(self.box_pos_cam[:, [0, 1]], dim=-1)
        
        view_ratio = physical_offset / depth_val
        out_of_fov_mask = view_ratio > fail_margin
        is_behind_mask = self.box_pos_cam[:, 2] <= 0 

        self.is_pview_fail = out_of_fov_mask | is_behind_mask
        self.is_object_visible_mask = ~self.is_pview_fail
        
        is_tracking_success = ~self.is_pview_fail
        self.success_steps_buf += is_tracking_success.float()

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

    def _perform_static_reset(self, env_ids: torch.Tensor):        
        num_resets = len(env_ids)
        if num_resets == 0:
            return
            
        final_weights = []
        for key in zone_keys:
            if not zone_activation.get(key, False):
                final_weights.append(0.0)
                continue
            z_part, x_part = key.split('_')
            combined_weight = x_weights.get(x_part, 1.0) * z_weights.get(z_part, 1.0)
            final_weights.append(combined_weight)
        
        weights_tensor = torch.tensor(final_weights, dtype=torch.float, device=self.device)
        selected_zone_indices = torch.multinomial(weights_tensor, num_resets, replacement=True)

        x_mins = torch.tensor([zone_definitions[zone_keys[i]]["x"][0] for i in selected_zone_indices], device=self.device)
        x_maxs = torch.tensor([zone_definitions[zone_keys[i]]["x"][1] for i in selected_zone_indices], device=self.device)
        z_mins = torch.tensor([zone_definitions[zone_keys[i]]["z"][0] for i in selected_zone_indices], device=self.device)
        z_maxs = torch.tensor([zone_definitions[zone_keys[i]]["z"][1] for i in selected_zone_indices], device=self.device)

        x_pos = torch.rand(num_resets, device=self.device) * (x_maxs - x_mins) + x_mins
        z_pos = torch.rand(num_resets, device=self.device) * (z_maxs - z_mins) + z_mins

        current_levels = self.current_reward_level[env_ids]
        y_pos = torch.zeros(num_resets, device=self.device)
        
        for level_idx in range(self.max_reward_level + 1):
            level_mask = (current_levels == level_idx)
            num_in_level = torch.sum(level_mask)
            
            if num_in_level > 0:
                y_range = reward_curriculum_levels[level_idx]["y_range"]
                y_pos[level_mask] = torch.rand(num_in_level, device=self.device) * (y_range[1] - y_range[0]) + y_range[0]

        self.rand_pos[env_ids] = torch.stack([x_pos, y_pos, z_pos], dim=1)
        rand_reset_pos = self.rand_pos[env_ids] + self.scene.env_origins[env_ids]
        
        random_angles = torch.rand(num_resets, device=self.device) * 2 * torch.pi
        rand_reset_rot = torch.stack([
            torch.cos(random_angles / 2),
            torch.zeros(num_resets, device=self.device),
            torch.zeros(num_resets, device=self.device),
            torch.sin(random_angles / 2)  
        ], dim=1)
        
        # ----------------------------------------------------------------------
        # [추가] 고정된 위치와 회전을 클래스 변수에 저장 (나중에 강제 고정용)
        # ----------------------------------------------------------------------
        self.current_box_pos[env_ids] = rand_reset_pos
        self.current_box_rot[env_ids] = rand_reset_rot
        # ----------------------------------------------------------------------

        rand_reset_box_pose = torch.cat([rand_reset_pos, rand_reset_rot], dim=-1)
        zero_root_velocity = torch.zeros((self.num_envs, 6), device=self.device)
        
        self._box.write_root_pose_to_sim(rand_reset_box_pose, env_ids=env_ids)
        self._box.write_root_velocity_to_sim(zero_root_velocity[env_ids], env_ids=env_ids)
        
        if training_mode == True:
            joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
            joint1_idx = self._robot.find_joints(["joint1"])[0]
            
            YAW_CANDIDATE_ANGLES = { 15.0: math.radians(15.0), 45.0: math.radians(45.0), 75.0: math.radians(75.0) }
            ANGLE_BOUNDARIES = [30.0, 60.0, 90.0]
            
            for i, env_id in enumerate(env_ids):
                object_pos_local = rand_reset_pos[i] - self.scene.env_origins[env_id]
                obj_x, obj_y, obj_z = object_pos_local[0], object_pos_local[1], object_pos_local[2]
                        
                if obj_x >= workspace_zones["x"]["far"]: x_zone = "far"
                elif obj_x >= workspace_zones["x"]["middle"]: x_zone = "middle"
                else: x_zone = "close"
                    
                if obj_z >= workspace_zones["z"]["top"]: z_zone = "top"
                elif obj_z >= workspace_zones["z"]["bottom"]: z_zone = "middle"
                else: z_zone = "bottom"
                    
                zone_key = f"{z_zone}_{x_zone}"
                target_pose_dict = pose_candidate[zone_key]
                
                for joint_name, pos in target_pose_dict.items():
                    if joint_name != "joint1":
                        joint_idx = self._robot.find_joints(joint_name)[0]
                        joint_pos[i, joint_idx] = pos
                        
                target_yaw_rad = torch.atan2(obj_y, obj_x)
                abs_yaw_deg = torch.abs(torch.rad2deg(target_yaw_rad))

                if abs_yaw_deg <= ANGLE_BOUNDARIES[0]: target_angle_deg = 15.0
                elif abs_yaw_deg <= ANGLE_BOUNDARIES[1]: target_angle_deg = 45.0
                else: target_angle_deg = 75.0

                final_yaw_rad = YAW_CANDIDATE_ANGLES[target_angle_deg] * torch.sign(obj_y)
                joint_pos[i, joint1_idx] = final_yaw_rad
                
            joint_pos[:, joint1_idx] = torch.clamp(joint_pos[:, joint1_idx], self.robot_dof_lower_limits[joint1_idx], self.robot_dof_upper_limits[joint1_idx])
            joint_vel = torch.zeros_like(joint_pos)
            
            self.robot_dof_targets[env_ids] = joint_pos
            
            self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
            self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
            
            self.episode_init_joint_pos[env_ids] = joint_pos
            
    def _perform_linear_reset(self, env_ids: torch.Tensor):
        if not training_mode:
            new_seed = int(time.time() * 1000) % (2**32 - 1)
            torch.manual_seed(new_seed)
        
        num_resets = len(env_ids)
        if num_resets == 0:
            return
        
        # 1. 현재 레벨 확인
        current_levels = self.current_reward_level[env_ids]
        mask_lv1 = (current_levels == 1)
        mask_lv2 = (current_levels == 2)
                
        # [공통] 새로운 시작 위치(Start Position) 생성 (완전 랜덤)
        # X, Y, Z 모두 전체 범위 내에서 균등 랜덤 생성 (가중치 제거됨)
        start_x = torch.rand(num_resets, device=self.device) * (rand_pos_range["x"][1] - rand_pos_range["x"][0]) + rand_pos_range["x"][0]
        start_y = torch.rand(num_resets, device=self.device) * (rand_pos_range["y"][1] - rand_pos_range["y"][0]) + rand_pos_range["y"][0]
        start_z = torch.rand(num_resets, device=self.device) * (rand_pos_range["z"][1] - rand_pos_range["z"][0]) + rand_pos_range["z"][0]
        
        start_pos = torch.stack([start_x, start_y, start_z], dim=1)
        
        # [중요] 물체 강제 이동 (Teleport) - 시뮬레이션 상의 위치를 즉시 변경
        random_angles = torch.rand(num_resets, device=self.device) * 2 * torch.pi
        reset_rot = torch.stack([
            torch.cos(random_angles / 2),
            torch.zeros(num_resets, device=self.device),
            torch.zeros(num_resets, device=self.device),
            torch.sin(random_angles / 2)  
        ], dim=1)
        
        reset_pose = torch.cat([start_pos + self.scene.env_origins[env_ids], reset_rot], dim=-1)
        zero_velocity = torch.zeros((num_resets, 6), device=self.device)
        
        self._box.write_root_pose_to_sim(reset_pose, env_ids=env_ids)
        self._box.write_root_velocity_to_sim(zero_velocity, env_ids=env_ids)

        start_pos_world = start_pos + self.scene.env_origins[env_ids]
        
        # 내부 상태 변수 초기화 (현재 위치 = 방금 이동시킨 랜덤 위치)
        self.new_box_pos_rand[env_ids] = start_pos_world
        self.current_box_pos[env_ids] = start_pos_world
        self.current_box_rot[env_ids] = reset_rot

        target_pos = start_pos.clone()
        
        # --- Level 1: 1차원 왕복 운동 초기화 ---
        if torch.any(mask_lv1):
            ids_lv1 = env_ids[mask_lv1]
            num_lv1 = len(ids_lv1)
            
            # 1. 축 랜덤 선택 (0:X, 1:Y, 2:Z) 및 저장 (새로운 에피소드용 축)
            axis_mode = torch.randint(0, 3, (num_lv1,), device=self.device)
            self.level1_axis_mode[ids_lv1] = axis_mode
            
            # 2. 범위 가져오기
            x_min, x_max = rand_pos_range["x"]
            y_min, y_max = rand_pos_range["y"]
            z_min, z_max = rand_pos_range["z"]
            
            # 3. 첫 번째 목표 설정: "현재 위치에서 가장 먼 끝점"으로 설정
            # (그래야 시작하자마자 긴 거리를 이동하며 왕복을 시작함)
            
            # X축 왕복 (Y, Z는 시작 위치 고정)
            cond_x = (axis_mode == 0)
            # 현재 x가 중간보다 작으면 Max로, 크면 Min으로
            target_x_dest = torch.where(start_x[mask_lv1] < (x_min + x_max)/2, torch.tensor(x_max, device=self.device), torch.tensor(x_min, device=self.device))
            target_pos[mask_lv1, 0] = torch.where(cond_x, target_x_dest, target_pos[mask_lv1, 0])
            
            # Y축 왕복 (X, Z는 시작 위치 고정)
            cond_y = (axis_mode == 1)
            target_y_dest = torch.where(start_y[mask_lv1] < (y_min + y_max)/2, torch.tensor(y_max, device=self.device), torch.tensor(y_min, device=self.device))
            target_pos[mask_lv1, 1] = torch.where(cond_y, target_y_dest, target_pos[mask_lv1, 1])
            
            # Z축 왕복 (X, Y는 시작 위치 고정)
            cond_z = (axis_mode == 2)
            target_z_dest = torch.where(start_z[mask_lv1] < (z_min + z_max)/2, torch.tensor(z_max, device=self.device), torch.tensor(z_min, device=self.device))
            target_pos[mask_lv1, 2] = torch.where(cond_z, target_z_dest, target_pos[mask_lv1, 2])

        # --- Level 2+: 3차원 연속 이동 초기화 ---
        if torch.any(mask_lv2):
            ids_lv2 = env_ids[mask_lv2]
            num_lv2 = len(ids_lv2)
            
            # 1. 평면 랜덤 선택 및 저장 (이번 에피소드 동안 고정)
            # 0:XY(Z고정), 1:XZ(Y고정), 2:YZ(X고정)
            plane_mode = torch.randint(0, 3, (num_lv2,), device=self.device)
            self.level2_plane_mode[ids_lv2] = plane_mode
            
            # 2. 시작 위치(Start)는 3D 공간상 완전 랜덤 (순간이동용)
            # (이미 위 공통 로직에서 start_pos가 계산되어 있음)
            
            # 3. 첫 번째 목표(Target) 설정
            # "선택된 평면 위"에 있어야 하므로, 고정축은 start_pos 값을 유지하고 나머지만 랜덤
            
            # 랜덤 후보군 생성
            rand_tx = torch.rand(num_lv2, device=self.device) * (rand_pos_range["x"][1] - rand_pos_range["x"][0]) + rand_pos_range["x"][0]
            rand_ty = torch.rand(num_lv2, device=self.device) * (rand_pos_range["y"][1] - rand_pos_range["y"][0]) + rand_pos_range["y"][0]
            rand_tz = torch.rand(num_lv2, device=self.device) * (rand_pos_range["z"][1] - rand_pos_range["z"][0]) + rand_pos_range["z"][0]
            
            # start_pos는 공통 로직에서 계산된 값 사용
            # 해당 환경(ids_lv2)에 맞는 start_pos 추출
            # start_pos 텐서는 (num_resets, 3) 크기이므로 마스크를 사용해 추출해야 함
            # 주의: start_pos는 전체 env_ids에 대한 것이므로, mask_lv2를 그대로 쓰면 됨
            
            start_x_lv2 = start_pos[mask_lv2, 0]
            start_y_lv2 = start_pos[mask_lv2, 1]
            start_z_lv2 = start_pos[mask_lv2, 2]
            
            # 평면 구속 조건 적용
            # XY 평면 (Z 고정)
            cond_xy = (plane_mode == 0)
            final_tz = torch.where(cond_xy, start_z_lv2, rand_tz)
            
            # XZ 평면 (Y 고정)
            cond_xz = (plane_mode == 1)
            final_ty = torch.where(cond_xz, start_y_lv2, rand_ty)
            
            # YZ 평면 (X 고정)
            cond_yz = (plane_mode == 2)
            final_tx = torch.where(cond_yz, start_x_lv2, rand_tx)
            
            # 나머지 축은 랜덤값 그대로 사용 (조건이 False인 곳은 rand값이 들어감)
            final_tx = torch.where(cond_yz, final_tx, rand_tx)
            final_ty = torch.where(cond_xz, final_ty, rand_ty)
            final_tz = torch.where(cond_xy, final_tz, rand_tz)

            # 최종 목표 할당
            target_pos[mask_lv2, 0] = final_tx
            target_pos[mask_lv2, 1] = final_ty
            target_pos[mask_lv2, 2] = final_tz
        
        # [공통] 최종 목표 적용 및 이동 벡터 계산
        self.target_box_pos[env_ids] = target_pos + self.scene.env_origins[env_ids]

        # direction = self.target_box_pos[env_ids] - self.new_box_pos_rand[env_ids]
        # direction_norm = torch.norm(direction, p=2, dim=-1, keepdim=True) + 1e-6
        # speed = self.obj_speed[env_ids].unsqueeze(-1)
        
        # self.rand_pos_step[env_ids] = (direction / direction_norm * speed)

        # [로봇 자세 초기화] (기존 코드 유지)
        if training_mode:
            joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
            joint1_idx = self._robot.find_joints(["joint1"])[0]
            
            YAW_CANDIDATE_ANGLES = { 15.0: math.radians(15.0), 45.0: math.radians(45.0), 75.0: math.radians(75.0) }
            ANGLE_BOUNDARIES = [30.0, 60.0, 90.0]
            
            for i, env_id in enumerate(env_ids):
                object_pos_local = start_pos[i] # 위에서 만든 랜덤 위치 사용
                obj_x, obj_y, obj_z = object_pos_local[0], object_pos_local[1], object_pos_local[2]
                        
                if obj_x >= workspace_zones["x"]["far"]: x_zone = "far"
                elif obj_x >= workspace_zones["x"]["middle"]: x_zone = "middle"
                else: x_zone = "close"
                    
                if obj_z >= workspace_zones["z"]["top"]: z_zone = "top"
                elif obj_z >= workspace_zones["z"]["bottom"]: z_zone = "middle"
                else: z_zone = "bottom"
                    
                zone_key = f"{z_zone}_{x_zone}"
                target_pose_dict = pose_candidate[zone_key]
                
                for joint_name, pos in target_pose_dict.items():
                    if joint_name != "joint1":
                        joint_idx = self._robot.find_joints(joint_name)[0]
                        joint_pos[i, joint_idx] = pos
                        
                target_yaw_rad = torch.atan2(obj_y, obj_x)
                abs_yaw_deg = torch.abs(torch.rad2deg(target_yaw_rad))

                if abs_yaw_deg <= ANGLE_BOUNDARIES[0]: target_angle_deg = 15.0
                elif abs_yaw_deg <= ANGLE_BOUNDARIES[1]: target_angle_deg = 45.0
                else: target_angle_deg = 75.0

                final_yaw_rad = YAW_CANDIDATE_ANGLES[target_angle_deg] * torch.sign(obj_y)
                joint_pos[i, joint1_idx] = final_yaw_rad
                
            joint_pos[:, joint1_idx] = torch.clamp(joint_pos[:, joint1_idx], self.robot_dof_lower_limits[joint1_idx], self.robot_dof_upper_limits[joint1_idx])
            joint_vel = torch.zeros_like(joint_pos)
            self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
            self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
            
            # [핵심] 로봇 목표값 동기화 (튀는 현상 방지)
            self.robot_dof_targets[env_ids] = joint_pos 
            self.episode_init_joint_pos[env_ids] = joint_pos
    
    def _reset_idx(self, env_ids: torch.Tensor | None):
        global reset_flag 
        if reset_flag:
            if training_mode == False:
                reset_flag = False
            
            # [신규 로직] 복합 조건 평가
            actual_lengths = self.episode_length_buf[env_ids].float() + 1e-6

            # 1. 성공률 (Tracking Ratio) 계산
            # (시야 유지 스텝 / 전체 에피소드 길이)
            success_ratio = self.success_steps_buf[env_ids] / actual_lengths
            pass_ratio = success_ratio >= self.MIN_PVIEW_RATIO

            # 2. 평균 거리 오차 계산
            avg_distance = self.avg_distance_error_buf[env_ids] / actual_lengths
            pass_distance = avg_distance <= self.MAX_DISTANCE_ERROR

            # 3. 총 보상 계산
            total_reward = self.episode_reward_buf[env_ids]
            pass_reward = total_reward >= self.MIN_TOTAL_REWARD

            # [최종 성공 판단] 3가지 조건을 모두(AND) 만족해야 성공
            success_mask_reward = pass_ratio & pass_distance #& pass_reward
            failure_mask_reward = ~success_mask_reward

            #승률 기반 레벨 이동 로직
            # 1. 성공/실패 각각 카운트 누적 (초기화 없음)
            self.success_count[env_ids] += success_mask_reward.long()
            self.failure_count[env_ids] += failure_mask_reward.long()

            # 2. 총 시도 횟수 계산
            total_attempts = self.success_count[env_ids] + self.failure_count[env_ids]

            # 3. 평가 주기(EVAL_BATCH_SIZE)가 된 환경들만 골라내기
            # 예: 20판을 채운 환경들
            check_mask = (total_attempts >= self.EVAL_BATCH_SIZE)

            if torch.any(check_mask):
                check_env_ids = env_ids[check_mask]

                # 승률 계산 (성공 / 전체)
                current_success_rate = self.success_count[check_env_ids].float() / total_attempts[check_env_ids].float()

                # A. 승급 심사 (90% 이상)
                promote_mask = current_success_rate >= self.PROMOTION_RATE
                if torch.any(promote_mask):
                    promote_ids = check_env_ids[promote_mask]
                    new_levels = self.current_reward_level[promote_ids] + 1
                    
                    # [핵심] 최대 레벨을 초과하지 않도록 clamp
                    new_levels = torch.clamp(new_levels, max=self.max_reward_level)
                    self.current_reward_level[promote_ids] = (self.current_reward_level[promote_ids] + 1).clamp(max=self.max_reward_level)

                # B. 강등 심사 (40% 이하)
                demote_mask = current_success_rate < self.DEMOTION_RATE
                if torch.any(demote_mask):
                    demote_ids = check_env_ids[demote_mask]
                    
                    # [수정] 현재 레벨 - 1 계산
                    new_levels = self.current_reward_level[demote_ids] - 1
                    
                    # [핵심] 0 미만으로 내려가지 않도록 clamp
                    new_levels = torch.clamp(new_levels, min=0)
                    
                    # 레벨다운
                    self.current_reward_level[demote_ids] = (self.current_reward_level[demote_ids] - 1).clamp(min=0)

                # [중요] 평가가 끝난 환경들은 카운터 리셋 (다음 20판을 위해)
                # 20판 채우면 무조건 리셋해야, 과거의 성적에 발목 잡히지 않음
                self.success_count[check_env_ids] = 0
                self.failure_count[check_env_ids] = 0

            self.log_counter += 5
            if 1 :#self.log_counter % self.LOG_INTERVAL == 0:
                level_counts = torch.bincount(self.current_reward_level, minlength=self.max_reward_level + 1)

                # [기존] 1. 평균 승률 계산
                total_attempts = self.success_count + self.failure_count
                valid_mask = total_attempts > 0
                if torch.any(valid_mask):
                    avg_rate = (self.success_count[valid_mask].float() / total_attempts[valid_mask].float()).mean().item()
                else:
                    avg_rate = 0.0

                # [기존] 현재 평균 몇 판째인지 계산
                avg_episodes = total_attempts.float().mean().item()
                max_episodes = total_attempts.max().item()

                # [기존] 전체 평균 통계 계산
                if len(env_ids) > 0:
                    current_actual_lengths = self.episode_length_buf[env_ids].float() + 1e-6

                    # A. 평균 성공률
                    avg_success_ratio_val = (self.success_steps_buf[env_ids] / current_actual_lengths).mean().item()
                    # B. 평균 거리 오차
                    avg_distance_error_val = (self.avg_distance_error_buf[env_ids] / current_actual_lengths).mean().item()
                    # C. 평균 총 보상
                    avg_total_reward_val = self.episode_reward_buf[env_ids].mean().item()
                else:
                    avg_success_ratio_val = 0.0
                    avg_distance_error_val = 0.0
                    avg_total_reward_val = 0.0

                print("=" * 80) # 구분선 길이 약간 늘림
                print(f"📊 Curriculum Level Distribution (Total: {self.num_envs})")
                print(f"🔄 Progress: {avg_episodes:.1f} / {self.EVAL_BATCH_SIZE} episodes (Max: {max_episodes})")
                print(f"📈 Level Up/Down Win Rate: {avg_rate * 100:.2f}% (Target: {self.PROMOTION_RATE*100:.0f}%)")
                print("-" * 80)

                # 1. 전체 평균 출력
                print(f"🔍 [Global Stats] Avg of {len(env_ids)} reset envs:")
                print(f"   Total  | Success: {avg_success_ratio_val * 100:6.2f}% | Dist: {avg_distance_error_val * 100:5.2f} cm | Reward: {avg_total_reward_val:6.1f}")

                print("-" * 80)
                print("🔍 [Level-wise Stats]")

                # 2. [추가됨] 각 레벨별 통계 계산 및 출력
                if len(env_ids) > 0:
                    current_levels_reset = self.current_reward_level[env_ids] # 현재 리셋되는 환경들의 레벨

                    # 각 레벨을 순회하며 통계 계산
                    for lvl in range(self.max_reward_level + 1):
                        # 현재 리셋된 환경들 중, 해당 레벨(lvl)인 것들만 마스킹
                        lvl_mask = (current_levels_reset == lvl)
                        lvl_count = torch.sum(lvl_mask).item()

                        if lvl_count > 0:
                            # 해당 레벨의 데이터 추출
                            # env_ids[lvl_mask]는 안됨. env_ids 자체가 인덱스이므로, 불리언 마스크를 사용하여 필터링해야 함
                            # 올바른 방법: 값을 추출한 뒤 마스킹

                            lvl_lengths = current_actual_lengths[lvl_mask]

                            # A. 성공률
                            lvl_success = (self.success_steps_buf[env_ids][lvl_mask] / lvl_lengths).mean().item()
                            # B. 거리 오차
                            lvl_dist = (self.avg_distance_error_buf[env_ids][lvl_mask] / lvl_lengths).mean().item()
                            # C. 총 보상
                            lvl_reward = self.episode_reward_buf[env_ids][lvl_mask].mean().item()

                            print(f"   Level {lvl} ({lvl_count:3d}) | Success: {lvl_success * 100:6.2f}% | Dist: {lvl_dist * 100:5.2f} cm | Reward: {lvl_reward:6.1f}")

                print("-" * 80)

                # 레벨 분포 바 그래프 출력 (기존 코드)
                for level_idx, count in enumerate(level_counts):
                    count_val = count.item()
                    ratio = (count_val / self.num_envs) * 100
                    bar = "#" * int(ratio / 5) 
                    print(f"  Level {level_idx}: {count_val:4d} envs ({ratio:5.1f}%) | {bar}")
                print("=" * 80)

                self.log_counter = 0 # 카운터 초기화

            self.episode_reward_buf[env_ids] = 0.0
            self.avg_distance_error_buf[env_ids] = 0.0
            self.success_steps_buf[env_ids] = 0.0

            # robot state ---------------------------------------------------------------------------------
            if training_mode:            
                new_k_c = torch.pow(self.curriculum_factor_k_c[env_ids], self.curriculum_factor_kd)
                self.curriculum_factor_k_c[env_ids] = new_k_c
                self.curriculum_factor_k_c.clamp_(max=1.0)    
            else:
                if not hasattr(self, "_initialized"):
                    self._initialized = False

                if not self._initialized:
                    joint_pos = self._robot.data.default_joint_pos[env_ids] 

                    joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
                    joint_vel = torch.zeros_like(joint_pos)
                    self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
                    self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

                    self.robot_dof_targets[env_ids] = joint_pos 

                    self._initialized = True

            if training_mode:
                current_levels_for_reset = self.current_reward_level[env_ids]

                # [수정] 5단계로 마스크 확장
                mask_level_0 = (current_levels_for_reset == 0)
                mask_level_1 = (current_levels_for_reset == 1)
                mask_level_2 = (current_levels_for_reset == 2)
                mask_level_3 = (current_levels_for_reset == 3)
                mask_level_4 = (current_levels_for_reset == 4)
                mask_level_5 = (current_levels_for_reset == 5)

                env_ids_level_0 = env_ids[mask_level_0]
                env_ids_level_1 = env_ids[mask_level_1]
                env_ids_level_2 = env_ids[mask_level_2]
                env_ids_level_3 = env_ids[mask_level_3]
                env_ids_level_4 = env_ids[mask_level_4]
                env_ids_level_5 = env_ids[mask_level_5]

                # Level 0: (Static, Robot Speed 0.5)
                if len(env_ids_level_0) > 0:
                    self.object_move_state[env_ids_level_0] = self.MOVE_STATE_STATIC
                    self.obj_speed[env_ids_level_0] = 0.0
                    self.action_scale_tensor[env_ids_level_0] = 3.0 
                    self._perform_static_reset(env_ids_level_0) 

                # [신규] Level 1: (Moving 0.0005, Robot Speed 0.5) - 물체 이동 먼저
                if len(env_ids_level_1) > 0:
                    self.object_move_state[env_ids_level_1] = self.MOVE_STATE_LINEAR
                    self.obj_speed[env_ids_level_1] = 0.05 # 물체 이동 시작
                    self.action_scale_tensor[env_ids_level_1] = 3.0 # 로봇 속도 유지
                    self._perform_linear_reset(env_ids_level_1)

                # [신규] Level 2: (Moving 0.0005, Robot Speed 1.0) - 다음 로봇 속도 증가
                if len(env_ids_level_2) > 0:
                    self.object_move_state[env_ids_level_2] = self.MOVE_STATE_LINEAR
                    self.obj_speed[env_ids_level_2] = 0.10
                    self.action_scale_tensor[env_ids_level_2] = 3.0 # 로봇 속도 증가
                    self._perform_linear_reset(env_ids_level_2)

                # [신규] Level 3: (Moving Random, Robot Speed 1.0) - 다음 물체 속도 증가
                if len(env_ids_level_3) > 0:
                    self.object_move_state[env_ids_level_3] = self.MOVE_STATE_LINEAR
                    num_level_3 = len(env_ids_level_3)
                    self.obj_speed[env_ids_level_3] = 0.20
                    self.action_scale_tensor[env_ids_level_3] = 3.0 # 로봇 속도 유지
                    self._perform_linear_reset(env_ids_level_3)

                # [신규] Level 4: (Moving Random, Robot Speed 1.5) - 최종
                if len(env_ids_level_4) > 0:
                    self.object_move_state[env_ids_level_4] = self.MOVE_STATE_LINEAR
                    # 0.5(최대) - 0.3(최소) = 0.2 (범위)
                    num_level_4 = len(env_ids_level_4)
                    self.obj_speed[env_ids_level_4] = 0.40
                    self.action_scale_tensor[env_ids_level_4] = 3.0 # 로봇 속도 증가
                    self._perform_linear_reset(env_ids_level_4)
                    
                if len(env_ids_level_5) > 0:
                    self.object_move_state[env_ids_level_5] = self.MOVE_STATE_LINEAR
                    # 0.5(최대) - 0.3(최소) = 0.2 (범위)
                    num_level_4 = len(env_ids_level_5)
                    self.obj_speed[env_ids_level_5] = 0.60
                    self.action_scale_tensor[env_ids_level_5] = 3.0 # 로봇 속도 증가
                    self._perform_linear_reset(env_ids_level_5)

            else: # training_mode == False (테스트 모드)
                self.action_scale_tensor[env_ids] = 2.0 # (4.0이 적용됨)

                if object_move == ObjectMoveType.STATIC:
                    self.object_move_state[env_ids] = self.MOVE_STATE_STATIC
                    self.obj_speed[env_ids] = 0.0
                    self._perform_static_reset(env_ids) 

                elif object_move == ObjectMoveType.LINEAR:
                    self.object_move_state[env_ids] = self.MOVE_STATE_LINEAR
                    self.obj_speed[env_ids] = 0.6 
                    self._perform_linear_reset(env_ids)

            self.cfg.current_time = 0
            self._compute_intermediate_values(env_ids)

            self.is_object_visible_mask[env_ids] = False 
            self.current_joint_pos_buffer[env_ids] = self._robot.data.joint_pos[env_ids]
            self.out_of_fov_counter[env_ids] = 0

            if hasattr(self, 'last_error'):
                current_dist = torch.norm(self.robot_grasp_pos[env_ids] - self.box_grasp_pos[env_ids], p=2, dim=-1)
                self.last_error[env_ids] = current_dist
        
        super()._reset_idx(env_ids)
        
        if env_ids is not None:
            self.grasp_phase[env_ids] = 0
            self.stable_timer[env_ids] = 0.0
            self.holding_timer[env_ids] = 0.0
        
        self._compute_intermediate_values(env_ids)

        # 1. 월드 좌표 초기화
        current_pos_w = self._box.data.body_link_pos_w[env_ids, 0, 0:3] - self.scene.env_origins[env_ids]
        self.prev_box_pos_w[env_ids] = current_pos_w.clone()

        # 2. 카메라 좌표 초기화
        camera_pos_w, camera_rot_w = self.compute_camera_world_pose(self.hand_pos[env_ids], self.hand_rot[env_ids])
        current_pos_c, _ = self.world_to_camera_pose(
            camera_pos_w, camera_rot_w,
            current_pos_w, 
            self.box_grasp_rot[env_ids]
        )
        self.prev_box_pos_c[env_ids] = current_pos_c[:, 0:3].clone()

    def _get_observations(self) -> dict:
        self.current_joint_pos_buffer[:] = self._robot.data.joint_pos
        
        dof_pos_scaled = (
            2.0
            * (self._robot.data.joint_pos - self.robot_dof_lower_limits)
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
            - 1.0
        )
        
        # [1] 월드 기준 실제 물체 위치
        box_pos_w_real = self._box.data.body_link_pos_w[:, 0, 0:3] - self.scene.env_origins
        
        # [2] 카메라 월드 위치 계산
        camera_pos_w, camera_rot_w = self.compute_camera_world_pose(self.hand_pos, self.hand_rot)
        
        if not training_mode:
            approach_mask = (self.grasp_phase == 1) # 접근 단계
            
            if torch.any(approach_mask):
                base_offset = camera_pos_w - self.robot_grasp_pos # (N, 3)
                forward_local = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1)
                gripper_forward_w = tf_vector(self.robot_grasp_rot, forward_local)
                
                depth_bias = 0.07
                final_offset = base_offset + (gripper_forward_w * depth_bias)
                
                box_pos_w_target = box_pos_w_real + final_offset
                
                final_box_pos_w = box_pos_w_real.clone()
                final_box_pos_w[approach_mask] = box_pos_w_target[approach_mask]
            
            else:
                final_box_pos_w = box_pos_w_real
        else:
            final_box_pos_w = box_pos_w_real
        
        # [3] 관측값 변환 (월드 좌표 -> 카메라 좌표계)
        box_pos_c_cur, _ = self.world_to_camera_pose(
            camera_pos_w, camera_rot_w,
            final_box_pos_w, # 조작된 월드 좌표 (Tracking: 실제 / Approach: 가상)
            self.box_grasp_rot
        )
        box_pos_c_cur = box_pos_c_cur[:, 0:3]
        
        obs_box_pos = box_pos_c_cur

        # [4] 목표 거리 설정
        if training_mode:
            target_dist = torch.tensor(0.40, device=self.device)
        else:
            approach_mask = (self.grasp_phase == 1)
            # 접근 시 목표 거리 (그리퍼 기준 4~5cm)
            target_dist = torch.where(
                approach_mask, 
                torch.tensor(0.04, device=self.device), 
                torch.tensor(0.40, device=self.device)
            )

        # [5] 최종 관측값 조립
        current_depth = obs_box_pos[:, 2] # (N,)
        z_error = (current_depth - target_dist).unsqueeze(-1)
        xy_offset = torch.norm(obs_box_pos[:, 0:2], p=2, dim=-1).unsqueeze(-1)
        
        obs = torch.cat(
            (
                dof_pos_scaled[:, :6], 
                (self._robot.data.joint_vel * self.cfg.dof_velocity_scale)[:, :6],
                
                obs_box_pos, 
                
                box_pos_w_real, # 기록용은 실제 위치 유지
                self.prev_box_pos_w,
                
                z_error,
                xy_offset,
            ),
            dim=-1,
        )
        
        self.prev_box_pos_w = box_pos_w_real.clone()
        self.prev_box_pos_c = box_pos_c_cur.clone()
        
        return {"policy": torch.clamp(obs, -5.0, 5.0),}
    
    # auxiliary methods

    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        # self.hand_pos = self._robot.data.body_link_pos_w[env_ids, self.hand_link_idx]
        # self.hand_rot = self._robot.data.body_link_quat_w[env_ids, self.hand_link_idx]
        
        self.hand_pos[env_ids] = self._robot.data.body_link_pos_w[env_ids, self.hand_link_idx]
        self.hand_rot[env_ids] = self._robot.data.body_link_quat_w[env_ids, self.hand_link_idx]
        
        box_pos_world = self._box.data.body_link_pos_w[env_ids, self.box_idx]
        box_rot_world = self._box.data.body_link_quat_w[env_ids, self.box_idx]
                
        (
            self.robot_grasp_rot[env_ids],
            self.robot_grasp_pos[env_ids],
            self.box_grasp_rot[env_ids],
            self.box_grasp_pos[env_ids],
        ) = self._compute_grasp_transforms(
            # self.hand_rot,
            # self.hand_pos,
            self.hand_rot[env_ids],          # [수정] env_ids 추가!
            self.hand_pos[env_ids],          # [수정] env_ids 추가!
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
        # 커리큘럼 기반 가중치 설정 (Reward Scales)
        levels = self.current_reward_level
        max_idx = self.max_reward_level
        
        distance_reward_scale = torch.tensor([reward_curriculum_levels[min(l.item(), max_idx)]["reward_scales"]["distance"] for l in levels], device=self.device)
        vector_align_reward_scale = torch.tensor([reward_curriculum_levels[min(l.item(), max_idx)]["reward_scales"]["vector_align"] for l in levels], device=self.device)
        position_align_reward_scale = torch.tensor([reward_curriculum_levels[min(l.item(), max_idx)]["reward_scales"]["position_align"] for l in levels], device=self.device)
        pview_reward_scale = torch.tensor([reward_curriculum_levels[min(l.item(), max_idx)]["reward_scales"]["pview"] for l in levels], device=self.device)
        joint_penalty_scale = torch.tensor([reward_curriculum_levels[min(l.item(), max_idx)]["reward_scales"]["joint_penalty"] for l in levels], device=self.device)
        blind_penalty_scale = torch.tensor([reward_curriculum_levels[min(l.item(), max_idx)]["reward_scales"]["blind_penalty"] for l in levels], device=self.device)
        
        # 커리큘럼 기반 마진 설정
        distance_margin_m = torch.tensor([reward_curriculum_levels[min(l.item(), max_idx)]["distance_margin"] for l in levels], device=self.device)
        vector_align_margin_rad = torch.tensor([reward_curriculum_levels[min(l.item(), max_idx)]["vector_align_margin"] for l in levels], device=self.device)
        position_align_margin_m = torch.tensor([reward_curriculum_levels[min(l.item(), max_idx)]["position_align_margin"] for l in levels], device=self.device)
        pview_margin_m = torch.tensor([reward_curriculum_levels[min(l.item(), max_idx)]["pview_margin"] for l in levels], device=self.device)
        
        ALPHA_DIST = 1.0 / (distance_margin_m + 1e-6)
        ALPHA_VEC = 1.0 / (vector_align_margin_rad + 1e-6)
        ALPHA_POS = 1.0 / (position_align_margin_m + 1e-6)
        ALPHA_PVIEW = 1.0 / (pview_margin_m + 1e-6)
        
        # ESCAPE_GRADIENT = 0.005 
        
        ## R1: 거리 유지 보상 (Distance Reward) - [카메라 기준 수정]
        target_distance = 0.40
        camera_real_distance = torch.norm(box_pos_cam, dim=-1)
        distance_error = camera_real_distance - target_distance
        
        # 너무 가까울 때(음수)는 오차를 2배로 뻥튀기해서 패널티를 키움
        weighted_error = torch.where(
            distance_error < 0, 
            torch.abs(distance_error) * 2.5,  # 가까우면 2.5배 더 민감하게 반응해라!
            torch.abs(distance_error) * 1.0   # 멀면 그냥 원래대로
        )
        
        distance_reward = torch.exp(-ALPHA_DIST * weighted_error)
                
        self.avg_distance_error_buf += distance_error
        # self.episode_steps_buf += 1.0 # 매 스텝 1씩 증가

        ## R2: 각도 정렬 보상 (Vector Alignment Reward)
        box_pos_local = box_pos_w - self.scene.env_origins
        obj_z = box_pos_local[:, 2]
        
        q_cam_in_hand = torch.tensor([0.7071, 0.0, 0.0, 0.7071], device=self.device).repeat(self.num_envs, 1)
        
        deg_bottom = -10.0
        deg_middle =   0.0
        deg_top    =  10.0

        target_angle_deg = torch.full_like(obj_z, deg_middle)
        target_angle_deg = torch.where(obj_z < 0.30, torch.tensor(deg_bottom, device=self.device), target_angle_deg)
        target_angle_deg = torch.where(obj_z >= 0.65, torch.tensor(deg_top, device=self.device), target_angle_deg)

        target_angle_rad = torch.deg2rad(target_angle_deg)

        camera_rot_w = self.quat_mul(franka_grasp_rot, q_cam_in_hand)
        camera_forward_axis_local = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).repeat(self.num_envs, 1)
        camera_forward_world = tf_vector(camera_rot_w, camera_forward_axis_local)
        actual_angle_rad = torch.asin(camera_forward_world[:, 2].clamp(-1.0, 1.0))
        
        angle_error_rad = torch.abs(actual_angle_rad - target_angle_rad)
        vector_alignment_reward = torch.exp(-ALPHA_VEC * angle_error_rad)

        ## R3: 그리퍼 위치 유지 보상 (Position Alignment Reward) - [카메라 기준 수정]
        is_in_front_mask = box_pos_cam[:, 2] > 0 
        center_offset_r3 = torch.norm(box_pos_cam[:, [0,1]], dim=-1)
        position_alignment_reward_raw = torch.exp(-ALPHA_POS * center_offset_r3)
        
        position_alignment_reward = torch.where(
            is_in_front_mask, 
            position_alignment_reward_raw, 
            torch.tensor(1e-6, device=self.device)
        )
                
        ## R4: 시야 유지 보상 (PView Reward) - [수정 없음]
        depth = torch.abs(box_pos_cam[:, 2]) + 1e-6
        physical_offset = torch.norm(box_pos_cam[:, [0,1]], dim=-1)
        view_error_ratio = physical_offset / depth

        pview_positive_reward = (
            torch.exp(-ALPHA_PVIEW * view_error_ratio) 
        )
        pview_reward = torch.where(is_in_front_mask, pview_positive_reward, torch.full_like(view_error_ratio, 1e-6))
        
        # R5: 충돌 패널티 (Collision Penalty)
        # 그리퍼(핑거)에 가해지는 힘을 측정
        contact_forces = torch.norm(self._contact_sensor.data.net_forces_w, p=2, dim=-1) 
        max_contact_force = torch.max(contact_forces, dim=-1)[0]
        self.is_collision = (max_contact_force > 1.0)
        
        collision_penalty = self.is_collision.float() * -8.0
        
        ## 접근 보상 (Approach Reward) - Shaping Reward
        if not hasattr(self, 'last_error'):
            self.last_error = distance_error.clone()
            
        error_improvement = (self.last_error - distance_error)
        approach_reward = torch.clamp(error_improvement, min=0.0) * 6.0
        self.last_error = distance_error.clone()
        
        ## Joint 5 (손목) 범위 제한 보상 (Soft Limit)
        joint5_val = self._robot.data.joint_pos[:, 4]
        
        # 제한 범위 설정 (라디안 변환)
        limit_min = torch.deg2rad(torch.tensor(-30.0, device=self.device))
        limit_max = torch.deg2rad(torch.tensor(-10.0, device=self.device))
    
        violation_min = torch.clamp(limit_min - joint5_val, min=0.0)
        violation_max = torch.clamp(joint5_val - limit_max, min=0.0)
        
        total_violation = violation_min + violation_max
        joint5_limit_penalty = (total_violation ** 2) * (-joint_penalty_scale)
        
        ## gating 기법
        gating_factor = torch.pow(pview_reward, pview_reward_scale)
        weighted_distance_reward = torch.pow(distance_reward, distance_reward_scale) * gating_factor
        
        task_reward = (
            weighted_distance_reward * # (거리 * 시야)
            torch.pow(vector_alignment_reward, vector_align_reward_scale) *
            torch.pow(position_alignment_reward, position_align_reward_scale)
        )
        
        # Blind Penalty (실패 비용 - 빼기)
        # 시야를 놓치면 레벨에 따라 감점 (-0.1 ~ -1.0)
        is_blind = self.is_pview_fail.float()
        blind_penalty = is_blind * (-blind_penalty_scale)
        
        # 최종 합산
        # (잘했니?) + (다가갔니?) - (놓쳤니?)
        rewards = task_reward + approach_reward + blind_penalty + joint5_limit_penalty + + collision_penalty
        self.last_step_reward = rewards.detach()
        
        # print("*" * 50)
        # forces = self._contact_sensor.data.net_forces_w
        # print("cotact_forces :", max_contact_force)
        # print("camera_real_distance :",camera_real_distance)
        # print("distance_reward :", distance_reward)
        # print("distance_error :", distance_error)
        # print("vector_alignment_reward :", vector_alignment_reward)
        # print("position_alignment_reward :", position_alignment_reward)
        # print("view_error_ratio :", view_error_ratio)
        # print("pview_reward :", pview_reward)
                
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