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

class RobotType(Enum):
    FRANKA = "franka"
    UF = "ufactory"
    DOOSAN = "doosan"
robot_type = RobotType.UF

class ObjectMoveType(Enum):
    STATIC = "static"
    CIRCLE = "circle"
    LINEAR = "linear"
    # CURRICULAR = "curricular"
# object_move = ObjectMoveType.STATIC
object_move = ObjectMoveType.LINEAR
# object_move = ObjectMoveType.CURRICULAR

training_mode = True
foundationpose_mode = False

camera_enable = False
image_publish = False
test_graph_mode = False

robot_action = False
robot_init_pose = False
robot_fix = False

init_reward = True
reset_flag = True

add_episode_length = 200
# add_episode_length = -800
# add_episode_length = -900
# add_episode_length = -500

rand_pos_range = {
    "x" : (  0.35, 0.75),
    "y" : ( -0.40, 0.40),
    "z" : (  0.08, 0.75),
    
    # "x" : (  0.5, 0.70),
    # "y" : ( -0.35, 0.35),
    # "z" : (  0.08, 0.7),
    
    # "x" : (  0.0, 0.1),
    # "y" : (  -0.3, 0.3),
    # "z" : (  0.0, 0.08),
}

# reward_curriculum_levels = [
#     # Level 0: (Static, Robot Speed 0.5) - 가장 넓은 마진
#     {
#         "reward_scales": {"pview": 1.0, "distance": 1.0, "vector_align": 0.6, "position_align": 0.8, "joint_penalty": 1.0, "blind_penalty": 0.1},
#         "success_multiplier": 1.2, "failure_multiplier": 0.8, 
#         "y_range" : ( -0.35, 0.35),

#         "distance_margin" : 0.15,
#         "vector_align_margin" : math.radians(20.0),
#         "position_align_margin" : 0.20,
#         "pview_margin" : 0.25,
#         "fail_margin" : 0.35,
#     },
#     # [신규] Level 1: (Moving 0.0005, Robot Speed 0.5) - 물체 이동 "먼저" 학습
#     {
#         "reward_scales": {"pview": 1.0, "distance": 1.0, "vector_align": 0.6, "position_align": 0.8, "joint_penalty": 1.0, "blind_penalty": 0.05},
#         "success_multiplier": 1.0, "failure_multiplier": 1.2, 
#         "y_range" : ( -0.35, 0.35),

#         "distance_margin" : 0.20, # 마진 약간 좁힘
#         "vector_align_margin" : math.radians(25.0),
#         "position_align_margin" : 0.25,
#         "pview_margin" : 0.25,
#         "fail_margin" : 0.35,
#     },
#     # [신규] Level 2: (Moving 0.0005, Robot Speed 1.0) - "그다음" 로봇 속도 증가
#     {
#         "reward_scales": {"pview": 1.0, "distance": 1.0, "vector_align": 0.8, "position_align": 0.8, "joint_penalty": 1.0, "blind_penalty": 0.05},
#         "success_multiplier": 0.9, "failure_multiplier": 1.0, 
#         "y_range": (-0.35, 0.35),

#         "distance_margin" : 0.15,
#         "vector_align_margin" : math.radians(20.0),
#         "position_align_margin" : 0.20,
#         "pview_margin" : 0.25,
#         "fail_margin" : 0.35
#     },
#     # [신규] Level 3: (Moving Random, Robot Speed 1.0) - "그다음" 물체 속도 증가
#     {
#         "reward_scales": {"pview": 1.0, "distance": 1.0, "vector_align": 0.8, "position_align": 0.8, "joint_penalty": 1.0, "blind_penalty": 0.1},
#         "success_multiplier": 0.8, "failure_multiplier": 1.0, 
#         "y_range": (-0.35, 0.35),

#         "distance_margin" : 0.10,
#         "vector_align_margin" : math.radians(15.0),
#         "position_align_margin" : 0.15,
#         "pview_margin" : 0.20,
#         "fail_margin" : 0.30
#     },
#     # [신규] Level 4: (Moving Random, Robot Speed 1.5) - 최종
#     {
#         "reward_scales": {"pview": 1.0, "distance": 1.0, "vector_align": 1.0, "position_align": 1.0, "joint_penalty": 0.5, "blind_penalty": 0.5},
#         "success_multiplier": 1.0, "failure_multiplier": 1.2, 
#         "y_range": (-0.35, 0.35),

#         "distance_margin" : 0.10,
#         "vector_align_margin" : math.radians(10.0),
#         "position_align_margin" : 0.10,
#         "pview_margin" : 0.15,
#         "fail_margin" : 0.30,
#     },
# ]

reward_curriculum_levels = [
    # Level 0: (Static) - 기초 단계부터 공격적으로 설정
    {
        "reward_scales": {
            "distance": 4.0,      # [핵심] 1.0 -> 4.0 (접근이 최우선)
            "pview": 1.0,         # 1.0 유지 (Gating을 위해 유지)
            "vector_align": 0.5,  # 0.6 -> 0.5 (각도는 나중에)
            "position_align": 0.5,# 0.8 -> 0.5 (중앙 정렬보다 거리 좁히기가 우선)
            "joint_penalty": 1.0,# [핵심] 1.0 -> 0.05 (팔 움직이는 비용 무료화)
            "blind_penalty": 0.5  # [상향] 0.1 -> 0.5 (놓치면 치명타)
        },
        "success_multiplier": 1.2, "failure_multiplier": 0.8, 
        "y_range" : ( -0.35, 0.35),

        "distance_margin" : 0.15,
        "vector_align_margin" : math.radians(20.0),
        "position_align_margin" : 0.20,
        "pview_margin" : 0.25,
        "fail_margin" : 0.35,
    },
    # Level 1: (Moving Slow) - 추적 시작
    {
        "reward_scales": {
            "distance": 4.0,      # 접근 강조
            "pview": 1.0,
            "vector_align": 0.5,
            "position_align": 0.5,
            "joint_penalty": 1.0,# 움직임 자유 보장
            "blind_penalty": 0.5  # 놓치지 마라
        },
        "success_multiplier": 1.0, "failure_multiplier": 1.2, 
        "y_range" : ( -0.35, 0.35),

        "distance_margin" : 0.20, 
        "vector_align_margin" : math.radians(25.0),
        "position_align_margin" : 0.25,
        "pview_margin" : 0.25,
        "fail_margin" : 0.35,
    },
    # Level 2: (Moving Planar) - 여기가 고비였음
    {
        "reward_scales": {
            "distance": 4.0,      # 멀어지는 물체 잡으려면 보상이 커야 함
            "pview": 1.0,
            "vector_align": 0.5,
            "position_align": 0.5,
            "joint_penalty": 1.0,# 멀리 뻗어도 감점 없게 함
            "blind_penalty": 0.7
        },
        "success_multiplier": 0.9, "failure_multiplier": 1.0, 
        "y_range": (-0.35, 0.35),

        "distance_margin" : 0.15,
        "vector_align_margin" : math.radians(20.0),
        "position_align_margin" : 0.20,
        "pview_margin" : 0.25,
        "fail_margin" : 0.35
    },
    # Level 3: (Moving Fast)
    {
        "reward_scales": {
            "distance": 4.0, 
            "pview": 1.0, 
            "vector_align": 0.6, # 상위 레벨이니 정밀도 약간 요구
            "position_align": 0.6, 
            "joint_penalty": 1.0, 
            "blind_penalty": 1.0  # 속도가 빠르니 놓치는 거에 더 엄격하게
        },
        "success_multiplier": 0.8, "failure_multiplier": 1.0, 
        "y_range": (-0.35, 0.35),

        "distance_margin" : 0.10,
        "vector_align_margin" : math.radians(15.0),
        "position_align_margin" : 0.15,
        "pview_margin" : 0.20,
        "fail_margin" : 0.30
    },
    # Level 4: (Moving Very Fast)
    {
        "reward_scales": {
            "distance": 4.0, 
            "pview": 1.0, 
            "vector_align": 0.8, 
            "position_align": 0.8, 
            "joint_penalty": 1.0, 
            "blind_penalty": 1.5 # 최고 난이도
        },
        "success_multiplier": 1.0, "failure_multiplier": 1.2, 
        "y_range": (-0.35, 0.35),

        "distance_margin" : 0.10,
        "vector_align_margin" : math.radians(10.0),
        "position_align_margin" : 0.10,
        "pview_margin" : 0.15,
        "fail_margin" : 0.30,
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
            joint_pos = initial_pose,
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "ufactory_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["joint1", "joint2", "joint3"],
                effort_limit = 87.0,
                
                velocity_limit = 2.61,
                stiffness = 2000.0,
                damping = 100.0,
            ),
            "ufactory_forearm": ImplicitActuatorCfg(
                joint_names_expr=["joint4", "joint5", "joint6"],
                effort_limit = 87.0,
                
                velocity_limit = 2.61,
                stiffness = 2000.0,
                damping = 100.0,
            ),
        },
    )

    ## camera
    if camera_enable:
        camera = CameraCfg(
            
            # prim_path="/World/envs/env_.*/xarm6_with_gripper/link6/hand_camera",
            prim_path="/World/envs/env_.*/xarm6/link6/hand_camera",
            update_period=0.03,
            height=480,
            width=640,
            data_types=["rgb", "depth"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=30.0, # 값이 클수록 확대
                focus_distance=60.0,
                horizontal_aperture=50.0,
                clipping_range=(0.1, 1.0e5),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.07, 0.03, -0.13), # 위/아래, 좌/우, 앞/뒤
                rot=(0.7071, 0.0, 0.0, 0.7071),
            )
        )
            
    
    ## mustard
    box = RigidObjectCfg(
        # prim_path="/World/envs/env_.*/base_link",
        prim_path="/World/envs/env_.*/bottle",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0, 0.08), rot=(0.923, 0, 0, -0.382)),
        spawn=UsdFileCfg(
            usd_path="/home/nmail-njh/NMAIL/01_Project/Robot_Grasping/objects_usd/google_objects_usd/006_mustard_bottle/006_mustard_bottle.usd",
            scale=(1.0, 1.0, 1.0),
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
                collision_enabled = False,  # [핵심] 0으로 설정하면 "아무것과도 충돌하지 않음"
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
        
        if not training_mode and test_graph_mode:
            if os.path.exists(CSV_FILEPATH):
                os.remove(CSV_FILEPATH)
                print(f"'{CSV_FILEPATH}' 파일을 삭제하고 새로 시작합니다.")
            
            self.csv_filepath = "tracking_data.csv"
            with open(self.csv_filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['gripper_x', 'gripper_y', 'gripper_z', 
                                 'object_x', 'object_y', 'object_z',
                                 'cam_x', 'cam_y','distance'])
        
        self.boundaries_x = torch.tensor([workspace_zones["x"]["middle"], workspace_zones["x"]["far"]], device=self.device)
        self.boundaries_z = torch.tensor([workspace_zones["z"]["middle"], workspace_zones["z"]["top"]], device=self.device)
        
        self.log_counter = 0
        self.LOG_INTERVAL = 6 
        
        # 성능 모니터링을 위한 버퍼
        self.episode_reward_buf = torch.zeros(self.num_envs, device=self.device)
        
        #보상 스케일만 조절하는 새로운 커리큘럼 레벨 정의
        self.max_reward_level = len(reward_curriculum_levels) - 1
        self.baseline_avg_reward = 0.05 # 계산된 기준 보상값

        #보상 커리큘럼을 위한 독립적인 상태 변수들
        self.current_reward_level = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # self.current_reward_level = torch.full((self.num_envs,), 2, dtype=torch.long, device=self.device)
        
        self.episode_init_joint_pos = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)
        
        self.curriculum_factor_k0 = 0.25  # k_c의 초기값 (논문 권장값)
        self.curriculum_factor_kd = 0.997 # k_c의 진전 속도
        
        # k_c (커리큘럼 계수) 상태 변수. 모든 환경이 k_c의 초기값에서 시작.
        # k_c는 (num_envs, 1) 형태로 저장됨
        self.curriculum_factor_k_c = torch.full((self.num_envs, 1), self.curriculum_factor_k0, device=self.device)
        
        # 물체 이동 상태를 정의하는 상수
        self.MOVE_STATE_STATIC = 0
        self.MOVE_STATE_LINEAR = 1

        # 4096개 환경의 이동 상태를 개별적으로 저장하는 텐서 (0 = STATIC, 1 = LINEAR)
        self.object_move_state = torch.full(
            (self.num_envs,), self.MOVE_STATE_STATIC, dtype=torch.long, device=self.device
        )
        
        # 4096개 환경의 물체 이동 속도를 개별적으로 저장하는 텐서
        self.obj_speed = torch.zeros(
            (self.num_envs,), device=self.device, dtype=torch.float32
        )
        
        # 4096개 환경의 액션 스케일(반응 속도)을 개별적으로 저장하는 텐서
        # Level 0의 기본값(낮은 속도)으로 초기화합니다.
        self.action_scale_tensor = torch.full(
            (self.num_envs,), 0.5, device=self.device, dtype=torch.float32
        )
        
        self.joint_names = ["joint1", "joint2", "joint3", "joint4","joint5", "joint6"]
        self.joint_init_values = [initial_pose[name] for name in self.joint_names]

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

        self.gripper_forward_axis = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).repeat((self.num_envs, 1))
            
        self.gripper_up_axis = torch.tensor([1, 0, 0], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        
        self.box_z_axis = torch.tensor([0,0,1], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs,1)
        )
        
        self.box_idx = self._box.find_bodies("base_link")[0][0]

        self.robot_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.robot_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        
        self.box_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.box_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.box_center = self._box.data.body_link_pos_w[:,0,:].clone()
        
        self.box_pos_cam = torch.zeros((self.num_envs, 4), device=self.device)        
        
        self.target_box_pos = torch.stack([
                torch.rand(self.num_envs, device=self.device) * (rand_pos_range["x"][1] - rand_pos_range["x"][0]) + rand_pos_range["x"][0],
                torch.rand(self.num_envs, device=self.device) * (rand_pos_range["y"][1] - rand_pos_range["y"][0]) + rand_pos_range["y"][0],
                torch.rand(self.num_envs, device=self.device) * (rand_pos_range["z"][1] - rand_pos_range["z"][0]) + rand_pos_range["z"][0],
            ], dim = 1)
        
        self.target_box_pos = self.target_box_pos + self.scene.env_origins
        self.new_box_pos_rand = torch.zeros((self.num_envs, 3), device=self.device)
        
        self.current_box_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.current_box_pos = torch.zeros((self.num_envs, 3), device=self.device)

        self.rand_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.rand_pos_step = torch.zeros((self.num_envs, 3), device=self.device)
        
        rclpy.init()
        self.last_publish_time = 0.0
        self.position_error = 0.0
        self.obj_origin_distance = 0.0
        self.out_of_fov_cnt = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        
        if image_publish:      
            qos_profile = QoSProfile(depth=10)
            qos_profile.reliability = QoSReliabilityPolicy.BEST_EFFORT

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
        
        self.out_of_fov_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.is_object_visible_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.current_joint_pos_buffer = self._robot.data.joint_pos.clone()
        
        # [수정] 승률 계산을 위한 단순 카운터 변수 (이름은 기존거 재활용해도 되지만, 명확히 하기 위해)
        self.success_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.failure_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
        self.avg_distance_error_buf = torch.zeros(self.num_envs, device=self.device)
        self.success_steps_buf = torch.zeros(self.num_envs, device=self.device)
        
        # 에피소드 성공 판단 기준
        self.EVAL_BATCH_SIZE = 10     # 20판마다 성적 평가
        self.PROMOTION_RATE = 0.80     # 승률 90% 이상이면 레벨업 (20판 중 18승)
        self.DEMOTION_RATE = 0.10      # 승률 40% 이하면 레벨다운 (20판 중 8승 이하)
        
        self.MIN_PVIEW_RATIO = 0.90  # 에피소드 길이의 90% 이상 유지
        self.MAX_DISTANCE_ERROR = 0.10  # 평균 거리 오차 10cm 미만
        self.MIN_TOTAL_REWARD = 200.0    # 에피소드 총 보상 최소 60.0 이상
        
        # 1. 시야 유지 비율 (90% -> 50%로 점차 완화)
        # self.LEVEL_THRESHOLDS_RATIO = torch.tensor([0.95, 0.90, 0.85, 0.80, 0.80], device=self.device)

        # # 2. 거리 오차 허용 (10cm -> 25cm로 점차 완화)
        # self.LEVEL_THRESHOLDS_DIST = torch.tensor([0.10, 0.15, 0.15, 0.20, 0.25], device=self.device)
        
        # # 3. 보상 점수 커트라인 (400점 -> 150점으로 완화)
        # self.LEVEL_THRESHOLDS_REWARD = torch.tensor([400.0, 350.0, 300.0, 200.0, 200.0], device=self.device)
        
        self.speed_change_timer = torch.zeros(self.num_envs, device=self.device)
        self.current_speed_factor = torch.ones(self.num_envs, device=self.device)
        
        self.zone_names_ordered = [
            "top_close", "top_middle", "top_far",
            "middle_close", "middle_middle", "middle_far",
            "bottom_close", "bottom_middle", "bottom_far"
        ]
        
        ref_poses = []
        for name in self.zone_names_ordered:
            pose_dict = pose_candidate[name]
            # 딕셔너리 값을 리스트로 변환 (joint 순서 주의)
            # xArm6 기준 joint names 순서대로 추출
            joint_vals = [pose_dict[jn] for jn in self.joint_names if jn in pose_dict]
            
            # 만약 joint1(Yaw)이 포함 안 되어 있다면 0.0으로 채움 (나중에 동적 계산)
            if len(joint_vals) < len(self.joint_names):
                # joint1을 위한 자리(0.0)를 맨 앞에 추가한다고 가정 (xArm 구조에 따라 조정 필요)
                joint_vals = [0.0] + joint_vals 
                
            ref_poses.append(joint_vals)
            
        self.ref_pose_tensor = torch.tensor(ref_poses, device=self.device, dtype=torch.float32)
        
        self.level1_axis_mode = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.level2_plane_mode = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.hand_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.hand_rot = torch.zeros((self.num_envs, 4), device=self.device)
        
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
        """실시간 그래프를 위한 Figure와 Axes를 초기화합니다."""
        from matplotlib.patches import Circle
        
        plt.ion()  # 대화형 모드 켜기
        
        # Figure 1: 3D Trajectory
        self.fig1 = plt.figure(figsize=(8, 7))
        self.ax1 = self.fig1.add_subplot(111, projection='3d')
        self.traj_obj_line, = self.ax1.plot([], [], [], label='Object Trajectory', color='blue')
        self.traj_grip_line, = self.ax1.plot([], [], [], label='Gripper Trajectory', color='red', linestyle='--')
        self.ax1.set_title('Real-time 3D Trajectory')
        self.ax1.set_xlabel('X'); self.ax1.set_ylabel('Y'); self.ax1.set_zlabel('Z')
        self.ax1.legend()
        self.ax1.grid(True)
        
        # Figure 2: Camera View
        self.fig2, self.ax2 = plt.subplots(figsize=(7, 7))
        self.cam_scatter = self.ax2.scatter([], [], label='Object in Camera View')
        self.ax2.axhline(0, color='black', linestyle='--', linewidth=1)
        self.ax2.axvline(0, color='black', linestyle='--', linewidth=1)
        self.ax2.set_title('Real-time Object Position in Camera Frame')
        self.ax2.set_xlabel('X'); self.ax2.set_ylabel('Y')
        self.ax2.set_aspect('equal', adjustable='box')
        self.ax2.grid(True)

        # pview_margin 원 추가 (초기 레벨 기준)
        pview_margin = reward_curriculum_levels[0]["pview_margin"]
        self.margin_circle = Circle((0, 0), pview_margin, color='red', fill=False, linestyle='-.', label=f'pview_margin')
        self.ax2.add_artist(self.margin_circle)
        self.ax2.legend()
        
        plt.show(block=False) # 창을 띄우되, 코드 실행을 막지 않음
        
    def _update_realtime_plots(self):
        """수집된 데이터로 그래프를 업데이트합니다."""
        # 데이터가 없으면 실행하지 않음
        if not self.graph_data["gripper_positions"]:
            return

        # numpy 배열로 변환
        gripper_pos = np.array(self.graph_data["gripper_positions"])
        object_pos = np.array(self.graph_data["object_positions"])
        cam_pos = np.array(self.graph_data["object_pos_in_cam"])
        
        # --- 3D Trajectory 업데이트 ---
        self.traj_obj_line.set_data(object_pos[:, 0], object_pos[:, 1])
        self.traj_obj_line.set_3d_properties(object_pos[:, 2])
        self.traj_grip_line.set_data(gripper_pos[:, 0], gripper_pos[:, 1])
        self.traj_grip_line.set_3d_properties(gripper_pos[:, 2])
        
        # 축 범위 자동 조절
        self.ax1.relim()
        self.ax1.autoscale_view(True, True, True)
        
        # --- Camera View 업데이트 ---
        # Scatter는 set_offsets로 효율적으로 업데이트
        self.cam_scatter.set_offsets(cam_pos)
        
        # 현재 레벨에 맞는 pview_margin으로 원 업데이트
        current_level = self.current_reward_level[0].item()
        pview_margin = reward_curriculum_levels[current_level]["pview_margin"]
        self.margin_circle.set_radius(pview_margin)
        
        # 축 범위 자동 조절
        self.ax2.relim()
        self.ax2.autoscale_view(True, True)

        # 캔버스 다시 그리기
        self.fig1.canvas.draw()
        self.fig2.canvas.draw()
        plt.pause(0.001) # GUI가 업데이트될 시간을 줌
    
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
        
        self._box = RigidObject(self.cfg.box)
        self.scene.rigid_objects["base_link"] = self._box

    # pre-physics step calls

    def _pre_physics_step(self, actions: torch.Tensor):
        
        self.actions = actions.clone().clamp(-1.0, 1.0)
                
        current_action_scale = self.action_scale_tensor.unsqueeze(-1) 
        potential_targets = self.robot_dof_targets + self.robot_dof_speed_scales * self.dt * self.actions * current_action_scale
        potential_targets_clamped = torch.clamp(potential_targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)

        if training_mode:
            self.robot_dof_targets[:] = potential_targets_clamped
        else:            
            hold_targets = self.current_joint_pos_buffer

            visible_mask_expanded = self.is_object_visible_mask.unsqueeze(-1) 
            
            self.robot_dof_targets[:] = torch.where(
                visible_mask_expanded, 
                potential_targets_clamped,  # 시야 O: 행동 적용
                # hold_targets                # 시야 X: 현재 위치 고수 (정지)
                potential_targets_clamped
            )

        self.cfg.current_time = self.cfg.current_time + self.dt
        current_time = torch.tensor(self.cfg.current_time, device=self.device, dtype=torch.float32)
        
        if image_publish:   
            self.last_publish_time += self.dt
            if self.last_publish_time >= (1.0 / 15.0):  # 30fps 기준
                self.publish_camera_data()
                rclpy.spin_once(self.node, timeout_sec=0.001)
                self.last_publish_time = 0.0

        # 물체 위치 랜덤 선형 이동 (Per-Environment)
        # 1. LINEAR 상태인 환경의 인덱스를 찾습니다.
        linear_move_mask = (self.object_move_state == self.MOVE_STATE_LINEAR)
        linear_env_ids = torch.where(linear_move_mask)[0]

        if len(linear_env_ids) > 0:
            # 2. 현재 시뮬레이션 상의 '실제' 위치 가져오기 (매우 중요!)
            # 기존에는 self.new_box_pos_rand 변수로 위치를 따로 관리했지만,
            # 이제는 물리 엔진이 이동시키므로 실제 위치를 조회해야 오차가 없습니다.
            current_pos_world = self._box.data.root_pos_w[linear_env_ids] # (N, 3)
            
            # 3. 목표 도달 확인 (거리 1cm 미만)
            target_pos = self.target_box_pos[linear_env_ids]
            distance_to_target = torch.norm(target_pos - current_pos_world, p=2, dim=-1)
            reached_target_mask = (distance_to_target < 0.01)

            # --- [A] 목표 도달 시: 새로운 목표 설정 (기존 로직과 유사하지만 변수 정리) ---
            if torch.any(reached_target_mask):
                env_ids_to_update = linear_env_ids[reached_target_mask]
                
                num_to_update = len(env_ids_to_update)
                current_levels = self.current_reward_level[env_ids_to_update]

                # 다음 목표 생성을 위한 랜덤 후보군 & 기준 위치 설정

                # 1. 전체 범위 내 랜덤 좌표 생성 (Level 2, 3용 차기 목표 후보)
                rand_x = torch.rand(num_to_update, device=self.device) * (rand_pos_range["x"][1] - rand_pos_range["x"][0]) + rand_pos_range["x"][0]
                rand_y = torch.rand(num_to_update, device=self.device) * (rand_pos_range["y"][1] - rand_pos_range["y"][0]) + rand_pos_range["y"][0]
                rand_z = torch.rand(num_to_update, device=self.device) * (rand_pos_range["z"][1] - rand_pos_range["z"][0]) + rand_pos_range["z"][0]

                # 2. 기준 위치 가져오기 (축 고정용)
                # "축을 고정한다" = "방금 도달한 목표 위치(직전 Target)를 그대로 유지한다"
                # (현재 물체 위치인 new_box_pos_rand 대신 target_box_pos를 써야 오차가 누적되지 않음)
                curr_target_world = self.target_box_pos[env_ids_to_update]
                curr_target_local = curr_target_world - self.scene.env_origins[env_ids_to_update]

                curr_x = curr_target_local[:, 0]
                curr_y = curr_target_local[:, 1]
                curr_z = curr_target_local[:, 2]

                # 3. 최종 목표 변수 초기화 (기본값: 3D Random)
                final_target_x = rand_x
                final_target_y = rand_y
                final_target_z = rand_z

                mask_lv1 = (current_levels == 1)
                mask_lv2 = (current_levels == 2)

                # [Level 1] 1차원 왕복 운동 (Reciprocating)
                if torch.any(mask_lv1):
                    # 1. Level 1에 해당하는 데이터만 추출 (Size: K)
                    ids_lv1 = env_ids_to_update[mask_lv1]
                    axis_modes = self.level1_axis_mode[ids_lv1]

                    # 2. Level 1용 임시 변수 추출
                    t_x_lv1 = final_target_x[mask_lv1]
                    t_y_lv1 = final_target_y[mask_lv1]
                    t_z_lv1 = final_target_z[mask_lv1]
                    
                    c_x_lv1 = curr_x[mask_lv1]
                    c_y_lv1 = curr_y[mask_lv1]
                    c_z_lv1 = curr_z[mask_lv1]

                    x_min, x_max = rand_pos_range["x"]
                    y_min, y_max = rand_pos_range["y"]
                    z_min, z_max = rand_pos_range["z"]

                    # --- X축 왕복 ---
                    # (axis_modes는 Size K이므로 바로 연산 가능)
                    sub_cond_x = (axis_modes == 0)
                    
                    dist_to_max_x = torch.abs(c_x_lv1 - x_max)
                    dist_to_min_x = torch.abs(c_x_lv1 - x_min)
                    next_x = torch.where(dist_to_max_x < dist_to_min_x, torch.tensor(x_min, device=self.device), torch.tensor(x_max, device=self.device))

                    t_x_lv1 = torch.where(sub_cond_x, next_x, t_x_lv1)
                    t_y_lv1 = torch.where(sub_cond_x, c_y_lv1, t_y_lv1) # Y 고정
                    t_z_lv1 = torch.where(sub_cond_x, c_z_lv1, t_z_lv1) # Z 고정

                    # --- Y축 왕복 ---
                    sub_cond_y = (axis_modes == 1)
                    dist_to_max_y = torch.abs(c_y_lv1 - y_max)
                    dist_to_min_y = torch.abs(c_y_lv1 - y_min)
                    next_y = torch.where(dist_to_max_y < dist_to_min_y, torch.tensor(y_min, device=self.device), torch.tensor(y_max, device=self.device))

                    t_x_lv1 = torch.where(sub_cond_y, c_x_lv1, t_x_lv1) # X 고정
                    t_y_lv1 = torch.where(sub_cond_y, next_y, t_y_lv1)
                    t_z_lv1 = torch.where(sub_cond_y, c_z_lv1, t_z_lv1) # Z 고정

                    # --- Z축 왕복 ---
                    sub_cond_z = (axis_modes == 2)
                    dist_to_max_z = torch.abs(c_z_lv1 - z_max)
                    dist_to_min_z = torch.abs(c_z_lv1 - z_min)
                    next_z = torch.where(dist_to_max_z < dist_to_min_z, torch.tensor(z_min, device=self.device), torch.tensor(z_max, device=self.device))

                    t_x_lv1 = torch.where(sub_cond_z, c_x_lv1, t_x_lv1) # X 고정
                    t_y_lv1 = torch.where(sub_cond_z, c_y_lv1, t_y_lv1) # Y 고정
                    t_z_lv1 = torch.where(sub_cond_z, next_z, t_z_lv1)
                    
                    # 3. 계산된 결과를 원본 텐서에 덮어쓰기
                    final_target_x[mask_lv1] = t_x_lv1
                    final_target_y[mask_lv1] = t_y_lv1
                    final_target_z[mask_lv1] = t_z_lv1

                # [Level 2] 2차원 연속 이동 (Continuous Planar Random Walk) - [수정됨]
                if torch.any(mask_lv2):
                    ids_lv2 = env_ids_to_update[mask_lv2]
                    plane_modes = self.level2_plane_mode[ids_lv2]
                    
                    t_x_lv2 = final_target_x[mask_lv2]
                    t_y_lv2 = final_target_y[mask_lv2]
                    t_z_lv2 = final_target_z[mask_lv2]
                    
                    c_x_lv2 = curr_x[mask_lv2]
                    c_y_lv2 = curr_y[mask_lv2]
                    c_z_lv2 = curr_z[mask_lv2]

                    # XY 평면 (Z 고정)
                    sub_cond_xy = (plane_modes == 0)
                    t_z_lv2 = torch.where(sub_cond_xy, c_z_lv2, t_z_lv2)

                    # XZ 평면 (Y 고정)
                    sub_cond_xz = (plane_modes == 1)
                    t_y_lv2 = torch.where(sub_cond_xz, c_y_lv2, t_y_lv2)

                    # YZ 평면 (X 고정)
                    sub_cond_yz = (plane_modes == 2)
                    t_x_lv2 = torch.where(sub_cond_yz, c_x_lv2, t_x_lv2)
                    
                    # 2. 결과 덮어쓰기
                    final_target_x[mask_lv2] = t_x_lv2
                    final_target_y[mask_lv2] = t_y_lv2
                    final_target_z[mask_lv2] = t_z_lv2

                new_targets = torch.stack([final_target_x, final_target_y, final_target_z], dim=1)

                self.target_box_pos[env_ids_to_update] = new_targets + self.scene.env_origins[env_ids_to_update]

            # 1. 타이머 감소
            self.speed_change_timer[linear_env_ids] -= self.dt
            
            # 2. 타이머가 0 이하로 떨어진 환경들 찾기 (속도를 바꿀 때가 된 환경들)
            # 주의: linear_env_ids 중에서 골라내야 하므로 인덱싱에 주의해야 합니다.
            # 전체 환경 기준 마스크를 씁니다.
            time_up_mask = (self.speed_change_timer <= 0.0) & linear_move_mask
            env_ids_to_change_speed = torch.where(time_up_mask)[0]
            
            if len(env_ids_to_change_speed) > 0:
                # 3. 새로운 노이즈 비율 생성 (0.7 ~ 1.3)
                new_noise = (torch.rand(len(env_ids_to_change_speed), device=self.device) * 0.6) + 0.7
                self.current_speed_factor[env_ids_to_change_speed] = new_noise
                
                # 4. 타이머 리셋 (0.5초 ~ 1.5초 사이 랜덤 유지)
                # 즉, 한 번 속도가 변하면 최소 0.5초, 최대 1.5초 동안은 그 속도를 유지함
                new_duration = (torch.rand(len(env_ids_to_change_speed), device=self.device) * 1.0) + 0.5
                self.speed_change_timer[env_ids_to_change_speed] = new_duration
            
            # 1) 방향 벡터 (매 프레임 갱신 - 타겟을 향해 계속 조향해야 하므로)
            target_pos_updated = self.target_box_pos[linear_env_ids]
            direction = target_pos_updated - current_pos_world
            direction_norm = torch.norm(direction, p=2, dim=-1, keepdim=True) + 1e-6
            unit_direction = direction / direction_norm
            
            # 2) 속도 크기 (저장된 factor 사용)
            # 매 프레임 바뀌는 게 아니라, 타이머에 의해 갱신된 값을 계속 사용
            base_speed = self.obj_speed[linear_env_ids].unsqueeze(-1)
            active_noise = self.current_speed_factor[linear_env_ids].unsqueeze(-1)
            
            final_speed = base_speed * active_noise
            
            lin_vel = unit_direction * final_speed
            
            # 3) 물리 엔진 적용
            velocity_command = torch.zeros((len(linear_env_ids), 6), device=self.device)
            velocity_command[:, 0:3] = lin_vel
            self._box.write_root_velocity_to_sim(velocity_command, env_ids=linear_env_ids)
        
    def _apply_action(self):
        
        global robot_action
        global robot_init_pose
        
        target_pos = self.robot_dof_targets.clone()
        
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
                        
                        if self.init_cnt > 50: 
                            robot_action = True
                            robot_init_pose = True
                            
                elif foundationpose_mode == False and max_err < 0.3:
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
                
                if max_err < 0.3:
                    robot_init_pose = True
                    robot_action = True
                
            elif robot_init_pose:
                self._robot.set_joint_position_target(target_pos)
        
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

    # Refresh the intermediate values after the physics steps
    def _get_rewards(self) -> torch.Tensor:
        self._compute_intermediate_values()
    
        camera_pos_w, camera_rot_w = self.compute_camera_world_pose(self.hand_pos, self.hand_rot)

        self.box_pos_cam, box_rot_cam = self.world_to_camera_pose(
            camera_pos_w, camera_rot_w,
            self.box_grasp_pos - self.scene.env_origins, self.box_grasp_rot,
        )
        
        gripper_to_box_dist = torch.norm(self.robot_grasp_pos - self.box_grasp_pos, p=2, dim=-1)

        if not training_mode and test_graph_mode: 
            gripper_pos = self.robot_grasp_pos[0].cpu().numpy()
            object_pos = self.box_grasp_pos[0].cpu().numpy()
            
            cam_pos = np.zeros(2) 
            cam_pos[0] = self.box_pos_cam[0,0].cpu().numpy() #x축
            cam_pos[1] = self.box_pos_cam[0,1].cpu().numpy() #y축
            
            distance_val = gripper_to_box_dist[0].cpu().numpy()

            with open(self.csv_filepath, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([gripper_pos[0], gripper_pos[1], gripper_pos[2],
                                 object_pos[0], object_pos[1], object_pos[2],
                                 cam_pos[0], cam_pos[1],
                                distance_val])
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
            if not zone_activation.get(key, False): # .get()으로 안전하게 접근
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

        # [수정] self.rand_pos[env_ids] 에 할당
        self.rand_pos[env_ids] = torch.stack([x_pos, y_pos, z_pos], dim=1)
        rand_reset_pos = self.rand_pos[env_ids] + self.scene.env_origins[env_ids]
        
        random_angles = torch.rand(num_resets, device=self.device) * 2 * torch.pi
        rand_reset_rot = torch.stack([
            torch.cos(random_angles / 2),
            torch.zeros(num_resets, device=self.device),
            torch.zeros(num_resets, device=self.device),
            torch.sin(random_angles / 2)  
        ], dim=1)
        
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
            
            # [추가] 액추에이터 목표 변수(self.robot_dof_targets)도 리셋합니다.
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

        direction = self.target_box_pos[env_ids] - self.new_box_pos_rand[env_ids]
        direction_norm = torch.norm(direction, p=2, dim=-1, keepdim=True) + 1e-6
        
        speed = self.obj_speed[env_ids].unsqueeze(-1)
        self.rand_pos_step[env_ids] = (direction / direction_norm * speed)

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
                mask_level_4_plus = (current_levels_for_reset >= 4)

                env_ids_level_0 = env_ids[mask_level_0]
                env_ids_level_1 = env_ids[mask_level_1]
                env_ids_level_2 = env_ids[mask_level_2]
                env_ids_level_3 = env_ids[mask_level_3]
                env_ids_level_4_plus = env_ids[mask_level_4_plus]

                # Level 0: (Static, Robot Speed 0.5)
                if len(env_ids_level_0) > 0:
                    self.object_move_state[env_ids_level_0] = self.MOVE_STATE_STATIC
                    self.obj_speed[env_ids_level_0] = 0.0
                    self.action_scale_tensor[env_ids_level_0] = 1.0 
                    self._perform_static_reset(env_ids_level_0) 

                # [신규] Level 1: (Moving 0.0005, Robot Speed 0.5) - 물체 이동 먼저
                if len(env_ids_level_1) > 0:
                    self.object_move_state[env_ids_level_1] = self.MOVE_STATE_LINEAR
                    self.obj_speed[env_ids_level_1] = 0.05 # 물체 이동 시작
                    self.action_scale_tensor[env_ids_level_1] = 1.0 # 로봇 속도 유지
                    self._perform_linear_reset(env_ids_level_1)

                # [신규] Level 2: (Moving 0.0005, Robot Speed 1.0) - 다음 로봇 속도 증가
                if len(env_ids_level_2) > 0:
                    self.object_move_state[env_ids_level_2] = self.MOVE_STATE_LINEAR
                    self.obj_speed[env_ids_level_2] = 0.07
                    self.action_scale_tensor[env_ids_level_2] = 1.0 # 로봇 속도 증가
                    self._perform_linear_reset(env_ids_level_2)

                # [신규] Level 3: (Moving Random, Robot Speed 1.0) - 다음 물체 속도 증가
                if len(env_ids_level_3) > 0:
                    self.object_move_state[env_ids_level_3] = self.MOVE_STATE_LINEAR
                    num_level_3 = len(env_ids_level_3)
                    random_speeds = torch.rand(num_level_3, device=self.device) * (0.0015 - 0.0007) + 0.0007
                    self.obj_speed[env_ids_level_3] = 0.1
                    self.action_scale_tensor[env_ids_level_3] = 1.0 # 로봇 속도 유지
                    self._perform_linear_reset(env_ids_level_3)

                # [신규] Level 4: (Moving Random, Robot Speed 1.5) - 최종
                if len(env_ids_level_4_plus) > 0:
                    self.object_move_state[env_ids_level_4_plus] = self.MOVE_STATE_LINEAR
                    # num_level_4_plus = len(env_ids_level_4_plus)
                    # random_speeds = torch.rand(num_level_4_plus, device=self.device) * (0.0015 - 0.0007) + 0.0007
                    # self.obj_speed[env_ids_level_4_plus] = random_speeds
                    self.obj_speed[env_ids_level_4_plus] = 0.15
                    self.action_scale_tensor[env_ids_level_4_plus] = 1.0 # 로봇 속도 증가
                    self._perform_linear_reset(env_ids_level_4_plus)

            else: # training_mode == False (테스트 모드)
                self.action_scale_tensor[env_ids] = 2.0 # (4.0이 적용됨)

                if object_move == ObjectMoveType.STATIC:
                    self.object_move_state[env_ids] = self.MOVE_STATE_STATIC
                    self.obj_speed[env_ids] = 0.0
                    self._perform_static_reset(env_ids) 

                elif object_move == ObjectMoveType.LINEAR:
                    self.object_move_state[env_ids] = self.MOVE_STATE_LINEAR
                    self.obj_speed[env_ids] = 0.3 
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
    
    def _get_observations(self) -> dict:
        self.current_joint_pos_buffer[:] = self._robot.data.joint_pos
        
        dof_pos_scaled = (
            2.0
            * (self._robot.data.joint_pos - self.robot_dof_lower_limits)
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
            - 1.0
        )
        
        # [수정 1] 월드 기준 벡터 대신, 계산해둔 '카메라 기준 물체 위치'를 사용 (치트키)
        # to_target 대신 box_pos_cam의 XYZ를 넣으면 훨씬 학습이 빠릅니다.
        # box_pos_cam은 _get_rewards에서 계산되므로, 순서상 _compute_intermediate_values가 먼저 호출되어야 합니다.
        # (DirectRLEnv 구조상 step -> pre_physics -> ... -> get_obs -> get_reward 순서라
        #  이전 스텝의 값이 들어가거나, 여기서 다시 계산해야 합니다. 안전하게 다시 계산합니다.)
        
        camera_pos_w, camera_rot_w = self.compute_camera_world_pose(self.hand_pos, self.hand_rot)
        box_pos_cam_obs, _ = self.world_to_camera_pose(
            camera_pos_w, camera_rot_w,
            self._box.data.body_link_pos_w[:, 0, 0:3] - self.scene.env_origins, # box_grasp_pos 대신 link pos 사용
            self.box_grasp_rot # 회전은 크게 중요치 않음
        )
        
        # [수정 2] 속도 값 뻥튀기 (Scaling)
        # vel_scale = 100.0 

        obs = torch.cat(
            (
                dof_pos_scaled,
                self._robot.data.joint_vel * self.cfg.dof_velocity_scale,
                box_pos_cam_obs[:, 0:3], 
                self._box.data.body_link_pos_w[:, 0, 0:3],
                self._box.data.body_link_vel_w[:, 0, 0:3],
            ),
            dim=-1,
        )
        
        # Observation Space 크기가 바뀌었으니 Config에서 수정 필요!
        # 기존 21 -> 21 (크기는 같음. to_target(3)을 box_pos_cam(3)으로 대체했으므로)
        # 만약 to_target도 남기고 싶다면 크기를 24로 늘려야 함.
        
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
        distance_error = torch.abs(camera_real_distance - target_distance)
        
        distance_reward = (
            torch.exp(-ALPHA_DIST * distance_error)
        )
                
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
        
        # 최종 보상 조합 (하이브리드 구조)
        # A. Task Reward (성공 조건들 - 곱하기)
        # task_reward = (
        #     torch.pow(distance_reward, distance_reward_scale) *
        #     torch.pow(vector_alignment_reward, vector_align_reward_scale) *
        #     torch.pow(position_alignment_reward, position_align_reward_scale) * 
        #     torch.pow(pview_reward, pview_reward_scale)
        # )
        
        # B. Blind Penalty (실패 비용 - 빼기)
        # 시야를 놓치면 레벨에 따라 감점 (-0.1 ~ -1.0)
        is_blind = self.is_pview_fail.float()
        blind_penalty = is_blind * (-blind_penalty_scale)
        
        
        
        # C. 최종 합산
        # (잘했니?) + (다가갔니?) - (놓쳤니?)
        rewards = task_reward + approach_reward + blind_penalty + joint5_limit_penalty
        self.last_step_reward = rewards.detach()
        
        # print("*" * 50)
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