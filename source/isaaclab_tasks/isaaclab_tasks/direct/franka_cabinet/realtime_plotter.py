import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import time
import os
import numpy as np

CSV_FILEPATH = "/home/nmail-njh/NMAIL/01_Project/Robot_Grasping/IsaacLab/tracking_data.csv"
UPDATE_INTERVAL_S = 0.1 # 그래프 업데이트 주기 (초)
GRAPH_RANGE = 0.5
PVIEW_MARGIN_H = 0.3 # 가로 반경 (+/- 0.3)
PVIEW_MARGIN_V = 0.2 # 세로 반경 (+/- 0.2)

DISTANCE_TARGET = 0.30
DISTANCE_MIN = 0.25
DISTANCE_MAX = 0.35

def main():
    """CSV 파일을 주기적으로 읽어 실시간으로 그래프를 업데이트합니다."""
    
    plt.ion() 

    # fig1 = plt.figure("3D Trajectory", figsize=(8, 7))
    # ax1 = fig1.add_subplot(111, projection='3d')
    # traj_obj_line, = ax1.plot([], [], [], label='Object Trajectory', color='blue', marker='.')
    # traj_grip_line, = ax1.plot([], [], [], label='Gripper Trajectory', color='red', linestyle='--', marker='.')
    # ax1.set_title('Real-time 3D Trajectory')
    # ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    # ax1.legend(); ax1.grid(True)
    
    fig2, ax2 = plt.subplots(figsize=(7, 7), num="Camera View")
    cam_path_line, = ax2.plot([], [], color='blue', alpha=0.6, linewidth=2, label='Past Trajectory')
    cam_current_dot, = ax2.plot([], [], color='red', marker='o', markersize=10, label='Current Position')
    ax2.axhline(0, color='black', linestyle='--', linewidth=1)
    ax2.axvline(0, color='black', linestyle='--', linewidth=1)
    margin_rect = Rectangle((-PVIEW_MARGIN_H, -PVIEW_MARGIN_V), 2 * PVIEW_MARGIN_H, 2 * PVIEW_MARGIN_V,
                              color='red', fill=False, linestyle='-.', linewidth=2, label='pview_margin')
    ax2.add_artist(margin_rect)
    ax2.set_title('Real-time Object Position in Camera Frame')
    ax2.set_xlabel('Horizontal View (Camera Y-axis)')
    ax2.set_ylabel('Vertical View (Camera X-axis)')
    ax2.set_aspect('equal', adjustable='box')
    ax2.set_xlim(-GRAPH_RANGE, GRAPH_RANGE)
    ax2.set_ylim(-GRAPH_RANGE, GRAPH_RANGE)
    ax2.legend(); ax2.grid(True)

    fig3, ax3 = plt.subplots(figsize=(10, 4), num="Distance Plot")
    distance_line, = ax3.plot([], [], color='green', label='Gripper-Object Distance')
    ax3.set_title('Real-time Distance')
    ax3.set_xlabel('Time (steps)')
    ax3.set_ylabel('Distance (m)')
    ax3.grid(True)
    
    ax3.axhline(y=DISTANCE_TARGET, color='g', linestyle='-', linewidth=2, label=f'Target ({DISTANCE_TARGET}m)')
    ax3.axhline(y=DISTANCE_MIN, color='r', linestyle='--', linewidth=1, label=f'Margin ({DISTANCE_MIN}m & {DISTANCE_MAX}m)')
    ax3.axhline(y=DISTANCE_MAX, color='r', linestyle='--', linewidth=1)
    ax3.legend()
    
    plt.show(block=False)

    print("실시간 플로팅 시작... (Ctrl+C to exit)")
    
    try:
        while True:
            try:
                data = pd.read_csv(CSV_FILEPATH)
                if len(data) < 1:
                    time.sleep(UPDATE_INTERVAL_S)
                    continue
            except (FileNotFoundError, pd.errors.EmptyDataError):
                print(f"'{CSV_FILEPATH}' 파일을 기다리는 중...", end='\r')
                time.sleep(UPDATE_INTERVAL_S)
                continue

            # traj_obj_line.set_data(data['object_x'], data['object_y'])
            # traj_obj_line.set_3d_properties(data['object_z'])
            # traj_grip_line.set_data(data['gripper_x'], data['gripper_y'])
            # traj_grip_line.set_3d_properties(data['gripper_z'])
            # ax1.relim(); ax1.autoscale_view(True, True, True)

            plot_data = data[['cam_y', 'cam_x']].values
            cam_path_line.set_data(plot_data[:, 0], plot_data[:, 1])
            cam_current_dot.set_data([plot_data[-1, 0]], [plot_data[-1, 1]])

            distance_data = data['distance'].values
            time_steps = np.arange(len(distance_data))
            distance_line.set_data(time_steps, distance_data)
            ax3.relim(); ax3.autoscale_view(True, True, True)

            fig2.canvas.draw()
            fig3.canvas.draw()
            
            plt.pause(UPDATE_INTERVAL_S)

    except KeyboardInterrupt:
        print("\n플로팅 종료.")

if __name__ == "__main__":
    main()