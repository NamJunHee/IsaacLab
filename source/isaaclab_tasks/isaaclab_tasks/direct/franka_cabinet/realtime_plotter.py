import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle  # <<-- Circle 대신 Rectangle을 import 합니다.
import time
import os
import numpy as np

# --- 설정 ---
CSV_FILEPATH = "/home/nmail-njh/NMAIL/01_Project/Robot_Grasping/IsaacLab/tracking_data.csv"
UPDATE_INTERVAL_S = 0.5 # 그래프 업데이트 주기 (초)
GRAPH_RANGE = 0.5

# <<-- (수정) 원형 마진 대신 사각형 마진의 가로/세로 반경을 정의합니다. -->>
PVIEW_MARGIN_H = 0.3 # 가로 반경 (+/- 0.3)
PVIEW_MARGIN_V = 0.2 # 세로 반경 (+/- 0.2)
# ------------

def main():
    """CSV 파일을 주기적으로 읽어 실시간으로 그래프를 업데이트합니다."""
    
    if os.path.exists(CSV_FILEPATH):
        os.remove(CSV_FILEPATH)
        print(f"'{CSV_FILEPATH}' 파일을 삭제하고 새로 시작합니다.")
    
    plt.ion() 

    # ... (Figure 1: 3D Trajectory 설정 코드는 기존과 동일) ...
    fig1 = plt.figure(figsize=(8, 7))
    ax1 = fig1.add_subplot(111, projection='3d')
    traj_obj_line, = ax1.plot([], [], [], label='Object Trajectory', color='blue', marker='.')
    traj_grip_line, = ax1.plot([], [], [], label='Gripper Trajectory', color='red', linestyle='--', marker='.')
    ax1.set_title('Real-time 3D Trajectory')
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    ax1.legend(); ax1.grid(True)
    
    # --- Figure 2: Camera View ---
    fig2, ax2 = plt.subplots(figsize=(7, 7))
    
    cam_path_line, = ax2.plot([], [], color='blue', alpha=0.6, linewidth=2, label='Past Trajectory')
    cam_current_dot, = ax2.plot([], [], color='red', marker='o', markersize=10, label='Current Position')

    ax2.axhline(0, color='black', linestyle='--', linewidth=1)
    ax2.axvline(0, color='black', linestyle='--', linewidth=1)
    
    # <<-- (수정) Circle 대신 Rectangle 객체를 생성하여 마진을 표시합니다. -->>
    margin_rect = Rectangle((-PVIEW_MARGIN_H, -PVIEW_MARGIN_V),  # (x, y) 좌측 하단 꼭짓점
                              2 * PVIEW_MARGIN_H,              # 가로 길이
                              2 * PVIEW_MARGIN_V,              # 세로 길이
                              color='red', fill=False, linestyle='-.', 
                              linewidth=2, label='pview_margin')
    ax2.add_artist(margin_rect)
    
    ax2.set_title('Real-time Object Position in Camera Frame')
    ax2.set_xlabel('Horizontal View (Camera Y-axis)')
    ax2.set_ylabel('Vertical View (Camera X-axis)')
    ax2.set_aspect('equal', adjustable='box')
    ax2.set_xlim(-GRAPH_RANGE, GRAPH_RANGE)
    ax2.set_ylim(-GRAPH_RANGE, GRAPH_RANGE)
    ax2.legend(); ax2.grid(True)
    
    plt.show(block=False)

    print("실시간 플로팅 시작... (Ctrl+C to exit)")
    
    try:
        while True:
            # ... (이하 CSV 파일 읽기 및 그래프 데이터 업데이트 로직은 기존과 동일) ...
            try:
                data = pd.read_csv(CSV_FILEPATH)
                if len(data) < 1:
                    time.sleep(UPDATE_INTERVAL_S)
                    continue
            except (FileNotFoundError, pd.errors.EmptyDataError):
                print(f"'{CSV_FILEPATH}' 파일을 기다리는 중...", end='\r')
                time.sleep(UPDATE_INTERVAL_S)
                continue

            traj_obj_line.set_data(data['object_x'], data['object_y'])
            traj_obj_line.set_3d_properties(data['object_z'])
            traj_grip_line.set_data(data['gripper_x'], data['gripper_y'])
            traj_grip_line.set_3d_properties(data['gripper_z'])
            ax1.relim(); ax1.autoscale_view(True, True, True)

            plot_data = data[['cam_y', 'cam_x']].values
            plot_data[:, 1] *= -1 
            
            cam_path_line.set_data(plot_data[:, 0], plot_data[:, 1])
            cam_current_dot.set_data([plot_data[-1, 0]], [plot_data[-1, 1]])

            fig1.canvas.draw()
            fig2.canvas.draw()
            
            plt.pause(UPDATE_INTERVAL_S)

    except KeyboardInterrupt:
        print("\n플로팅 종료.")
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    main()