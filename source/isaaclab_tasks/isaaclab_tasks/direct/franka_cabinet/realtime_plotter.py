import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import time
import os
import numpy as np

# --- 설정 ---
CSV_FILEPATH = "/home/nmail-njh/NMAIL/01_Project/Robot_Grasping/IsaacLab/tracking_data.csv"
UPDATE_INTERVAL_S = 0.1   # 업데이트 주기
GRAPH_RANGE = 0.4         # 그래프 표시 범위 (m)

# Env 파일의 Level 설정에 맞춘 마진값
PVIEW_MARGIN_RADIUS = 0.15 

# Env 파일의 Target Distance에 맞춤
DISTANCE_TARGET = 0.40 
DISTANCE_TOLERANCE = 0.05 

# [추가] 사용자가 요청한 거리 범위 가이드라인
DISTANCE_GUIDE_MIN = 0.35
DISTANCE_GUIDE_MAX = 0.45

def main():
    """CSV 파일을 읽어 카메라 뷰와 거리를 실시간으로 시각화합니다 (전체 경로 유지 + 거리 범위 표시)."""
    
    plt.ion() 
    
    # --- Figure 2: Camera View (X-Y Plane) ---
    fig2, ax2 = plt.subplots(figsize=(6, 6), num="Camera View (Robot Eye)")
    
    # 경로(Trace)와 현재 위치(Dot)
    cam_path_line, = ax2.plot([], [], color='blue', alpha=0.5, linewidth=1.5, label='Trace')
    cam_current_dot, = ax2.plot([], [], color='red', marker='o', markersize=8, label='Current Object')
    
    # 중심선
    ax2.axhline(0, color='black', linestyle=':', linewidth=1, alpha=0.5)
    ax2.axvline(0, color='black', linestyle=':', linewidth=1, alpha=0.5)
    
    # 마진 표시 (원형)
    margin_circle = Circle((0, 0), PVIEW_MARGIN_RADIUS, 
                           color='green', fill=False, linestyle='--', linewidth=2, label=f'Margin (r={PVIEW_MARGIN_RADIUS})')
    ax2.add_artist(margin_circle)
    
    # 축 설정 (ROS Convention: X=Right, Y=Down)
    ax2.set_title('Object Position in Camera Frame\n(ROS: X=Right, Y=Down)')
    ax2.set_xlabel('Camera X (Right +)')
    ax2.set_ylabel('Camera Y (Down +)')
    ax2.set_aspect('equal', adjustable='box')
    ax2.set_xlim(-GRAPH_RANGE, GRAPH_RANGE)
    ax2.set_ylim(-GRAPH_RANGE, GRAPH_RANGE)
    
    # [중요] 카메라 좌표계 직관성을 위해 Y축 반전 (아래가 +)
    ax2.invert_yaxis() 
    
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.5)

    # --- Figure 3: Distance Plot ---
    fig3, ax3 = plt.subplots(figsize=(8, 3), num="Distance Monitor")
    distance_line, = ax3.plot([], [], color='blue', linewidth=1.5, label='Real Distance')
    
    ax3.set_title('Distance to Object')
    ax3.set_xlabel('Time Steps')
    ax3.set_ylabel('Euclidean Dist (m)')
    ax3.grid(True, linestyle='--', alpha=0.5)
    
    # [기존] 타겟 및 허용 범위 표시
    ax3.axhline(y=DISTANCE_TARGET, color='green', linestyle='-', linewidth=2, label=f'Target ({DISTANCE_TARGET}m)')
    ax3.fill_between([-1e5, 1e5], 
                     DISTANCE_TARGET - DISTANCE_TOLERANCE, 
                     DISTANCE_TARGET + DISTANCE_TOLERANCE, 
                     color='green', alpha=0.1)

    # [추가] 0.3m ~ 0.5m 범위 선 표시 (오렌지색 점선)
    ax3.axhline(y=DISTANCE_GUIDE_MIN, color='orange', linestyle='--', linewidth=1.5, label=f'Bound ({DISTANCE_GUIDE_MIN}m)')
    ax3.axhline(y=DISTANCE_GUIDE_MAX, color='orange', linestyle='--', linewidth=1.5, label=f'Bound ({DISTANCE_GUIDE_MAX}m)')

    # 시각적 효과를 위해 범위 사이를 아주 연하게 칠함
    ax3.fill_between([-1e5, 1e5], DISTANCE_GUIDE_MIN, DISTANCE_GUIDE_MAX, color='orange', alpha=0.05)
    
    ax3.set_ylim(0.0, 0.8) 
    ax3.legend(loc='upper right')

    plt.tight_layout()
    plt.show(block=False)

    print(f"실시간 플로팅 시작... 타겟 파일: {CSV_FILEPATH}")
    print("거리 범위(0.3~0.5m)가 표시됩니다. Ctrl+C를 눌러 종료하세요.")
    
    # --- Main Loop ---
    try:
        while True:
            try:
                if not os.path.exists(CSV_FILEPATH) or os.path.getsize(CSV_FILEPATH) == 0:
                    time.sleep(UPDATE_INTERVAL_S)
                    continue
                    
                try:
                    # 전체 경로 유지를 위해 전체 데이터 로드
                    data = pd.read_csv(CSV_FILEPATH)
                except pd.errors.EmptyDataError:
                    continue

                if len(data) < 1:
                    continue

            except Exception as e:
                print(f"데이터 읽기 오류: {e}", end='\r')
                time.sleep(UPDATE_INTERVAL_S)
                continue

            # --- Update Camera View ---
            if 'cam_x' in data.columns and 'cam_y' in data.columns:
                cam_x = data['cam_x'].values
                cam_y = data['cam_y'].values
                
                cam_path_line.set_data(cam_x, cam_y)
                cam_current_dot.set_data([cam_x[-1]], [cam_y[-1]])
            
            # --- Update Distance Plot ---
            if 'distance' in data.columns:
                dist_data = data['distance'].values
                steps = np.arange(len(dist_data))
                
                distance_line.set_data(steps, dist_data)
                ax3.set_xlim(0, len(dist_data) + 10)

            # 렌더링
            fig2.canvas.draw()
            fig2.canvas.flush_events()
            fig3.canvas.draw()
            fig3.canvas.flush_events()
            
            time.sleep(UPDATE_INTERVAL_S)

    except KeyboardInterrupt:
        print("\n플로팅 종료.")
        plt.close('all')

if __name__ == "__main__":
    main()