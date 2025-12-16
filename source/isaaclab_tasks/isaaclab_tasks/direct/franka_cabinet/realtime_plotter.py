import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import time
import os
import numpy as np
import math  # [추가] 삼각함수 계산용

# --- 설정 ---
CSV_FILEPATH = "/home/nmail-njh/NMAIL/01_Project/Robot_Grasping/IsaacLab/tracking_data.csv"
UPDATE_INTERVAL_S = 0.1   # 업데이트 주기
GRAPH_RANGE = 0.4         # 그래프 표시 범위 (m)

# Env 파일의 Level 설정에 맞춘 마진값
PVIEW_MARGIN_RADIUS = 0.15 

# Env 파일의 Target Distance에 맞춤
DISTANCE_TARGET = 0.40 
DISTANCE_TOLERANCE = 0.05 

# 사용자가 요청한 거리 범위 가이드라인
DISTANCE_GUIDE_MIN = 0.35
DISTANCE_GUIDE_MAX = 0.45

def main():
    """CSV 파일을 읽어 파지 정보, 카메라 뷰, 거리를 실시간으로 시각화합니다."""
    
    plt.ion() 
    
    # ------------------------------------------------------------------
    # [추가] Figure 1: Grasping Status (Angle & Width)
    # ------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(5, 5), num="Grasping Status")
    
    # 그리퍼 (빨간색 굵은 선)
    grasp_line, = ax1.plot([], [], color='red', linewidth=6, label='Gripper', solid_capstyle='round')
    # 중심점
    ax1.scatter([0], [0], color='black', s=100, marker='+', label='Center')
    
    # 상태 텍스트 (좌측 상단)
    grasp_text = ax1.text(0.05, 0.95, 'Waiting...', transform=ax1.transAxes, 
                          verticalalignment='top', fontsize=12,
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax1.set_title('Real-time Grasping Angle & Width')
    # 그리퍼 크기에 맞춰 줌인 (약 +/- 12cm)
    ax1.set_xlim(-0.12, 0.12)
    ax1.set_ylim(-0.12, 0.12)
    ax1.set_aspect('equal')
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # 십자선 가이드
    ax1.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax1.axvline(0, color='gray', linestyle='--', linewidth=0.5)

    # ------------------------------------------------------------------
    # Figure 2: Camera View (Robot Eye) - [기존 유지]
    # ------------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(6, 6), num="Camera View (Robot Eye)")
    
    cam_path_line, = ax2.plot([], [], color='blue', alpha=0.5, linewidth=1.5, label='Trace')
    cam_current_dot, = ax2.plot([], [], color='red', marker='o', markersize=8, label='Current Object')
    
    ax2.axhline(0, color='black', linestyle=':', linewidth=1, alpha=0.5)
    ax2.axvline(0, color='black', linestyle=':', linewidth=1, alpha=0.5)
    
    margin_circle = Circle((0, 0), PVIEW_MARGIN_RADIUS, 
                           color='green', fill=False, linestyle='--', linewidth=2, label=f'Margin (r={PVIEW_MARGIN_RADIUS})')
    ax2.add_artist(margin_circle)
    
    ax2.set_xlabel('Camera X (Right +)')
    ax2.set_ylabel('Camera Y (Down +)')
    ax2.set_aspect('equal', adjustable='box')
    ax2.set_xlim(-GRAPH_RANGE, GRAPH_RANGE)
    ax2.set_ylim(-GRAPH_RANGE, GRAPH_RANGE)
    ax2.invert_yaxis() 
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.5)

    # ------------------------------------------------------------------
    # Figure 3: Distance Plot - [기존 유지]
    # ------------------------------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(8, 3), num="Distance Monitor")
    distance_line, = ax3.plot([], [], color='blue', linewidth=1.5, label='Real Distance')
    
    ax3.set_xlabel('Time Steps')
    ax3.set_ylabel('Euclidean Dist (m)')
    ax3.grid(True, linestyle='--', alpha=0.5)
    
    ax3.axhline(y=DISTANCE_TARGET, color='green', linestyle='-', linewidth=2, label=f'Target ({DISTANCE_TARGET}m)')
    ax3.fill_between([-1e5, 1e5], 
                     DISTANCE_TARGET - DISTANCE_TOLERANCE, 
                     DISTANCE_TARGET + DISTANCE_TOLERANCE, 
                     color='green', alpha=0.1)

    ax3.axhline(y=DISTANCE_GUIDE_MIN, color='orange', linestyle='--', linewidth=1.5, label=f'Bound ({DISTANCE_GUIDE_MIN}m)')
    ax3.axhline(y=DISTANCE_GUIDE_MAX, color='orange', linestyle='--', linewidth=1.5, label=f'Bound ({DISTANCE_GUIDE_MAX}m)')
    ax3.fill_between([-1e5, 1e5], DISTANCE_GUIDE_MIN, DISTANCE_GUIDE_MAX, color='orange', alpha=0.05)
    
    ax3.set_ylim(0.0, 0.8) 
    ax3.legend(loc='upper right')

    plt.tight_layout()
    plt.show(block=False)

    print(f"실시간 플로팅 시작... 타겟 파일: {CSV_FILEPATH}")
    print("Grasping 정보 및 거리 통계가 표시됩니다. Ctrl+C를 눌러 종료하세요.")
    
    # --- Main Loop ---
    try:
        while True:
            try:
                if not os.path.exists(CSV_FILEPATH) or os.path.getsize(CSV_FILEPATH) == 0:
                    time.sleep(UPDATE_INTERVAL_S)
                    continue
                    
                try:
                    data = pd.read_csv(CSV_FILEPATH, on_bad_lines='skip') # [수정] 쓰기 에러 방지
                except Exception:
                    continue

                if len(data) < 1:
                    continue

            except Exception as e:
                time.sleep(UPDATE_INTERVAL_S)
                continue

            # ------------------------------------------------------------------
            # [추가] Update Grasping View
            # ------------------------------------------------------------------
            if 'grasp_angle' in data.columns and 'grasp_width' in data.columns:
                # 가장 최신 데이터 1개만 사용 (현재 상태)
                last_row = data.iloc[-1]
                angle_deg = last_row['grasp_angle']
                width_m = last_row['grasp_width']
                
                # 좌표 계산 (중심 기준 양쪽으로 뻗어나가도록)
                theta = math.radians(angle_deg)
                half_w = width_m / 2.0
                
                dx = half_w * math.cos(theta)
                dy = half_w * math.sin(theta)
                
                # 선 그리기: (-dx, -dy) 에서 (dx, dy) 까지
                grasp_line.set_data([-dx, dx], [-dy, dy])
                
                # 텍스트 업데이트
                grasp_text.set_text(f"Angle: {angle_deg:.1f}°\nWidth: {width_m*100:.1f} cm")
                
                fig1.canvas.draw()
                fig1.canvas.flush_events()

            # ------------------------------------------------------------------
            # Update Camera View - [기존 유지]
            # ------------------------------------------------------------------
            if 'cam_x' in data.columns and 'cam_y' in data.columns:
                cam_x = data['cam_x'].values
                cam_y = data['cam_y'].values
                
                cam_error = np.sqrt(cam_x**2 + cam_y**2)
                cam_mean = np.mean(cam_error)
                cam_std = np.std(cam_error)

                cam_path_line.set_data(cam_x, cam_y)
                cam_current_dot.set_data([cam_x[-1]], [cam_y[-1]])
                
                ax2.set_title(f'Object Position\nErr Mean: {cam_mean*100:.2f}cm, Std: {cam_std*100:.2f}cm')
                
                fig2.canvas.draw()
                fig2.canvas.flush_events()
            
            # ------------------------------------------------------------------
            # Update Distance Plot - [기존 유지]
            # ------------------------------------------------------------------
            if 'distance' in data.columns:
                dist_data = data['distance'].values
                steps = np.arange(len(dist_data))
                
                dist_mean = np.mean(dist_data)
                dist_std = np.std(dist_data)

                distance_line.set_data(steps, dist_data)
                # 데이터가 많아지면 최근 데이터 위주로 스크롤 (옵션)
                if len(dist_data) > 500:
                     ax3.set_xlim(len(dist_data) - 500, len(dist_data) + 10)
                else:
                     ax3.set_xlim(0, len(dist_data) + 10)
                
                ax3.set_title(f'Distance to Object\nMean: {dist_mean*100:.2f}cm, Std: {dist_std*100:.2f}cm')

                fig3.canvas.draw()
                fig3.canvas.flush_events()
            
            time.sleep(UPDATE_INTERVAL_S)

    except KeyboardInterrupt:
        print("\n플로팅 종료.")
        plt.close('all')

if __name__ == "__main__":
    main()