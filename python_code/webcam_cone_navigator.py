#!/usr/bin/env python3

import cv2
import numpy as np
import time
from collections import deque

class WebcamConeNavigator:
    def __init__(self):
        print("🚢 Initializing Webcam Cone Navigator...")
        
        # 웹캠 설정
        self.cap = cv2.VideoCapture(0)  # 기본 웹캠
        
        if not self.cap.isOpened():
            print("❌ 웹캠을 열 수 없습니다!")
            exit(1)
        
        # 해상도 설정
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # 색상 범위 설정
        self.setup_color_ranges()
        
        # 네비게이션 상태
        self.cone_history = {
            'green': deque(maxlen=5),
            'red': deque(maxlen=5)
        }
        
        self.navigation_active = False
        self.target_path = None
        
        # 화면 설정
        self.display_width = 640
        self.display_height = 480
        
        print("✅ Navigator initialized!")
        print("📹 Controls:")
        print("   [S] - Start/Stop navigation")
        print("   [R] - Reset")
        print("   [Q] - Quit")
        print("   [C] - Color calibration mode")

    def setup_color_ranges(self):
        """색상 범위 설정 (웹캠용으로 조정)"""
        # 초록색 범위 (더 관대하게)
        self.green_lower = np.array([30, 40, 40])
        self.green_upper = np.array([90, 255, 255])
        
        # 빨간색 범위
        self.red_lower1 = np.array([0, 50, 50])
        self.red_upper1 = np.array([20, 255, 255])
        self.red_lower2 = np.array([160, 50, 50])
        self.red_upper2 = np.array([180, 255, 255])

    def detect_cones(self, color_image, color_type):
        """콘 검출 (깊이 정보 없음)"""
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        
        # 색상 마스크
        if color_type == 'green':
            mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
        else:  # red
            mask1 = cv2.inRange(hsv, self.red_lower1, self.red_upper1)
            mask2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)
            mask = cv2.bitwise_or(mask1, mask2)
        
        # 노이즈 제거
        kernel = np.ones((7,7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 컨투어 찾기
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        cones = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # 최소 크기
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = h / w if w > 0 else 0
                
                # 콘 모양 필터 (더 관대하게)
                if 0.5 < aspect_ratio < 5.0:
                    center_x = x + w // 2
                    center_y = y + h // 2
                    
                    # 추정 거리 (픽셀 크기 기반)
                    estimated_distance = max(1.0, 5000.0 / area)  # 간단한 추정
                    
                    cone_info = {
                        'color': color_type,
                        'pixel_pos': (center_x, center_y),
                        'distance': estimated_distance,
                        'bbox': (x, y, w, h),
                        'area': area,
                        'mask': mask  # 디버깅용
                    }
                    cones.append(cone_info)
        
        return cones

    def get_best_cone(self, cones):
        """최적의 콘 선택"""
        if not cones:
            return None
        
        # 면적과 중앙 위치를 고려한 스코어링
        def score_cone(cone):
            area_score = min(cone['area'] / 3000.0, 1.0)
            
            # 화면 중앙에 가까울수록 높은 점수
            center_x = cone['pixel_pos'][0]
            center_distance = abs(center_x - self.display_width // 2)
            center_score = max(0, 1 - center_distance / (self.display_width // 2))
            
            return area_score * 0.7 + center_score * 0.3
        
        return max(cones, key=score_cone)

    def get_stable_cone_position(self, color):
        """안정화된 콘 위치"""
        if not self.cone_history[color]:
            return None
        
        recent_cones = list(self.cone_history[color])
        
        # 평균 계산
        avg_pixel_x = sum(cone['pixel_pos'][0] for cone in recent_cones) / len(recent_cones)
        avg_pixel_y = sum(cone['pixel_pos'][1] for cone in recent_cones) / len(recent_cones)
        avg_distance = sum(cone['distance'] for cone in recent_cones) / len(recent_cones)
        
        return {
            'color': color,
            'pixel_pos': (int(avg_pixel_x), int(avg_pixel_y)),
            'distance': avg_distance,
            'bbox': recent_cones[-1]['bbox']
        }

    def calculate_navigation_path(self, green_cone, red_cone):
        """경로 계산"""
        green_pixel = green_cone['pixel_pos']
        red_pixel = red_cone['pixel_pos']
        
        # 중점 계산
        mid_pixel_x = (green_pixel[0] + red_pixel[0]) // 2
        mid_pixel_y = (green_pixel[1] + red_pixel[1]) // 2
        
        # 경로 폭 (픽셀 거리)
        path_width_pixels = abs(red_pixel[0] - green_pixel[0])
        estimated_path_width = path_width_pixels * 0.01  # 간단한 추정
        
        avg_distance = (green_cone['distance'] + red_cone['distance']) / 2
        
        self.target_path = {
            'center_pixel': (mid_pixel_x, mid_pixel_y),
            'width': estimated_path_width,
            'distance': avg_distance,
            'width_pixels': path_width_pixels
        }

    def draw_navigation_display(self, image, green_cone, red_cone):
        """네비게이션 디스플레이 그리기"""
        # 콘 표시
        if green_cone:
            self.draw_cone(image, green_cone, (0, 255, 0))
        if red_cone:
            self.draw_cone(image, red_cone, (0, 0, 255))
        
        # 경로 표시
        if green_cone and red_cone and self.target_path:
            self.draw_path_overlay(image, green_cone, red_cone)
        else:
            self.draw_search_status(image)
        
        # UI 정보
        self.draw_ui_elements(image)

    def draw_cone(self, image, cone, color):
        """콘 그리기"""
        x, y, w, h = cone['bbox']
        center_x, center_y = cone['pixel_pos']
        distance = cone['distance']
        
        # 바운딩 박스
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        
        # 중심점
        cv2.circle(image, (center_x, center_y), 8, color, -1)
        cv2.circle(image, (center_x, center_y), 12, color, 2)
        
        # 거리 정보
        text = f"{cone['color']}: ~{distance:.1f}m"
        cv2.putText(image, text, (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def draw_path_overlay(self, image, green_cone, red_cone):
        """경로 오버레이 그리기"""
        green_pixel = green_cone['pixel_pos']
        red_pixel = red_cone['pixel_pos']
        center_pixel = self.target_path['center_pixel']
        screen_center = (self.display_width // 2, self.display_height // 2)
        
        # 두 콘을 연결하는 선
        cv2.line(image, green_pixel, red_pixel, (255, 255, 255), 3)
        
        # 목표 지점
        cv2.circle(image, center_pixel, 15, (255, 255, 0), -1)
        cv2.circle(image, center_pixel, 20, (255, 255, 0), 3)
        cv2.putText(image, "TARGET", (center_pixel[0]-30, center_pixel[1]-25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # 화면 중앙에서 목표까지 화살표
        cv2.arrowedLine(image, screen_center, center_pixel, 
                       (0, 255, 255), 4, tipLength=0.2)
        
        # 편차 계산 및 표시
        deviation = center_pixel[0] - screen_center[0]
        deviation_distance = abs(deviation)
        
        # 편차 정보
        # 이모지 대신 일반 텍스트 사용
        if deviation_distance < 30:
            status_text = "ON TRACK"        
            status_color = (0, 255, 0)
        elif deviation_distance < 80:
            direction = "LEFT" if deviation < 0 else "RIGHT"
            status_text = f"ADJUST {direction}"  
            status_color = (0, 165, 255)
        else:
            direction = "LEFT" if deviation < 0 else "RIGHT"
            status_text = f"TURN {direction}" 
            status_color = (0, 0, 255)
        
        # 상태 표시
        cv2.putText(image, status_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        # 경로 정보
        path_info = f"Path: ~{self.target_path['width']:.1f}m wide, ~{self.target_path['distance']:.1f}m away"
        cv2.putText(image, path_info, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 편차 수치
        deviation_text = f"Deviation: {deviation:+d}px"
        cv2.putText(image, deviation_text, (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

    def draw_search_status(self, image):
        """검색 상태 표시"""
        cv2.putText(image, "🔍 SEARCHING FOR CONE PAIR...", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # 십자선
        center = (self.display_width // 2, self.display_height // 2)
        cv2.line(image, (center[0]-30, center[1]), (center[0]+30, center[1]), (255, 255, 255), 2)
        cv2.line(image, (center[0], center[1]-30), (center[0], center[1]+30), (255, 255, 255), 2)

    def draw_ui_elements(self, image):
        """UI 요소 그리기"""
        # 네비게이션 상태
        nav_status = "🎯 NAVIGATION: ACTIVE" if self.navigation_active else "⏸️ NAVIGATION: PAUSED"
        color = (0, 255, 0) if self.navigation_active else (128, 128, 128)
        cv2.putText(image, nav_status, (10, self.display_height - 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # 컨트롤
        cv2.putText(image, "Controls: [S]tart/Stop [R]eset [C]alibrate [Q]uit", 
                   (10, self.display_height - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # 카메라 타입
        cv2.putText(image, "📹 Webcam Mode (No Depth)", (10, self.display_height - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        # 화면 중앙 십자선
        center = (self.display_width // 2, self.display_height // 2)
        cv2.line(image, (center[0]-10, center[1]), (center[0]+10, center[1]), (128, 128, 128), 1)
        cv2.line(image, (center[0], center[1]-10), (center[0], center[1]+10), (128, 128, 128), 1)

    def color_calibration_mode(self, image):
        """색상 캘리브레이션 모드"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 초록색과 빨간색 마스크 표시
        green_mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
        red_mask1 = cv2.inRange(hsv, self.red_lower1, self.red_upper1)
        red_mask2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        # 마스크를 컬러로 변환
        green_colored = cv2.applyColorMap(green_mask, cv2.COLORMAP_GREEN)
        red_colored = cv2.applyColorMap(red_mask, cv2.COLORMAP_HOT)
        
        # 원본 이미지와 블렌딩
        result = cv2.addWeighted(image, 0.7, green_colored, 0.3, 0)
        result = cv2.addWeighted(result, 0.7, red_colored, 0.3, 0)
        
        # 캘리브레이션 정보 표시
        cv2.putText(result, "COLOR CALIBRATION MODE", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(result, "Green areas highlighted, Red areas highlighted", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(result, "Press [C] again to exit calibration", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return result

    def run(self):
        """메인 실행 루프"""
        frame_count = 0
        fps_start = time.time()
        calibration_mode = False
        
        try:
            while True:
                # 프레임 읽기
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ 프레임을 읽을 수 없습니다!")
                    break
                
                # 프레임 크기 조정
                frame = cv2.resize(frame, (self.display_width, self.display_height))
                
                if calibration_mode:
                    # 캘리브레이션 모드
                    display_image = self.color_calibration_mode(frame)
                else:
                    # 일반 네비게이션 모드
                    # 콘 검출
                    green_cones = self.detect_cones(frame, 'green')
                    red_cones = self.detect_cones(frame, 'red')
                    
                    # 최적 콘 선택
                    best_green = self.get_best_cone(green_cones)
                    best_red = self.get_best_cone(red_cones)
                    
                    # 히스토리 업데이트
                    if best_green:
                        self.cone_history['green'].append(best_green)
                    if best_red:
                        self.cone_history['red'].append(best_red)
                    
                    # 안정화된 위치
                    stable_green = self.get_stable_cone_position('green')
                    stable_red = self.get_stable_cone_position('red')
                    
                    # 경로 계산
                    if stable_green and stable_red:
                        self.calculate_navigation_path(stable_green, stable_red)
                    
                    # 디스플레이 업데이트
                    display_image = frame.copy()
                    self.draw_navigation_display(display_image, stable_green, stable_red)
                
                # 화면 표시
                cv2.imshow('🚢 Webcam Cone Navigation', display_image)
                
                # FPS 계산
                frame_count += 1
                if frame_count % 30 == 0:
                    current_time = time.time()
                    fps = 30 / (current_time - fps_start)
                    print(f"📊 FPS: {fps:.1f}")
                    fps_start = current_time
                
                # 키보드 입력 처리
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("🛑 Quitting...")
                    break
                elif key == ord('s'):
                    self.navigation_active = not self.navigation_active
                    status = "ACTIVE" if self.navigation_active else "PAUSED"
                    print(f"🎯 Navigation: {status}")
                elif key == ord('r'):
                    self.cone_history['green'].clear()
                    self.cone_history['red'].clear()
                    self.target_path = None
                    self.navigation_active = False
                    print("🔄 Navigation reset")
                elif key == ord('c'):
                    calibration_mode = not calibration_mode
                    mode = "ENABLED" if calibration_mode else "DISABLED"
                    print(f"🎨 Calibration mode: {mode}")
                
        except KeyboardInterrupt:
            print("🛑 Interrupted by user")
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            print("✅ Navigator stopped")


def main():
    print("🚢 Starting Webcam Cone Navigator...")
    print("📋 Setup Instructions:")
    print("1. Place green and red objects (cones, bottles, etc.) in camera view")
    print("2. Use [C] key for color calibration if detection is poor")
    print("3. Use [S] key to start navigation")
    print()
    
    navigator = WebcamConeNavigator()
    navigator.run()

if __name__ == '__main__':
    main()