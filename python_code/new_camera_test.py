#!/usr/bin/env python3

import cv2
import numpy as np
import time
from collections import deque

class StrictColorNavigator:
    def __init__(self):
        print("🚢 Initializing Strict Color Navigator...")
        
        # 웹캠 설정
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            print("❌ 웹캠을 열 수 없습니다!")
            exit(1)
        
        # 해상도 설정
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # 매우 엄격한 색상 범위 설정
        self.setup_strict_color_ranges()
        
        # 네비게이션 상태
        self.cone_history = {
            'green': deque(maxlen=5),
            'red': deque(maxlen=5)
        }
        
        # 공간적 안정화를 위한 고정 영역
        self.stable_zones = {
            'green': None,  # {'center': (x, y), 'radius': r, 'confidence': n}
            'red': None
        }
        
        self.navigation_active = False
        self.target_path = None
        
        # 화면 설정
        self.display_width = 640
        self.display_height = 480
        
        print("✅ Navigator initialized with STRICT color detection!")
        print("📹 Controls:")
        print("   [S] - Start/Stop navigation")
        print("   [R] - Reset")
        print("   [Q] - Quit")
        print("   [C] - Color calibration mode")
        print("   [1] - Use preset 1 (bright colors)")
        print("   [2] - Use preset 2 (normal colors)")
        print("   [3] - Use preset 3 (dark colors)")

    def setup_strict_color_ranges(self):
        """매우 엄격한 색상 범위 설정"""
        
        # 초록색: 연두~진녹색 모든 초록 계열 포함
        self.green_lower = np.array([30, 40, 40])    # 연두색부터 포함
        self.green_upper = np.array([90, 255, 255])  # 진한 녹색까지 포함
        
        # 빨간색: 범위 축소 (더 정확하게)
        self.red_lower1 = np.array([0, 100, 100])    # 채도와 명도 다시 높임
        self.red_upper1 = np.array([12, 255, 255])   # 색상 범위 축소
        self.red_lower2 = np.array([168, 100, 100])  # 채도와 명도 다시 높임
        self.red_upper2 = np.array([180, 255, 255])  # 색상 범위 축소
        
        # 현재 사용중인 프리셋
        self.current_preset = 1
        
        print("🎨 현재 색상 설정: STRICT (순수 색상만 검출)")
        print(f"   초록색 범위: H[{self.green_lower[0]}-{self.green_upper[0]}] S[{self.green_lower[1]}-255] V[{self.green_lower[2]}-255]")
        print(f"   빨간색 범위: H[0-{self.red_upper1[0]}|{self.red_lower2[0]}-180] S[{self.red_lower1[1]}-255] V[{self.red_lower1[2]}-255]")

    def set_color_preset(self, preset_num):
        """색상 프리셋 변경"""
        if preset_num == 1:  # 밝고 선명한 색상용
            self.green_lower = np.array([45, 80, 80])    # 확장된 범위
            self.green_upper = np.array([75, 255, 255])  # 확장된 범위
            self.red_lower1 = np.array([0, 120, 120])
            self.red_upper1 = np.array([10, 255, 255])
            self.red_lower2 = np.array([170, 120, 120])
            self.red_upper2 = np.array([180, 255, 255])
            print("🎨 프리셋 1: 밝고 선명한 색상")
            
        elif preset_num == 2:  # 일반 색상용
            self.green_lower = np.array([40, 70, 70])    # 조금 더 관대하게
            self.green_upper = np.array([80, 255, 255])  # 범위 확장
            self.red_lower1 = np.array([0, 100, 100])
            self.red_upper1 = np.array([12, 255, 255])
            self.red_lower2 = np.array([168, 100, 100])
            self.red_upper2 = np.array([180, 255, 255])
            print("🎨 프리셋 2: 일반 색상")
            
        elif preset_num == 3:  # 어두운 색상용
            self.green_lower = np.array([40, 60, 60])
            self.green_upper = np.array([80, 255, 200])
            self.red_lower1 = np.array([0, 80, 80])
            self.red_upper1 = np.array([15, 255, 200])
            self.red_lower2 = np.array([165, 80, 80])
            self.red_upper2 = np.array([180, 255, 200])
            print("🎨 프리셋 3: 어두운 색상")
        
        self.current_preset = preset_num

    def detect_cones(self, color_image, color_type):
        """매우 엄격한 콘 검출"""
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        
        # 가우시안 블러로 노이즈 제거 (색상 검출 전)
        hsv = cv2.GaussianBlur(hsv, (5, 5), 0)
        
        # 색상 마스크 생성
        if color_type == 'green':
            mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
        else:  # red
            mask1 = cv2.inRange(hsv, self.red_lower1, self.red_upper1)
            mask2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)
            mask = cv2.bitwise_or(mask1, mask2)
        
        # 더 강력한 노이즈 제거
        kernel_small = np.ones((3,3), np.uint8)
        kernel_large = np.ones((7,7), np.uint8)
        
        # 1차: 작은 노이즈 제거
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=2)
        # 2차: 구멍 메우기
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large, iterations=1)
        # 3차: 경계 다듬기
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
        
        # 컨투어 찾기
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        cones = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # 더 엄격한 면적 필터
            if area > 800:  # 최소 면적 증가 (작은 노이즈 제거)
                x, y, w, h = cv2.boundingRect(contour)
                
                # 더 엄격한 형태 필터
                aspect_ratio = h / w if w > 0 else 0
                if 0.8 < aspect_ratio < 3.0:  # 더 좁은 종횡비 범위
                    
                    # 컨투어의 복잡도 검사 (더 매끄러운 형태만)
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        if circularity > 0.3:  # 너무 복잡한 모양 제외
                            
                            center_x = x + w // 2
                            center_y = y + h // 2
                            
                            # HSV 값 재검증 (중심점 기준)
                            if self.verify_color_at_point(hsv, center_x, center_y, color_type):
                                
                                estimated_distance = max(1.0, 8000.0 / area)
                                
                                cone_info = {
                                    'color': color_type,
                                    'pixel_pos': (center_x, center_y),
                                    'distance': estimated_distance,
                                    'bbox': (x, y, w, h),
                                    'area': area,
                                    'circularity': circularity,
                                    'aspect_ratio': aspect_ratio
                                }
                                cones.append(cone_info)
        
        return cones

    def verify_color_at_point(self, hsv, x, y, color_type):
        """특정 점에서 색상 재검증"""
        if 0 <= x < hsv.shape[1] and 0 <= y < hsv.shape[0]:
            h, s, v = hsv[y, x]
            
            if color_type == 'green':
                return (self.green_lower[0] <= h <= self.green_upper[0] and
                        s >= self.green_lower[1] and v >= self.green_lower[2])
            else:  # red
                return ((self.red_lower1[0] <= h <= self.red_upper1[0] or
                        self.red_lower2[0] <= h <= self.red_upper2[0]) and
                        s >= self.red_lower1[1] and v >= self.red_lower1[2])
        return False

    def get_best_cone(self, cones):
        """최적의 콘 선택 (공간적 안정화 포함)"""
        if not cones:
            return None
        
        def score_cone(cone):
            # 기본 점수 계산
            area_score = min(cone['area'] / 5000.0, 1.0)
            center_x = cone['pixel_pos'][0]
            center_distance = abs(center_x - self.display_width // 2)
            center_score = max(0, 1 - center_distance / (self.display_width // 2))
            shape_score = cone['circularity']
            aspect_score = 1.0 - abs(cone['aspect_ratio'] - 1.5) / 1.5
            
            base_score = (area_score * 0.4 + center_score * 0.3 + 
                         shape_score * 0.2 + aspect_score * 0.1)
            
            # 안정화 영역 보너스
            stable_zone = self.stable_zones.get(cone['color'])
            if stable_zone:
                cone_pos = cone['pixel_pos']
                zone_center = stable_zone['center']
                distance = np.sqrt((cone_pos[0] - zone_center[0])**2 + 
                                 (cone_pos[1] - zone_center[1])**2)
                
                if distance <= stable_zone['radius']:
                    # 안정화 영역 안에 있으면 큰 보너스
                    stability_bonus = 0.5 * (stable_zone['confidence'] / 10.0)
                    return base_score + stability_bonus
            
            return base_score
        
        return max(cones, key=score_cone)

    def update_stable_zone(self, cone, color):
        """안정화 영역 업데이트"""
        cone_pos = cone['pixel_pos']
        
        if self.stable_zones[color] is None:
            # 새로운 안정화 영역 생성
            self.stable_zones[color] = {
                'center': cone_pos,
                'radius': 50,  # 50픽셀 반경
                'confidence': 1
            }
        else:
            stable_zone = self.stable_zones[color]
            zone_center = stable_zone['center']
            distance = np.sqrt((cone_pos[0] - zone_center[0])**2 + 
                             (cone_pos[1] - zone_center[1])**2)
            
            if distance <= stable_zone['radius']:
                # 안정화 영역 안에서 검출됨 - 신뢰도 증가
                stable_zone['confidence'] = min(stable_zone['confidence'] + 1, 20)
                
                # 중심점을 서서히 조정 (가중 평균)
                weight = 0.1  # 10%만 새 위치 반영
                stable_zone['center'] = (
                    int(zone_center[0] * (1 - weight) + cone_pos[0] * weight),
                    int(zone_center[1] * (1 - weight) + cone_pos[1] * weight)
                )
            else:
                # 안정화 영역에서 벗어남 - 신뢰도 감소
                stable_zone['confidence'] = max(stable_zone['confidence'] - 2, 0)
                
                if stable_zone['confidence'] <= 0:
                    # 신뢰도가 0이 되면 새로운 영역으로 이동
                    stable_zone['center'] = cone_pos
                    stable_zone['confidence'] = 1

    def get_stable_cone_position(self, color):
        """안정화된 콘 위치 (공간적 안정화 적용)"""
        if not self.cone_history[color]:
            return None
        
        recent_cones = list(self.cone_history[color])
        stable_zone = self.stable_zones.get(color)
        
        if stable_zone and stable_zone['confidence'] >= 3:
            # 안정화 영역이 충분히 신뢰할 만하면 고정 위치 사용
            zone_center = stable_zone['center']
            
            # 최근 콘의 거리 정보는 사용
            avg_distance = sum(cone['distance'] for cone in recent_cones) / len(recent_cones)
            avg_area = sum(cone['area'] for cone in recent_cones) / len(recent_cones)
            avg_circularity = sum(cone['circularity'] for cone in recent_cones) / len(recent_cones)
            
            return {
                'color': color,
                'pixel_pos': zone_center,  # 안정화된 위치 사용
                'distance': avg_distance,
                'bbox': recent_cones[-1]['bbox'],
                'area': avg_area,
                'circularity': avg_circularity
            }
        else:
            # 일반적인 가중 평균 사용
            weights = [i+1 for i in range(len(recent_cones))]
            total_weight = sum(weights)
            
            avg_pixel_x = sum(cone['pixel_pos'][0] * w for cone, w in zip(recent_cones, weights)) / total_weight
            avg_pixel_y = sum(cone['pixel_pos'][1] * w for cone, w in zip(recent_cones, weights)) / total_weight
            avg_distance = sum(cone['distance'] * w for cone, w in zip(recent_cones, weights)) / total_weight
            avg_area = sum(cone['area'] * w for cone, w in zip(recent_cones, weights)) / total_weight
            avg_circularity = sum(cone['circularity'] * w for cone, w in zip(recent_cones, weights)) / total_weight
            
            return {
                'color': color,
                'pixel_pos': (int(avg_pixel_x), int(avg_pixel_y)),
                'distance': avg_distance,
                'bbox': recent_cones[-1]['bbox'],
                'area': avg_area,
                'circularity': avg_circularity
            }

    def calculate_navigation_path(self, green_cone, red_cone):
        """경로 계산"""
        green_pixel = green_cone['pixel_pos']
        red_pixel = red_cone['pixel_pos']
        
        mid_pixel_x = (green_pixel[0] + red_pixel[0]) // 2
        mid_pixel_y = (green_pixel[1] + red_pixel[1]) // 2
        
        path_width_pixels = abs(red_pixel[0] - green_pixel[0])
        estimated_path_width = path_width_pixels * 0.01
        
        avg_distance = (green_cone['distance'] + red_cone['distance']) / 2
        
        self.target_path = {
            'center_pixel': (mid_pixel_x, mid_pixel_y),
            'width': estimated_path_width,
            'distance': avg_distance,
            'width_pixels': path_width_pixels
        }

    def draw_navigation_display(self, image, green_cone, red_cone):
        """네비게이션 디스플레이 그리기 - 둘 다 있을 때만 경로 표시"""
        # 콘 개별 표시 (각각 독립적으로)
        if green_cone:
            self.draw_cone(image, green_cone, (0, 255, 0))
        if red_cone:
            self.draw_cone(image, red_cone, (0, 0, 255))
        
        # 경로 표시 - 둘 다 있을 때만!
        if green_cone and red_cone and self.target_path:
            self.draw_path_overlay(image, green_cone, red_cone)
        else:
            # 둘 중 하나라도 없으면 검색 상태 표시
            self.draw_search_status(image)
        
        self.draw_ui_elements(image)

    def draw_cone(self, image, cone, color):
        """콘 그리기 (품질 정보 및 안정화 영역 포함)"""
        x, y, w, h = cone['bbox']
        center_x, center_y = cone['pixel_pos']
        distance = cone['distance']
        
        # 바운딩 박스
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        
        # 중심점
        cv2.circle(image, (center_x, center_y), 8, color, -1)
        cv2.circle(image, (center_x, center_y), 12, color, 2)
        
        # 안정화 영역 표시
        stable_zone = self.stable_zones.get(cone['color'])
        if stable_zone and stable_zone['confidence'] >= 3:
            zone_color = tuple(int(c * 0.3) for c in color)  # 더 어둡게
            cv2.circle(image, stable_zone['center'], stable_zone['radius'], zone_color, 1)
            cv2.putText(image, f"STABLE({stable_zone['confidence']})", 
                       (stable_zone['center'][0] - 30, stable_zone['center'][1] + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, zone_color, 1)
        
        # 정보 텍스트
        text = f"{cone['color']}: ~{distance:.1f}m"
        cv2.putText(image, text, (x, y-30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # 품질 정보
        quality_text = f"A:{int(cone['area'])} C:{cone['circularity']:.2f}"
        cv2.putText(image, quality_text, (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

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
        
        # 화살표
        cv2.arrowedLine(image, screen_center, center_pixel, 
                       (0, 255, 255), 4, tipLength=0.2)
        
        # 편차 계산
        deviation = center_pixel[0] - screen_center[0]
        deviation_distance = abs(deviation)
        
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
        
        path_info = f"Path: ~{self.target_path['width']:.1f}m wide, ~{self.target_path['distance']:.1f}m away"
        cv2.putText(image, path_info, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        deviation_text = f"Deviation: {deviation:+d}px"
        cv2.putText(image, deviation_text, (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

    def draw_search_status(self, image):
        """검색 상태 표시"""
        cv2.putText(image, "SEARCHING FOR PURE COLOR CONE PAIR...", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        center = (self.display_width // 2, self.display_height // 2)
        cv2.line(image, (center[0]-30, center[1]), (center[0]+30, center[1]), (255, 255, 255), 2)
        cv2.line(image, (center[0], center[1]-30), (center[0], center[1]+30), (255, 255, 255), 2)

    def draw_ui_elements(self, image):
        """UI 요소 그리기"""
        nav_status = f"NAVIGATION: {'ACTIVE' if self.navigation_active else 'PAUSED'}"
        color = (0, 255, 0) if self.navigation_active else (128, 128, 128)
        cv2.putText(image, nav_status, (10, self.display_height - 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        preset_info = f"Color Preset: {self.current_preset} (Press 1,2,3 to change)"
        cv2.putText(image, preset_info, (10, self.display_height - 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.putText(image, "Controls: [S]tart/Stop [R]eset [C]alibrate [1,2,3]Presets [Q]uit", 
                   (10, self.display_height - 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.putText(image, "STRICT Color Mode - Pure colors only", 
                   (10, self.display_height - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        center = (self.display_width // 2, self.display_height // 2)
        cv2.line(image, (center[0]-10, center[1]), (center[0]+10, center[1]), (128, 128, 128), 1)
        cv2.line(image, (center[0], center[1]-10), (center[0], center[1]+10), (128, 128, 128), 1)

    def color_calibration_mode(self, image):
        """색상 캘리브레이션 모드"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        green_mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
        red_mask1 = cv2.inRange(hsv, self.red_lower1, self.red_upper1)
        red_mask2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        # 마스크 시각화
        result = image.copy()
        result[green_mask > 0] = [0, 255, 0]  # 초록색 영역을 순수 초록으로
        result[red_mask > 0] = [0, 0, 255]    # 빨간색 영역을 순수 빨강으로
        
        # 블렌딩
        result = cv2.addWeighted(image, 0.5, result, 0.5, 0)
        
        cv2.putText(result, "STRICT COLOR CALIBRATION MODE", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(result, f"Preset {self.current_preset} - Only pure colors highlighted", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(result, "Press [1,2,3] to change presets, [C] to exit", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return result

    def run(self):
        """메인 실행 루프"""
        frame_count = 0
        fps_start = time.time()
        calibration_mode = False
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ 프레임을 읽을 수 없습니다!")
                    break
                
                frame = cv2.resize(frame, (self.display_width, self.display_height))
                
                if calibration_mode:
                    display_image = self.color_calibration_mode(frame)
                else:
                    green_cones = self.detect_cones(frame, 'green')
                    red_cones = self.detect_cones(frame, 'red')
                    
                    best_green = self.get_best_cone(green_cones)
                    best_red = self.get_best_cone(red_cones)
                    
                    # 현재 프레임에서 콘이 검출되면 히스토리에 추가
                    if best_green:
                        self.cone_history['green'].append(best_green)
                        self.update_stable_zone(best_green, 'green')  # 안정화 영역 업데이트
                    else:
                        # 검출되지 않으면 히스토리 클리어 (즉시 사라지게)
                        self.cone_history['green'].clear()
                        if self.stable_zones['green']:
                            # 안정화 영역 신뢰도 감소
                            self.stable_zones['green']['confidence'] = max(
                                self.stable_zones['green']['confidence'] - 3, 0
                            )
                            if self.stable_zones['green']['confidence'] <= 0:
                                self.stable_zones['green'] = None
                    
                    if best_red:
                        self.cone_history['red'].append(best_red)
                        self.update_stable_zone(best_red, 'red')  # 안정화 영역 업데이트
                    else:
                        # 검출되지 않으면 히스토리 클리어 (즉시 사라지게)
                        self.cone_history['red'].clear()
                        if self.stable_zones['red']:
                            # 안정화 영역 신뢰도 감소
                            self.stable_zones['red']['confidence'] = max(
                                self.stable_zones['red']['confidence'] - 3, 0
                            )
                            if self.stable_zones['red']['confidence'] <= 0:
                                self.stable_zones['red'] = None
                    
                    # 안정화된 위치 (히스토리가 비어있으면 None 반환)
                    stable_green = self.get_stable_cone_position('green')
                    stable_red = self.get_stable_cone_position('red')
                    
                    # 둘 다 있을 때만 경로 계산
                    if stable_green and stable_red:
                        self.calculate_navigation_path(stable_green, stable_red)
                    else:
                        # 하나라도 없으면 경로 정보 클리어
                        self.target_path = None
                    
                    display_image = frame.copy()
                    self.draw_navigation_display(display_image, stable_green, stable_red)
                
                cv2.imshow('Strict Color Cone Navigation', display_image)
                
                frame_count += 1
                if frame_count % 30 == 0:
                    current_time = time.time()
                    fps = 30 / (current_time - fps_start)
                    print(f"📊 FPS: {fps:.1f}")
                    fps_start = current_time
                
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
                elif key == ord('1'):
                    self.set_color_preset(1)
                elif key == ord('2'):
                    self.set_color_preset(2)
                elif key == ord('3'):
                    self.set_color_preset(3)
                
        except KeyboardInterrupt:
            print("🛑 Interrupted by user")
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            print("✅ Navigator stopped")


def main():
    print("🚢 Starting Strict Color Cone Navigator...")
    print("📋 Features:")
    print("- Very strict color detection (pure colors only)")
    print("- 3 color presets for different lighting")
    print("- Advanced noise filtering")
    print("- Shape and quality verification")
    print()
    
    navigator = StrictColorNavigator()
    navigator.run()

if __name__ == '__main__':
    main()