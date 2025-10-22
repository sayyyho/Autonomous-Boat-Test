#!/usr/bin/env python3

import serial
import sys, termios, tty, select
import time
import numpy as np
import cv2
from collections import deque
import threading
import json
from datetime import datetime

class ConeDetector:
    """LiDAR로 꼬깔(삼각뿔) 형태 감지"""
    def __init__(self, logger_func):
        self.logger = logger_func
        self.min_cone_points = 5
        self.max_cone_width = 0.5
        self.angle_tolerance = 15
        
    def detect_cones(self, ranges, angle_min, angle_increment):
        ranges = np.array(ranges)
        valid_mask = ~(np.isinf(ranges) | np.isnan(ranges) | (ranges <= 0.1) | (ranges > 10.0))
        
        if not np.any(valid_mask):
            return []
        
        clusters = self._cluster_points(ranges, valid_mask, angle_min, angle_increment)
        
        cones = []
        for cluster in clusters:
            if self._is_cone_shaped(cluster):
                cone_info = self._compute_cone_center(cluster)
                cones.append(cone_info)
        
        return cones
    
    def _cluster_points(self, ranges, valid_mask, angle_min, angle_increment):
        clusters = []
        current_cluster = []
        
        indices = np.where(valid_mask)[0]
        
        for i, idx in enumerate(indices):
            distance = ranges[idx]
            angle = angle_min + idx * angle_increment
            
            point = {
                'index': idx,
                'distance': distance,
                'angle': np.degrees(angle),
                'angle_rad': angle
            }
            
            if not current_cluster:
                current_cluster.append(point)
            else:
                prev = current_cluster[-1]
                angle_diff = abs(point['angle'] - prev['angle'])
                dist_diff = abs(point['distance'] - prev['distance'])
                
                if angle_diff < 5 and dist_diff < 0.3:
                    current_cluster.append(point)
                else:
                    if len(current_cluster) >= self.min_cone_points:
                        clusters.append(current_cluster)
                    current_cluster = [point]
        
        if len(current_cluster) >= self.min_cone_points:
            clusters.append(current_cluster)
        
        return clusters
    
    def _is_cone_shaped(self, cluster):
        if len(cluster) < self.min_cone_points:
            return False
        
        distances = np.array([p['distance'] for p in cluster])
        angles = np.array([p['angle'] for p in cluster])
        
        min_idx = np.argmin(distances)
        is_v_shape = (min_idx > 0 and min_idx < len(distances) - 1)
        
        angle_span = abs(angles[-1] - angles[0])
        if angle_span > self.angle_tolerance:
            return False
        
        if len(cluster) >= 2:
            left = cluster[0]
            right = cluster[-1]
            
            left_x = left['distance'] * np.sin(left['angle_rad'])
            left_y = left['distance'] * np.cos(left['angle_rad'])
            right_x = right['distance'] * np.sin(right['angle_rad'])
            right_y = right['distance'] * np.cos(right['angle_rad'])
            
            width = np.sqrt((right_x - left_x)**2 + (right_y - left_y)**2)
            
            if width > self.max_cone_width or width < 0.1:
                return False
        
        return True
    
    def _compute_cone_center(self, cluster):
        angles = np.array([p['angle'] for p in cluster])
        angle_rads = np.array([p['angle_rad'] for p in cluster])
        distances = np.array([p['distance'] for p in cluster])
        
        center_angle = np.mean(angles)
        center_angle_rad = np.mean(angle_rads)
        center_distance = np.min(distances) * 0.6 + np.mean(distances) * 0.4
        
        # 3D 좌표
        x = center_distance * np.sin(center_angle_rad)
        y = center_distance * np.cos(center_angle_rad)
        z = 0.3
        
        left = cluster[0]
        right = cluster[-1]
        left_x = left['distance'] * np.sin(left['angle_rad'])
        right_x = right['distance'] * np.sin(right['angle_rad'])
        width = abs(right_x - left_x)
        
        return {
            'angle': center_angle,
            'angle_rad': center_angle_rad,
            'distance': center_distance,
            'width': width,
            'x': x,
            'y': y,
            'z': z,
            'is_cone': True,
            'point_count': len(cluster)
        }


class ColorRegionClassifier:
    """색 공간 이분법 분류기"""
    def __init__(self, logger):
        self.logger = logger
        self.hue_boundary = 90
        
        # 🕐 색상 안정화 시스템 (1.5초 유지)
        self.color_history = {}  # {angle: {'colors': deque, 'timestamps': deque}}
        self.stability_duration = 1.5  # 1.5초
        self.min_samples = 8  # 최소 샘플 수 (0.1초마다 측정하면 0.8초)
        
    def classify_region_at_angle(self, frame, target_angle, camera_fov=87):
        h, w = frame.shape[:2]
        
        normalized = (target_angle + camera_fov / 2) / camera_fov
        x_pixel = int(normalized * w)
        x_pixel = np.clip(x_pixel, 0, w - 1)
        
        x_start = max(0, x_pixel - 25)
        x_end = min(w, x_pixel + 25)
        y_start = h // 4
        y_end = 3 * h // 4
        
        roi = frame[y_start:y_end, x_start:x_end]
        
        if roi.size == 0:
            return self._get_stable_color(target_angle, 'UNKNOWN')
        
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0]
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        
        # 더 관대한 조건으로 변경
        valid_mask = (saturation > 60) & (value > 60)  # 선명한 색만
        
        if not np.any(valid_mask):
            # 그래도 안되면 전체 영역 사용
            valid_hues = hue.flatten()
            self.logger.debug(f"색상감지: 전체영역 사용 (각도={target_angle:.1f}°)")
        else:
            valid_hues = hue[valid_mask]
            self.logger.debug(f"색상감지: 유효픽셀={len(valid_hues)} (각도={target_angle:.1f}°)")
        
        if len(valid_hues) == 0:
            return self._get_stable_color(target_angle, 'UNKNOWN')
        
        avg_hue = np.mean(valid_hues)
        
        # 🎯 즉시 색상 판정 (안정화 전)
        if (avg_hue <= 9) or (avg_hue >= 175):         # 좁게!
            instant_result = 'RED'
            self.logger.debug(f"✓ 빨강 감지: Hue={avg_hue:.1f}")
        elif (60 <= avg_hue <= 80):
            instant_result = 'GREEN'
            self.logger.debug(f"✓ 초록 감지: Hue={avg_hue:.1f}")
        elif (100 <= avg_hue <= 120):
            instant_result = 'GREEN'  # 파랑도 초록으로
            self.logger.debug(f"✓ 파랑(청록) 감지: Hue={avg_hue:.1f}")
        else:
            instant_result = 'UNKNOWN'                          # 애매하면 UNKNOWN!
            self.logger.debug(f"? 애매한 색: Hue={avg_hue:.1f}")
        
        # 🕐 안정화된 색상 반환 (1.5초간 같은 색상 범위 유지 필요)
        stable_result = self._get_stable_color(target_angle, instant_result, avg_hue)
        
        self.logger.debug(f"색상결과: 즉시={instant_result} 안정화={stable_result} (Hue={avg_hue:.1f})")
        return stable_result
    
    def _get_stable_color(self, angle, current_color, current_hue):
        """1.5초간 같은 색상 범위 유지하면 안정화된 색상 반환"""
        current_time = time.time()
        
        # 각도 키 (반올림으로 그룹화)
        angle_key = round(angle / 5) * 5  # 5도 단위로 그룹화
        
        # 히스토리 초기화
        if angle_key not in self.color_history:
            self.color_history[angle_key] = {
                'colors': deque(maxlen=50),
                'timestamps': deque(maxlen=50),
                'hues': deque(maxlen=50)
            }
        
        history = self.color_history[angle_key]
        
        # 현재 데이터 추가
        history['colors'].append(current_color)
        history['timestamps'].append(current_time)
        history['hues'].append(current_hue)
        
        # 🧹 오래된 데이터 제거 (1.5초 넘은 것들)
        while (history['timestamps'] and 
               current_time - history['timestamps'][0] > self.stability_duration):
            history['colors'].popleft()
            history['timestamps'].popleft()
            history['hues'].popleft()
        
        # 📊 안정성 분석
        if len(history['colors']) < self.min_samples:
            # 샘플 부족 - UNKNOWN 반환
            return 'UNKNOWN'
        
        # 🎨 색상 범위별 연속성 체크
        def is_same_color_range(color1, color2, hue1, hue2):
            """두 색상이 같은 범위에 있는지 확인"""
            if color1 == color2 and color1 != 'UNKNOWN':
                return True
            # 같은 색상 범위 내에서 Hue 차이가 10 이하면 같은 색상으로 간주
            if color1 == color2 == 'RED':
                return abs(hue1 - hue2) < 10 or abs(hue1 - hue2 + 180) < 10 or abs(hue1 - hue2 - 180) < 10
            elif color1 == color2 == 'GREEN':
                return abs(hue1 - hue2) < 15
            return False
        
        # 연속된 같은 색상 범위 카운트
        consecutive_count = 1
        target_color = history['colors'][-1]
        target_hue = history['hues'][-1]
        
        if target_color == 'UNKNOWN':
            return 'UNKNOWN'
        
        # 뒤에서부터 연속으로 같은 색상 범위인지 확인
        for i in range(len(history['colors']) - 2, -1, -1):
            if is_same_color_range(target_color, history['colors'][i], target_hue, history['hues'][i]):
                consecutive_count += 1
            else:
                break
        
        # 🎯 안정성 기준: 전체 샘플의 80% 이상이 연속으로 같은 색상 범위
        stability_ratio = consecutive_count / len(history['colors'])
        
        if stability_ratio >= 0.8 and consecutive_count >= self.min_samples:
            self.logger.debug(f"🎯 안정화된 색상: {target_color} (연속={consecutive_count}/{len(history['colors'])}, {stability_ratio:.1%})")
            return target_color
        else:
            self.logger.debug(f"⏳ 색상 불안정: {target_color} (연속={consecutive_count}/{len(history['colors'])}, {stability_ratio:.1%})")
            return 'UNKNOWN'


class SimpleLogger:
    """간단한 로거 클래스"""
    def __init__(self):
        self.enable_debug = True
        
    def info(self, msg):
        if self.enable_debug:
            print(f"[INFO] {datetime.now().strftime('%H:%M:%S')} - {msg}")
    
    def warning(self, msg):
        print(f"[WARN] {datetime.now().strftime('%H:%M:%S')} - {msg}")
    
    def error(self, msg):
        print(f"[ERROR] {datetime.now().strftime('%H:%M:%S')} - {msg}")
    
    def debug(self, msg):
        if self.enable_debug:
            print(f"[DEBUG] {datetime.now().strftime('%H:%M:%S')} - {msg}")


class MockLidarData:
    """테스트용 가상 LiDAR 데이터"""
    def __init__(self):
        self.angle_min = -np.pi/4
        self.angle_max = np.pi/4
        self.angle_increment = (self.angle_max - self.angle_min) / 360
        self.ranges = self.generate_test_data()
    
    def generate_test_data(self):
        ranges = np.full(360, 10.0)
        
        # 왼쪽 콘 (-20도, 3m)
        left_center = int((-20 - np.degrees(self.angle_min)) / np.degrees(self.angle_increment))
        for i in range(-5, 6):
            if 0 <= left_center + i < len(ranges):
                ranges[left_center + i] = 3.0 + abs(i) * 0.1
        
        # 오른쪽 콘 (+25도, 3.5m)
        right_center = int((25 - np.degrees(self.angle_min)) / np.degrees(self.angle_increment))
        for i in range(-4, 5):
            if 0 <= right_center + i < len(ranges):
                ranges[right_center + i] = 3.5 + abs(i) * 0.15
        
        return ranges


class GateNavigator:
    """🧠 LiDAR 꼬깔 감지 + 색상 이분법 통합 항법 + 기억 시스템"""
    def __init__(self, logger):
        self.logger = logger
        
        self.cone_detector = ConeDetector(logger.info)
        self.color_classifier = ColorRegionClassifier(logger)
        
        # 카메라 초기화
        self.cap = self.find_camera()
        self.camera_available = (self.cap is not None and self.cap.isOpened())
        
        if self.camera_available:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.logger.info("카메라 활성화")
        else:
            self.logger.warning("카메라 없음 - LiDAR 단독 모드")
        
        # 게이트 상태
        self.detected_gates = []
        self.target_gate = None
        self.color_rule = None
        
        # 플래그
        self.left_cone_flag = False
        self.right_cone_flag = False
        
        # 🧠 기억 시스템 (핵심!)
        self.last_seen_cones = {'RED': None, 'GREEN': None}
        self.memory_timeout = 5.0  # 5초
        
        # 탐색 상태 머신
        self.search_state = 'IDLE'  # 'IDLE', 'SEARCHING', 'MEMORY_NAV', 'TARGET_ACQUIRED'
        
        # 시각화
        if self.camera_available:
            cv2.namedWindow('Gate Detection Debug')
        
        self.logger.info("🧠 게이트 네비게이터 초기화 완료 (기억 시스템 활성)")
    
    def show_live_camera(self):
        """카메라가 있으면 실시간 화면을 항상 띄움 + 색상 감지 표시"""
        if not self.camera_available:
            return
        ret, frame = self.cap.read()
        if ret and frame is not None:
            h, w = frame.shape[:2]
            live_frame = frame.copy()
            
            # 중앙 십자선
            cv2.line(live_frame, (w//2-20, h//2), (w//2+20, h//2), (0, 255, 0), 2)
            cv2.line(live_frame, (w//2, h//2-20), (w//2, h//2+20), (0, 255, 0), 2)
            
            # 색상 감지 영역 표시 (좌측, 중앙, 우측)
            test_angles = [-30, 0, 30]  # 테스트할 각도들
            for i, angle in enumerate(test_angles):
                # 각도를 픽셀 위치로 변환
                normalized = (angle + 43.5) / 87
                x_pixel = int(normalized * w)
                x_pixel = np.clip(x_pixel, 0, w - 1)
                
                # ROI 영역
                x_start = max(0, x_pixel - 25)
                x_end = min(w, x_pixel + 25)
                y_start = h // 4
                y_end = 3 * h // 4
                
                roi = frame[y_start:y_end, x_start:x_end]
                
                if roi.size > 0:
                    # 색상 분류
                    color_result = self.color_classifier.classify_region_at_angle(frame, angle)
                    
                    # 안정성 정보 가져오기 (연속성 기반)
                    angle_key = round(angle / 5) * 5
                    stability_info = ""
                    if angle_key in self.color_classifier.color_history:
                        history = self.color_classifier.color_history[angle_key]
                        if len(history['colors']) > 0:
                            # 연속성 계산
                            consecutive_count = 1
                            target_color = history['colors'][-1]
                            
                            if target_color != 'UNKNOWN' and len(history['hues']) > 0:
                                target_hue = history['hues'][-1]
                                
                                for i in range(len(history['colors']) - 2, -1, -1):
                                    current_color = history['colors'][i]
                                    current_hue = history['hues'][i]
                                    
                                    # 같은 색상 범위인지 확인
                                    is_same = False
                                    if current_color == target_color and current_color != 'UNKNOWN':
                                        if current_color == 'RED':
                                            is_same = abs(target_hue - current_hue) < 10 or abs(target_hue - current_hue + 180) < 10 or abs(target_hue - current_hue - 180) < 10
                                        elif current_color == 'GREEN':
                                            is_same = abs(target_hue - current_hue) < 15
                                    
                                    if is_same:
                                        consecutive_count += 1
                                    else:
                                        break
                                
                                stability_ratio = consecutive_count / len(history['colors'])
                                stability_info = f" ({consecutive_count}연속/{stability_ratio:.0%})"
                    
                    # ROI 박스 그리기 (안정화된 색상은 굵게, 불안정하면 점선 효과)
                    box_color = (0, 0, 255) if color_result == 'RED' else (0, 255, 0) if color_result == 'GREEN' else (128, 128, 128)
                    thickness = 3 if color_result in ['RED', 'GREEN'] else 1
                    cv2.rectangle(live_frame, (x_start, y_start), (x_end, y_end), box_color, thickness)
                    
                    # 색상 결과 텍스트 + 안정성
                    result_text = f"{color_result}{stability_info}"
                    cv2.putText(live_frame, result_text, (x_start, y_start-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
                    
                    # HSV 평균값도 표시
                    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    mean_hsv = np.mean(hsv_roi, axis=(0,1))
                    hsv_text = f"H:{mean_hsv[0]:.0f}"
                    cv2.putText(live_frame, hsv_text, (x_start, y_end+15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1)
            
            # 정보 텍스트
            cv2.putText(live_frame, "Live Camera + Color Detection", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            current_time_str = datetime.now().strftime('%H:%M:%S')
            cv2.putText(live_frame, current_time_str, (10, h-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(live_frame, f"Hue Boundary: {self.color_classifier.hue_boundary}", (10, h-40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            cv2.imshow('Live Camera Feed', live_frame)
            cv2.waitKey(1)
    
    def find_camera(self):
        for index in range(10):
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and len(frame.shape) == 3 and frame.shape[2] == 3:
                    self.logger.info(f"✅ RGB 카메라 발견: video{index}")
                    return cap
                cap.release()
        return None
    
    def update(self, lidar_data):
        """메인 업데이트 (LiDAR + 카메라 융합 + 기억)"""
        # 1. LiDAR로 꼬깔 감지
        cones = self.cone_detector.detect_cones(
            lidar_data.ranges,
            lidar_data.angle_min,
            lidar_data.angle_increment
        )
        
        # 2. 카메라 프레임 획득 (우선 시도)
        frame = None
        if self.camera_available:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                # 성공적으로 프레임 획득
                pass
            else:
                self.logger.warning("카메라 프레임 읽기 실패")
                frame = None
        
        # 3. 각 꼬깔에 색상 레이블 부여
        for cone in cones:
            if frame is not None:
                color_region = self.color_classifier.classify_region_at_angle(
                    frame, cone['angle']
                )
                cone['color'] = color_region
            else:
                # 테스트용 가상 색상
                cone['color'] = 'RED' if cone['angle'] < 0 else 'GREEN'
        
        # 🧠 3-1. 발견한 꼬깔 정보 기억하기
        current_time = time.time()
        for cone in cones:
            if cone['color'] in ['RED', 'GREEN']:
                self.last_seen_cones[cone['color']] = {
                    'angle': cone['angle'],
                    'distance': cone['distance'],
                    'x': cone['x'],
                    'y': cone['y'],
                    'z': cone['z'],
                    'timestamp': current_time
                }
                self.logger.debug(f"🧠 기억: {cone['color']} 각도={cone['angle']:.1f}° 거리={cone['distance']:.1f}m")
        
        # 4. 좌/우 플래그 업데이트 (기억 포함)
        self._update_cone_flags(cones)
        
        # 5. 유효한 게이트 찾기
        self.detected_gates = self._find_valid_gates(cones)
        
        # 6. 첫 게이트로 색 규칙 학습
        if self.detected_gates and self.color_rule is None:
            self._learn_color_rule(self.detected_gates[0])
        
        # 7. 가장 가까운 게이트 선택 or 기억 기반 가상 게이트
        if self.detected_gates:
            self.target_gate = min(self.detected_gates, key=lambda g: g['distance'])
            self.search_state = 'TARGET_ACQUIRED'
        else:
            # 게이트를 못 찾았지만, 기억이 있으면 기억 기반 항법
            if self._has_valid_memory():
                self.search_state = 'MEMORY_NAV'
                self.target_gate = self._create_virtual_gate_from_memory()
            else:
                self.search_state = 'SEARCHING'
                self.target_gate = None
        
        # 8. 디버그 시각화 (카메라 있을 때만)
        if self.camera_available:
            if frame is not None:
                self._show_debug_image(frame, cones)
            else:
                # 더미 프레임으로라도 표시
                dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(dummy_frame, "Camera Feed Lost", (200, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Live Camera Feed', dummy_frame)
    
    def _has_valid_memory(self):
        """유효한 기억이 있는지 확인"""
        current_time = time.time()
        
        red_valid = (self.last_seen_cones['RED'] is not None and 
                     (current_time - self.last_seen_cones['RED']['timestamp']) < self.memory_timeout)
        
        green_valid = (self.last_seen_cones['GREEN'] is not None and 
                       (current_time - self.last_seen_cones['GREEN']['timestamp']) < self.memory_timeout)
        
        return red_valid and green_valid
    
    def _create_virtual_gate_from_memory(self):
        """기억된 꼬깔 위치로 가상 게이트 생성"""
        red_mem = self.last_seen_cones['RED']
        green_mem = self.last_seen_cones['GREEN']
        
        if not red_mem or not green_mem:
            return None
        
        virtual_gate = {
            'left': {
                'angle': red_mem['angle'] if red_mem['angle'] < green_mem['angle'] else green_mem['angle'],
                'distance': red_mem['distance'] if red_mem['angle'] < green_mem['angle'] else green_mem['distance'],
                'x': red_mem['x'] if red_mem['angle'] < green_mem['angle'] else green_mem['x'],
                'y': red_mem['y'] if red_mem['angle'] < green_mem['angle'] else green_mem['y'],
                'z': red_mem['z'] if red_mem['angle'] < green_mem['angle'] else green_mem['z'],
                'color': 'RED' if red_mem['angle'] < green_mem['angle'] else 'GREEN',
                'is_memory': True
            },
            'right': {
                'angle': green_mem['angle'] if red_mem['angle'] < green_mem['angle'] else red_mem['angle'],
                'distance': green_mem['distance'] if red_mem['angle'] < green_mem['angle'] else red_mem['distance'],
                'x': green_mem['x'] if red_mem['angle'] < green_mem['angle'] else red_mem['x'],
                'y': green_mem['y'] if red_mem['angle'] < green_mem['angle'] else red_mem['y'],
                'z': green_mem['z'] if red_mem['angle'] < green_mem['angle'] else red_mem['z'],
                'color': 'GREEN' if red_mem['angle'] < green_mem['angle'] else 'RED',
                'is_memory': True
            },
            'mid_angle': (red_mem['angle'] + green_mem['angle']) / 2,
            'distance': (red_mem['distance'] + green_mem['distance']) / 2,
            'is_virtual': True
        }
        
        self.logger.info(f"🧠 기억 기반 가상 게이트 생성: 각도={virtual_gate['mid_angle']:.1f}° 거리={virtual_gate['distance']:.1f}m")
        
        return virtual_gate
    
    def _update_cone_flags(self, cones):
        """좌/우 꼬깔 플래그 업데이트 (기억 포함)"""
        left_cones = [c for c in cones if c['angle'] < -5]
        right_cones = [c for c in cones if c['angle'] > 5]
        
        if left_cones:
            if self.color_rule:
                left_match = any(c['color'] == self.color_rule['left'] for c in left_cones)
                self.left_cone_flag = left_match
            else:
                self.left_cone_flag = True
        else:
            # 좌측에 현재 감지 안 됨 - 기억 확인
            if self.color_rule and self._has_left_memory():
                self.left_cone_flag = True
            else:
                self.left_cone_flag = False
        
        if right_cones:
            if self.color_rule:
                right_match = any(c['color'] == self.color_rule['right'] for c in right_cones)
                self.right_cone_flag = right_match
            else:
                self.right_cone_flag = True
        else:
            # 우측에 현재 감지 안 됨 - 기억 확인
            if self.color_rule and self._has_right_memory():
                self.right_cone_flag = True
            else:
                self.right_cone_flag = False
    
    def _has_left_memory(self):
        current_time = time.time()
        left_color = self.color_rule['left'] if self.color_rule else None
        
        if left_color and self.last_seen_cones[left_color]:
            age = current_time - self.last_seen_cones[left_color]['timestamp']
            return age < self.memory_timeout
        return False
    
    def _has_right_memory(self):
        current_time = time.time()
        right_color = self.color_rule['right'] if self.color_rule else None
        
        if right_color and self.last_seen_cones[right_color]:
            age = current_time - self.last_seen_cones[right_color]['timestamp']
            return age < self.memory_timeout
        return False
    
    def _find_valid_gates(self, cones):
        red_cones = [c for c in cones if c['color'] == 'RED']
        green_cones = [c for c in cones if c['color'] == 'GREEN']
        
        if not red_cones or not green_cones:
            return []
        
        gates = []
        for red in red_cones:
            for green in green_cones:
                angle_diff = abs(red['angle'] - green['angle'])
                
                if 15 < angle_diff < 60:
                    left_cone = red if red['angle'] < green['angle'] else green
                    right_cone = green if red['angle'] < green['angle'] else red
                    
                    mid_angle = (red['angle'] + green['angle']) / 2
                    mid_distance = (red['distance'] + green['distance']) / 2
                    
                    gates.append({
                        'left': left_cone,
                        'right': right_cone,
                        'mid_angle': mid_angle,
                        'distance': mid_distance
                    })
        
        return gates
    
    def _learn_color_rule(self, first_gate):
        self.color_rule = {
            'left': first_gate['left']['color'],
            'right': first_gate['right']['color']
        }
        self.logger.info(f"🎓 색 규칙 학습: 왼쪽={self.color_rule['left']}, 오른쪽={self.color_rule['right']}")
    
    def _show_debug_image(self, frame, cones):
        """디버그 이미지 표시 (기억된 꼬깔 포함)"""
        debug_frame = frame.copy()
        h, w = debug_frame.shape[:2]
        
        cv2.line(debug_frame, (w//2, 0), (w//2, h), (128, 128, 128), 2)
        
        # 감지된 꼬깔 표시
        for cone in cones:
            angle = cone['angle']
            x = int((angle + 43.5) / 87 * w)
            
            color_map = {'RED': (0, 0, 255), 'GREEN': (0, 255, 0), 'UNKNOWN': (128, 128, 128)}
            color = color_map.get(cone['color'], (255, 255, 255))
            
            cv2.circle(debug_frame, (x, h//2), 15, color, -1)
            cv2.putText(debug_frame, f"{cone['distance']:.1f}m", 
                       (x-20, h//2-20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # 🧠 기억된 꼬깔 표시 (반투명)
        current_time = time.time()
        for color, memory in self.last_seen_cones.items():
            if memory and (current_time - memory['timestamp']) < self.memory_timeout:
                # 현재 감지되지 않은 것만 표시
                if not any(c['color'] == color for c in cones):
                    angle = memory['angle']
                    x = int((angle + 43.5) / 87 * w)
                    
                    mem_color = (128, 128, 255) if color == 'RED' else (128, 255, 128)
                    cv2.circle(debug_frame, (x, h//2), 12, mem_color, 2)  # 테두리만
                    cv2.putText(debug_frame, "MEM", (x-15, h//2+25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, mem_color, 1)
        
        # 타겟 게이트 표시
        if self.target_gate:
            left_x = int((self.target_gate['left']['angle'] + 43.5) / 87 * w)
            right_x = int((self.target_gate['right']['angle'] + 43.5) / 87 * w)
            mid_x = (left_x + right_x) // 2
            
            gate_color = (128, 128, 0) if self.target_gate.get('is_virtual') else (255, 255, 0)
            cv2.line(debug_frame, (left_x, h//2), (right_x, h//2), gate_color, 3)
            cv2.circle(debug_frame, (mid_x, h//2), 20, (255, 0, 255), -1)
            
            # 가상 게이트 표시
            if self.target_gate.get('is_virtual'):
                cv2.putText(debug_frame, "VIRTUAL", (mid_x-30, h//2-30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 0), 2)
        
        # 상태 정보
        flag_text = f"L:{self.left_cone_flag} R:{self.right_cone_flag} State:{self.search_state}"
        cv2.putText(debug_frame, flag_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        if self.color_rule:
            rule_text = f"Rule: L={self.color_rule['left']} R={self.color_rule['right']}"
            cv2.putText(debug_frame, rule_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 기억 정보
        memory_status = "MEM:OK" if self._has_valid_memory() else "MEM:NONE"
        cv2.putText(debug_frame, memory_status, (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 128, 0), 2)
        
        cv2.imshow('Gate Detection Debug', debug_frame)
        cv2.waitKey(1)
    
    def get_navigation_command(self):
        """항법 명령 반환 (기억 기반 항법 포함)"""
        # 상태별 처리
        if self.search_state == 'TARGET_ACQUIRED':
            if not self.target_gate:
                return None
            
            if not (self.left_cone_flag and self.right_cone_flag):
                return 'SEARCH_L'
            
            mid_angle = self.target_gate['mid_angle']
            
            if mid_angle < -8:
                return 'L'
            elif mid_angle > 8:
                return 'R'
            else:
                return 'F'
        
        elif self.search_state == 'MEMORY_NAV':
            if not self.target_gate:
                return 'SEARCH_L'
            
            mid_angle = self.target_gate['mid_angle']
            
            self.logger.info(f"🧠 기억 항법: 목표각도={mid_angle:.1f}°")
            
            if mid_angle < -8:
                return 'L'
            elif mid_angle > 8:
                return 'R'
            else:
                return 'F'
        
        elif self.search_state == 'SEARCHING':
            return 'SEARCH_L'
        
        else:
            return None
    
    def get_status(self):
        return {
            'left_flag': self.left_cone_flag,
            'right_flag': self.right_cone_flag,
            'gates_detected': len(self.detected_gates),
            'target_distance': self.target_gate['distance'] if self.target_gate else None,
            'target_angle': self.target_gate['mid_angle'] if self.target_gate else None,
            'search_state': self.search_state,
            'has_memory': self._has_valid_memory(),
            'is_virtual_gate': self.target_gate.get('is_virtual', False) if self.target_gate else False
        }
    
    def cleanup(self):
        if self.camera_available and self.cap:
            self.cap.release()
        cv2.destroyAllWindows()


class LocalBoatController:
    def __init__(self):
        self.logger = SimpleLogger()
        
        self.emergency_stop_time = None
        self.is_in_emergency = False
        self.left_speed = 0
        self.right_speed = 0
        self.speed_step = 10
        self.arduino = None
        self.arduino_connected = False

        self.control_mode = 0
        self.emergency_stop = False

        self.danger_threshold = 0.7
        self.safe_threshold = 1.2
        self.emergency_threshold = 0.15
        
        self.auto_command = 'F'
        self.previous_auto_command = 'F'
        
        self.gate_nav = GateNavigator(self.logger)
        self.mock_lidar = MockLidarData()
        
        try:
            self.settings = termios.tcgetattr(sys.stdin)
        except Exception as e:
            self.logger.error(f"터미널 설정 실패: {e}")
            self.settings = None

        self.connect_arduino()
        
        self.running = True
        self.auto_thread = threading.Thread(target=self.auto_control_loop, daemon=True)
        self.auto_thread.start()

        self.print_instructions()

    def connect_arduino(self):
        possible_ports = ['/dev/ttyACM0', '/dev/ttyACM1', '/dev/ttyUSB0', '/dev/ttyUSB1']
        for port in possible_ports:
            try:
                self.arduino = serial.Serial(port, 115200, timeout=0.1)
                time.sleep(2)
                self.arduino_connected = True
                self.logger.info(f"아두이노 연결: {port}")
                break
            except:
                continue

        if not self.arduino_connected:
            self.logger.error("아두이노 연결 실패 - 시뮬레이션 모드")

    def print_instructions(self):
        status = "연결완료" if self.arduino_connected else "시뮬레이션"
        camera = "활성" if self.gate_nav.camera_available else "비활성"
        mode_names = ["수동", "LiDAR시뮬", "게이트(기억시스템)"]
        
        print(f"""
{status} - 로컬 보트 컨트롤러 🧠 기억 시스템
========================================
현재: {mode_names[self.control_mode]} | 카메라: {camera}

모드: 1(수동) 2(LiDAR시뮬) 3(게이트🧠) x(긁급정지)
수동: w/s(전후) a/d(좌우) space(정지)

🧠 기억 시스템 특징:
  - 한쪽만 보여도 기억으로 항법
  - 5초간 기억 유지
  - 회전하며 탐색 후 중앙 통과

속도: L{self.left_speed} R{self.right_speed}
========================================
        """)

    def get_key(self):
        if not self.settings:
            return ''
        
        try:
            tty.setraw(sys.stdin.fileno())
            rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
            if rlist:
                key = sys.stdin.read(1)
                if key == '\x1b':
                    rlist2, _, _ = select.select([sys.stdin], [], [], 0.1)
                    if rlist2:
                        sys.stdin.read(2)
                    key = 'ESC'
            else:
                key = ''
        except:
            key = ''
        finally:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
            except:
                pass
        return key

    def clamp_speed(self, speed):
        return max(-255, min(255, speed))

    def send_motor_command(self):
        if self.emergency_stop:
            self.left_speed = 0
            self.right_speed = 0

        if not self.arduino_connected:
            print(f"\r모터: L{self.left_speed:4d} R{self.right_speed:4d}   ", end='', flush=True)
            return

        try:
            self.arduino.flushInput()
            self.arduino.flushOutput()
            command = f"L{self.left_speed},R{self.right_speed}\n"
            self.arduino.write(command.encode('utf-8'))
            time.sleep(0.05)
        except Exception as e:
            self.logger.error(f"통신 에러: {e}")

    def auto_control_loop(self):
        """자동 제어 루프 (별도 스레드)"""
        while self.running:
            if self.control_mode == 0:
                time.sleep(0.1)
                continue
            
            command = None
            
            if self.control_mode == 1:
                # LiDAR 시뮬레이션
                command = self.simulate_lidar_avoidance()
            
            elif self.control_mode == 2:
                # 🧠 게이트 항법 (기억 시스템)
                self.gate_nav.update(self.mock_lidar)
                nav_command = self.gate_nav.get_navigation_command()
                status = self.gate_nav.get_status()
                
                # 탐색 명령 처리
                if nav_command == 'SEARCH_L':
                    command = 'SEARCH_L'
                    if command != self.previous_auto_command:
                        self.logger.warning(
                            f"[게이트 탐색] 좌회전 탐색 중 - "
                            f"State:{status['search_state']} "
                            f"Memory:{status['has_memory']}"
                        )
                        self.previous_auto_command = command
                
                elif nav_command in ['F', 'L', 'R']:
                    command = nav_command
                    if command != self.previous_auto_command:
                        gate_type = "🧠기억" if status['is_virtual_gate'] else "👁실시간"
                        self.logger.info(
                            f"[게이트 {gate_type}] {command} - "
                            f"L:{status['left_flag']} R:{status['right_flag']} "
                            f"Gates:{status['gates_detected']} "
                            f"Dist:{status['target_distance']:.1f}m " if status['target_distance'] else "Dist:N/A "
                            f"Angle:{status['target_angle']:.1f}°" if status['target_angle'] else "Angle:N/A"
                        )
                        self.previous_auto_command = command
                
                else:
                    command = 'S'
                    if command != self.previous_auto_command:
                        self.logger.warning("[게이트] 정지")
                        self.previous_auto_command = command
            
            # 모터 제어 (탐색 명령 추가)
            if command:
                speed_map = {
                    'F': (190, -190),
                    'B': (-190, 190),
                    'L': (190, 190),
                    'R': (-190, -190),
                    'SEARCH_L': (80, 80),      # 느린 좌회전 탐색
                    'SEARCH_R': (-80, -80),    # 느린 우회전 탐색
                    'S': (0, 0)
                }
                
                if command in speed_map:
                    self.left_speed, self.right_speed = speed_map[command]
                    self.send_motor_command()
            
            time.sleep(0.1)

    def simulate_lidar_avoidance(self):
        """가상 LiDAR 장애물 회피"""
        ranges = self.mock_lidar.ranges
        
        front_ranges = ranges[170:190]
        left_ranges = ranges[30:120]
        right_ranges = ranges[240:330]
        
        front_min = np.min(front_ranges)
        left_min = np.min(left_ranges)
        right_min = np.min(right_ranges)
        
        if front_min < 0.5:
            return 'S'
        elif front_min < 1.0:
            return 'L' if left_min > right_min else 'R'
        else:
            return 'F'

    def run(self):
        if not self.settings:
            return

        try:
            while True:
                # 실시간 카메라 화면 항상 띄우기
                self.gate_nav.show_live_camera()

                key = self.get_key()

                if key == '1':
                    self.control_mode = 0
                    self.emergency_stop = False
                    self.left_speed = self.right_speed = 0
                    print("\n수동 모드")
                elif key == '2':
                    self.control_mode = 1
                    self.emergency_stop = False
                    print("\nLiDAR 시뮬레이션 모드")
                elif key == '3':
                    self.control_mode = 2
                    self.emergency_stop = False
                    print("\n🧠 게이트 네비게이션 모드 (기억 시스템 활성)")
                elif key == 'x':
                    self.emergency_stop = True
                    self.left_speed = self.right_speed = 0
                    print("\n긴급정지")
                elif key == '\x03':  # Ctrl+C
                    break

                if self.emergency_stop and key != 'x':
                    continue

                if self.control_mode == 0 and not self.emergency_stop:
                    manual_map = {
                        'w': (175, -175), 's': (-175, 175),
                        'a': (175, 175), 'd': (-175, -175),
                        ' ': (0, 0), 'r': (0, 0)
                    }
                    
                    if key in manual_map:
                        self.left_speed, self.right_speed = manual_map[key]
                    elif key in ['q', 'z', 'e', 'c']:
                        delta = self.speed_step if key in ['q', 'e'] else -self.speed_step
                        if key in ['q', 'z']:
                            self.left_speed = self.clamp_speed(self.left_speed + delta)
                        else:
                            self.right_speed = self.clamp_speed(self.right_speed + delta)

                if key and key != '\x03' and self.control_mode == 0:
                    self.send_motor_command()

        except KeyboardInterrupt:
            pass
        finally:
            self.cleanup()

    def cleanup(self):
        try:
            self.running = False
            self.left_speed = self.right_speed = 0
            self.send_motor_command()
            
            if self.settings:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
            if self.arduino_connected and self.arduino:
                self.arduino.close()
            
            self.gate_nav.cleanup()
            self.logger.info("시스템 종료")
        except Exception as e:
            self.logger.error(f"종료 에러: {e}")


def main():
    print("""
╔═══════════════════════════════════════════════════════╗
║   🧠 게이트 네비게이터 with 기억 시스템               ║
║                                                       ║
║   특징:                                               ║
║   - LiDAR 꼬깔 감지                                   ║
║   - 색상 이분법 (HSV Hue 90도 기준)                   ║
║   - 🧠 5초간 위치 기억                                ║
║   - 회전 탐색 → 중앙 통과                             ║
║                                                       ║
║   OpenCV 창에서 실시간 디버그 확인!                   ║
╚═══════════════════════════════════════════════════════╝
    """)
    
    controller = LocalBoatController()

    if not controller.settings:
        print("터미널 설정 실패")
        return

    try:
        controller.run()
    except Exception as e:
        controller.logger.error(f"실행 에러: {e}")
        import traceback
        traceback.print_exc()
    finally:
        controller.cleanup()


if __name__ == '__main__':
    main()