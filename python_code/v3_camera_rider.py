#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image, PointCloud2, PointField
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Path
from std_msgs.msg import Header, ColorRGBA
from cv_bridge import CvBridge
import serial
import sys, termios, tty, select
import time
import numpy as np
import cv2
from collections import deque
import threading
import struct

class ConeDetector:
    """
    LiDAR로 꼬깔(삼각뿔) 형태 감지
    """
    def __init__(self, logger):
        self.logger = logger
        self.min_cone_points = 5
        self.max_cone_width = 0.5
        self.angle_tolerance = 15
        
    def detect_cones(self, ranges, angle_min, angle_increment):
        """LiDAR 스캔에서 꼬깔 형태 객체 감지"""
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
        """거리 기반 클러스터링"""
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
        """클러스터가 꼬깔 형태인지 판단"""
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
        """꼬깔의 중심 각도/거리 계산"""
        angles = np.array([p['angle'] for p in cluster])
        angle_rads = np.array([p['angle_rad'] for p in cluster])
        distances = np.array([p['distance'] for p in cluster])
        
        center_angle = np.mean(angles)
        center_angle_rad = np.mean(angle_rads)
        center_distance = np.min(distances) * 0.6 + np.mean(distances) * 0.4
        
        # 3D 좌표 계산 (극좌표 → 직교좌표)
        x = center_distance * np.sin(center_angle_rad)
        y = center_distance * np.cos(center_angle_rad)
        z = 0.3  # 부표 높이 추정
        
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
        
    def classify_region_at_angle(self, frame, target_angle, camera_fov=87):
        """특정 각도 방향의 색 영역 판단"""
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
            return 'UNKNOWN'
        
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0]
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        
        valid_mask = (saturation > 50) & (value > 50)
        
        if not np.any(valid_mask):
            return 'UNKNOWN'
        
        valid_hues = hue[valid_mask]
        avg_hue = np.mean(valid_hues)
        
        if avg_hue < self.hue_boundary:
            return 'RED'
        else:
            return 'GREEN'


class RVizVisualizer:
    """
    RViz2 3D 시각화
    """
    def __init__(self, node):
        self.node = node
        
        # 퍼블리셔들
        self.marker_pub = node.create_publisher(MarkerArray, '/gate/markers', 10)
        self.path_pub = node.create_publisher(Path, '/gate/planned_path', 10)
        self.cone_cloud_pub = node.create_publisher(PointCloud2, '/gate/cone_cloud', 10)
        
        self.node.get_logger().info("RViz 시각화 퍼블리셔 초기화")
    
    def publish_cones(self, cones):
        """감지된 꼬깔들을 마커로 표시"""
        marker_array = MarkerArray()
        
        for i, cone in enumerate(cones):
            # 원뿔 마커
            marker = Marker()
            marker.header.frame_id = "base_link"
            marker.header.stamp = self.node.get_clock().now().to_msg()
            marker.ns = "cones"
            marker.id = i
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            
            marker.pose.position.x = cone['x']
            marker.pose.position.y = cone['y']
            marker.pose.position.z = cone['z'] / 2
            marker.pose.orientation.w = 1.0
            
            marker.scale.x = cone['width']
            marker.scale.y = cone['width']
            marker.scale.z = cone['z']
            
            # 색상 설정
            if cone.get('color') == 'RED':
                marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.8)
            elif cone.get('color') == 'GREEN':
                marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.8)
            else:
                marker.color = ColorRGBA(r=0.5, g=0.5, b=0.5, a=0.8)
            
            marker.lifetime.sec = 0
            marker.lifetime.nanosec = 200000000  # 0.2초
            
            marker_array.markers.append(marker)
            
            # 텍스트 레이블
            text_marker = Marker()
            text_marker.header = marker.header
            text_marker.ns = "cone_labels"
            text_marker.id = i + 1000
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            
            text_marker.pose.position.x = cone['x']
            text_marker.pose.position.y = cone['y']
            text_marker.pose.position.z = cone['z'] + 0.3
            
            text_marker.text = f"{cone.get('color', 'UNK')}\n{cone['distance']:.1f}m"
            text_marker.scale.z = 0.2
            text_marker.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            text_marker.lifetime = marker.lifetime
            
            marker_array.markers.append(text_marker)
        
        self.marker_pub.publish(marker_array)
    
    def publish_gate(self, gate):
        """타겟 게이트 시각화"""
        marker_array = MarkerArray()
        
        if gate is None:
            self.marker_pub.publish(marker_array)
            return
        
        # 게이트 라인
        line_marker = Marker()
        line_marker.header.frame_id = "base_link"
        line_marker.header.stamp = self.node.get_clock().now().to_msg()
        line_marker.ns = "gate_line"
        line_marker.id = 9000
        line_marker.type = Marker.LINE_STRIP
        line_marker.action = Marker.ADD
        
        p1 = Point()
        p1.x = gate['left']['x']
        p1.y = gate['left']['y']
        p1.z = gate['left']['z']
        
        p2 = Point()
        p2.x = gate['right']['x']
        p2.y = gate['right']['y']
        p2.z = gate['right']['z']
        
        line_marker.points = [p1, p2]
        line_marker.scale.x = 0.05
        line_marker.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)
        line_marker.lifetime.sec = 0
        line_marker.lifetime.nanosec = 200000000
        
        marker_array.markers.append(line_marker)
        
        # 중앙 목표점
        target_marker = Marker()
        target_marker.header = line_marker.header
        target_marker.ns = "target_point"
        target_marker.id = 9001
        target_marker.type = Marker.SPHERE
        target_marker.action = Marker.ADD
        
        mid_x = (gate['left']['x'] + gate['right']['x']) / 2
        mid_y = (gate['left']['y'] + gate['right']['y']) / 2
        mid_z = (gate['left']['z'] + gate['right']['z']) / 2
        
        target_marker.pose.position.x = mid_x
        target_marker.pose.position.y = mid_y
        target_marker.pose.position.z = mid_z
        target_marker.pose.orientation.w = 1.0
        
        target_marker.scale.x = 0.3
        target_marker.scale.y = 0.3
        target_marker.scale.z = 0.3
        target_marker.color = ColorRGBA(r=1.0, g=0.0, b=1.0, a=1.0)
        target_marker.lifetime = line_marker.lifetime
        
        marker_array.markers.append(target_marker)
        
        self.marker_pub.publish(marker_array)
    
    def publish_path(self, current_pos, target_gate):
        """계획된 경로 표시"""
        if target_gate is None:
            return
        
        path = Path()
        path.header.frame_id = "base_link"
        path.header.stamp = self.node.get_clock().now().to_msg()
        
        # 현재 위치
        pose1 = PoseStamped()
        pose1.header = path.header
        pose1.pose.position.x = current_pos[0]
        pose1.pose.position.y = current_pos[1]
        pose1.pose.position.z = 0.0
        pose1.pose.orientation.w = 1.0
        
        # 목표 위치 (게이트 중앙)
        pose2 = PoseStamped()
        pose2.header = path.header
        mid_x = (target_gate['left']['x'] + target_gate['right']['x']) / 2
        mid_y = (target_gate['left']['y'] + target_gate['right']['y']) / 2
        pose2.pose.position.x = mid_x
        pose2.pose.position.y = mid_y
        pose2.pose.position.z = 0.0
        pose2.pose.orientation.w = 1.0
        
        path.poses = [pose1, pose2]
        
        self.path_pub.publish(path)
    
    def publish_cone_pointcloud(self, cones):
        """꼬깔들을 PointCloud2로 표시"""
        if not cones:
            return
        
        points = []
        for cone in cones:
            # RGB 색상 인코딩
            if cone.get('color') == 'RED':
                rgb = struct.unpack('I', struct.pack('BBBB', 255, 0, 0, 255))[0]
            elif cone.get('color') == 'GREEN':
                rgb = struct.unpack('I', struct.pack('BBBB', 0, 255, 0, 255))[0]
            else:
                rgb = struct.unpack('I', struct.pack('BBBB', 128, 128, 128, 255))[0]
            
            points.append([cone['x'], cone['y'], cone['z'], rgb])
        
        # PointCloud2 메시지 생성
        header = Header()
        header.frame_id = "base_link"
        header.stamp = self.node.get_clock().now().to_msg()
        
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
        ]
        
        cloud_data = []
        for point in points:
            cloud_data.extend(struct.pack('fffI', *point))
        
        cloud_msg = PointCloud2()
        cloud_msg.header = header
        cloud_msg.height = 1
        cloud_msg.width = len(points)
        cloud_msg.fields = fields
        cloud_msg.is_bigendian = False
        cloud_msg.point_step = 16
        cloud_msg.row_step = cloud_msg.point_step * len(points)
        cloud_msg.is_dense = True
        cloud_msg.data = bytes(cloud_data)
        
        self.cone_cloud_pub.publish(cloud_msg)


class GateNavigator:
    """LiDAR 꼬깔 감지 + 색상 이분법 통합 항법 + 기억 시스템"""
    def __init__(self, logger, node=None):
        self.logger = logger
        self.node = node
        
        self.cone_detector = ConeDetector(logger)
        self.color_classifier = ColorRegionClassifier(logger)
        
        # RViz 시각화
        if node:
            self.visualizer = RVizVisualizer(node)
        else:
            self.visualizer = None
        
        # 카메라 초기화
        self.cap = self.find_camera()
        self.camera_available = (self.cap is not None and self.cap.isOpened())
        
        if self.camera_available:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.logger.info("카메라 활성화")
        else:
            self.logger.warning("카메라 없음 - LiDAR 단독 모드")
        
        # ROS2 퍼블리셔
        if self.node and self.camera_available:
            self.bridge = CvBridge()
            self.debug_pub = self.node.create_publisher(Image, '/gate/debug', 10)
        
        # 게이트 상태
        self.detected_gates = []
        self.target_gate = None
        self.color_rule = None
        
        # 플래그
        self.left_cone_flag = False
        self.right_cone_flag = False
        
        # 🧠 기억 시스템 (핵심!)
        self.last_seen_cones = {'RED': None, 'GREEN': None}
        self.memory_timeout = 5.0  # 5초 이상 오래된 기억은 무시
        
        # 탐색 상태 머신
        self.search_state = 'IDLE'  # 'IDLE', 'SEARCHING', 'MEMORY_NAV', 'TARGET_ACQUIRED'
        
        # 경로 히스토리
        self.path_history = deque(maxlen=100)
        
        self.logger.info("게이트 네비게이터 초기화 완료 (기억 시스템 활성)")
    
    def find_camera(self):
        """RGB 카메라 찾기"""
        for index in range(10):
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and len(frame.shape) == 3 and frame.shape[2] == 3:
                    self.logger.info(f"✅ RGB 카메라 발견: video{index}")
                    return cap
                cap.release()
        return None
    
    def update(self, lidar_msg):
        """메인 업데이트 (LiDAR + 카메라 융합 + 기억 시스템)"""
        # 1. LiDAR로 꼬깔 감지
        cones = self.cone_detector.detect_cones(
            lidar_msg.ranges,
            lidar_msg.angle_min,
            lidar_msg.angle_increment
        )
        
        # 2. 카메라 프레임 획득
        frame = None
        if self.camera_available:
            ret, frame = self.cap.read()
            if not ret:
                frame = None
        
        # 3. 각 꼬깔에 색상 레이블 부여
        for cone in cones:
            if frame is not None:
                color_region = self.color_classifier.classify_region_at_angle(
                    frame, cone['angle']
                )
                cone['color'] = color_region
            else:
                cone['color'] = 'UNKNOWN'
        
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
        
        # 4. 좌/우 플래그 업데이트
        self._update_cone_flags(cones)
        
        # 5. 유효한 게이트 찾기
        self.detected_gates = self._find_valid_gates(cones)
        
        # 6. 첫 게이트로 색 규칙 학습
        if self.detected_gates and self.color_rule is None:
            self._learn_color_rule(self.detected_gates[0])
        
        # 7. 가장 가까운 게이트 선택
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
        
        # 8. RViz 시각화
        if self.visualizer:
            # 실제 감지된 꼬깔 + 기억된 꼬깔 함께 표시
            all_cones_to_visualize = cones.copy()
            
            # 기억된 꼬깔을 반투명하게 추가
            for color, memory in self.last_seen_cones.items():
                if memory and (current_time - memory['timestamp']) < self.memory_timeout:
                    # 실제로 현재 감지되지 않은 것만 추가
                    if not any(c['color'] == color for c in cones):
                        memory_cone = {
                            'angle': memory['angle'],
                            'distance': memory['distance'],
                            'x': memory['x'],
                            'y': memory['y'],
                            'z': memory['z'],
                            'color': color,
                            'is_memory': True  # 기억된 것 표시
                        }
                        all_cones_to_visualize.append(memory_cone)
            
            self.visualizer.publish_cones(all_cones_to_visualize)
            self.visualizer.publish_gate(self.target_gate)
            self.visualizer.publish_path([0, 0], self.target_gate)
            self.visualizer.publish_cone_pointcloud(all_cones_to_visualize)
        
        # 9. 카메라 디버그 이미지
        if frame is not None and self.node:
            self._publish_debug_image(frame, cones)
    
    def _has_valid_memory(self):
        """유효한 기억이 있는지 확인"""
        current_time = time.time()
        
        red_valid = (self.last_seen_cones['RED'] is not None and 
                     (current_time - self.last_seen_cones['RED']['timestamp']) < self.memory_timeout)
        
        green_valid = (self.last_seen_cones['GREEN'] is not None and 
                       (current_time - self.last_seen_cones['GREEN']['timestamp']) < self.memory_timeout)
        
        return red_valid and green_valid
    
    def _create_virtual_gate_from_memory(self):
        """
        기억된 꼬깔 위치로 가상 게이트 생성
        실제로는 보이지 않지만, 기억을 바탕으로 목표점 계산
        """
        red_mem = self.last_seen_cones['RED']
        green_mem = self.last_seen_cones['GREEN']
        
        if not red_mem or not green_mem:
            return None
        
        # 가상 게이트 생성 (기억 기반)
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
            'is_virtual': True  # 가상 게이트 표시
        }
        
        self.logger.info(f"🧠 기억 기반 가상 게이트 생성: 각도={virtual_gate['mid_angle']:.1f}° 거리={virtual_gate['distance']:.1f}m")
        
        return virtual_gate
        """좌/우 꼬깔 플래그 업데이트"""
        left_cones = [c for c in cones if c['angle'] < -5]
        right_cones = [c for c in cones if c['angle'] > 5]
        
        if left_cones:
            if self.color_rule:
                left_match = any(c['color'] == self.color_rule['left'] for c in left_cones)
                self.left_cone_flag = left_match
            else:
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
            self.right_cone_flag = False
    
    def _find_valid_gates(self, cones):
        """RED-GREEN 쌍으로 유효한 게이트 찾기"""
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
        """첫 게이트로 좌우 색 규칙 학습"""
        self.color_rule = {
            'left': first_gate['left']['color'],
            'right': first_gate['right']['color']
        }
        self.logger.info(f"🎓 색 규칙 학습: 왼쪽={self.color_rule['left']}, 오른쪽={self.color_rule['right']}")
    
    def _publish_debug_image(self, frame, cones):
        """디버그 이미지 퍼블리시"""
        debug_frame = frame.copy()
        h, w = debug_frame.shape[:2]
        
        cv2.line(debug_frame, (w//2, 0), (w//2, h), (128, 128, 128), 2)
        
        for cone in cones:
            angle = cone['angle']
            x = int((angle + 43.5) / 87 * w)
            
            color_map = {'RED': (0, 0, 255), 'GREEN': (0, 255, 0), 'UNKNOWN': (128, 128, 128)}
            color = color_map.get(cone['color'], (255, 255, 255))
            
            cv2.circle(debug_frame, (x, h//2), 15, color, -1)
            cv2.putText(debug_frame, f"{cone['distance']:.1f}m", 
                       (x-20, h//2-20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        if self.target_gate:
            left_x = int((self.target_gate['left']['angle'] + 43.5) / 87 * w)
            right_x = int((self.target_gate['right']['angle'] + 43.5) / 87 * w)
            mid_x = (left_x + right_x) // 2
            
            cv2.line(debug_frame, (left_x, h//2), (right_x, h//2), (255, 255, 0), 3)
            cv2.circle(debug_frame, (mid_x, h//2), 20, (255, 0, 255), -1)
        
        flag_text = f"L:{self.left_cone_flag} R:{self.right_cone_flag}"
        cv2.putText(debug_frame, flag_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        if self.color_rule:
            rule_text = f"Rule: L={self.color_rule['left']} R={self.color_rule['right']}"
            cv2.putText(debug_frame, rule_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        try:
            msg = self.bridge.cv2_to_imgmsg(debug_frame, encoding='bgr8')
            self.debug_pub.publish(msg)
        except Exception as e:
            self.logger.error(f"디버그 이미지 퍼블리시 실패: {e}")
    
    def get_navigation_command(self):
        """
        항법 명령 반환 (기억 기반 항법 포함)
        
        Returns:
            'F'(직진), 'L'(좌회전), 'R'(우회전), 'SEARCH_L'(탐색 좌회전), 'SEARCH_R'(탐색 우회전), None(정지)
        """
        # 상태별 처리
        if self.search_state == 'TARGET_ACQUIRED':
            # 실제 게이트가 보이는 경우
            if not self.target_gate:
                return None
            
            # 양쪽 플래그 확인 (기억 포함)
            if not (self.left_cone_flag and self.right_cone_flag):
                return 'SEARCH_L'  # 한쪽이 안 보이면 탐색
            
            mid_angle = self.target_gate['mid_angle']
            
            if mid_angle < -8:
                return 'L'
            elif mid_angle > 8:
                return 'R'
            else:
                return 'F'
        
        elif self.search_state == 'MEMORY_NAV':
            # 기억 기반 항법
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
            # 탐색 모드 - 제자리 회전
            return 'SEARCH_L'  # 왼쪽으로 천천히 회전하며 탐색
        
        else:
            return None
    
    def get_status(self):
        """현재 상태 정보"""
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


class HybridBoatController(Node):
    def __init__(self):
        super().__init__('hybrid_boat_controller')

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
        
        # 게이트 네비게이터 초기화
        self.gate_nav = GateNavigator(self.get_logger(), node=self)
        
        try:
            self.settings = termios.tcgetattr(sys.stdin)
        except Exception as e:
            self.get_logger().error(f"터미널 설정 실패: {e}")
            self.settings = None

        self.connect_arduino()

        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            10
        )

        self.auto_timer = self.create_timer(0.1, self.auto_control_update)

        self.print_instructions()

    def connect_arduino(self):
        possible_ports = ['/dev/ttyACM0', '/dev/ttyACM1', '/dev/ttyUSB0', '/dev/ttyUSB1']
        for port in possible_ports:
            try:
                self.arduino = serial.Serial(port, 115200, timeout=0.1)
                time.sleep(2)
                self.arduino_connected = True
                self.get_logger().info(f"아두이노 연결: {port}")
                break
            except:
                continue

        if not self.arduino_connected:
            self.get_logger().error("아두이노 연결 실패 - 시뮬레이션 모드")

    def print_instructions(self):
        status = "연결완료" if self.arduino_connected else "시뮬레이션"
        camera = "활성" if self.gate_nav.camera_available else "비활성"
        mode_names = ["수동", "라이다", "게이트"]
        
        print(f"""
{status} - 하이브리드 보트 (RViz 시각화 지원)
========================================
현재: {mode_names[self.control_mode]} | 카메라: {camera}

모드: 1(수동) 2(라이다) 3(게이트) x(긴급정지)
수동: w/s(전후) a/d(좌우) space(정지)

RViz2 토픽:
  - /gate/markers (3D 꼬깔 마커)
  - /gate/planned_path (계획 경로)
  - /gate/cone_cloud (포인트클라우드)
  - /gate/debug (카메라 디버그)

속도: L{self.left_speed} R{self.right_speed}
========================================

🎨 RViz2 설정 방법:
1. rviz2 실행
2. Fixed Frame을 'base_link'로 설정
3. Add 버튼 클릭
4. By topic에서 다음 항목 추가:
   - /gate/markers (MarkerArray)
   - /gate/planned_path (Path)
   - /gate/cone_cloud (PointCloud2)
   - /gate/debug (Image)
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
            return

        try:
            self.arduino.flushInput()
            self.arduino.flushOutput()
            command = f"L{self.left_speed},R{self.right_speed}\n"
            self.arduino.write(command.encode('utf-8'))
            time.sleep(0.05)
        except Exception as e:
            self.get_logger().error(f"통신 에러: {e}")

    def lidar_callback(self, msg):
        """LiDAR 콜백"""
        if self.control_mode == 3:
            # 게이트 모드
            self.gate_nav.update(msg)
        elif self.control_mode == 2:
            # 라이다 단독 모드
            self.enhanced_scan_callback(msg)

    def enhanced_scan_callback(self, msg):
        """기존 라이다 장애물 회피 로직"""
        try:
            ranges = np.array(msg.ranges)
            ranges = np.where(np.isinf(ranges) | np.isnan(ranges) | (ranges <= 0), 10.0, ranges)
        except:
            return
        
        total = len(ranges)
        front = np.min(ranges[0:30])
        left = np.min(ranges[30:120])
        right = np.min(ranges[total-120:total-30])
        
        if front < 0.5:
            self.auto_command = 'S'
        elif front < 1.0:
            self.auto_command = 'L' if left > right else 'R'
        else:
            self.auto_command = 'F'

    def auto_control_update(self):
        if self.control_mode == 0:
            return
        
        command = None
        
        if self.control_mode == 2:
            # 라이다 단독
            command = self.auto_command
        
        elif self.control_mode == 3:
            # 게이트 항법 (기억 시스템 포함)
            nav_command = self.gate_nav.get_navigation_command()
            status = self.gate_nav.get_status()
            
            # 탐색 명령 처리
            if nav_command == 'SEARCH_L':
                command = 'SEARCH_L'
                if command != self.previous_auto_command:
                    self.get_logger().warning(
                        f"[게이트 탐색] 좌회전 탐색 중 - "
                        f"State:{status['search_state']} "
                        f"Memory:{status['has_memory']}"
                    )
                    self.previous_auto_command = command
            
            elif nav_command == 'SEARCH_R':
                command = 'SEARCH_R'
                if command != self.previous_auto_command:
                    self.get_logger().warning(
                        f"[게이트 탐색] 우회전 탐색 중 - "
                        f"State:{status['search_state']}"
                    )
                    self.previous_auto_command = command
            
            elif nav_command in ['F', 'L', 'R']:
                command = nav_command
                if command != self.previous_auto_command:
                    gate_type = "🧠기억" if status['is_virtual_gate'] else "👁실시간"
                    self.get_logger().info(
                        f"[게이트 {gate_type}] {command} - "
                        f"L:{status['left_flag']} R:{status['right_flag']} "
                        f"Gates:{status['gates_detected']} "
                        f"Dist:{status['target_distance']:.1f}m "
                        f"Angle:{status['target_angle']:.1f}°"
                    )
                    self.previous_auto_command = command
            
            else:
                command = 'S'
                if command != self.previous_auto_command:
                    self.get_logger().warning("[게이트] 정지")
                    self.previous_auto_command = command
        
        # 모터 제어 (탐색 명령 추가)
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

    def run(self):
        if not self.settings:
            return

        try:
            while True:
                key = self.get_key()

                if key == '1':
                    self.control_mode = 0
                    self.emergency_stop = False
                    self.left_speed = self.right_speed = 0
                    print("수동 모드")
                elif key == '2':
                    self.control_mode = 2
                    self.emergency_stop = False
                    print("라이다 모드")
                elif key == '3':
                    self.control_mode = 3
                    self.emergency_stop = False
                    print("게이트 네비게이션 모드 (RViz 시각화 활성)")
                elif key == 'x':
                    self.emergency_stop = True
                    self.left_speed = self.right_speed = 0
                    print("긴급정지")
                elif key == '\x03':
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
            self.left_speed = self.right_speed = 0
            self.send_motor_command()
            if self.settings:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
            if self.arduino_connected and self.arduino:
                self.arduino.close()
            self.gate_nav.cleanup()
            self.get_logger().info("시스템 종료")
        except Exception as e:
            self.get_logger().error(f"종료 에러: {e}")


def main(args=None):
    rclpy.init(args=args)
    controller = HybridBoatController()

    if not controller.settings:
        controller.destroy_node()
        rclpy.shutdown()
        return

    ros_thread = threading.Thread(target=rclpy.spin, args=(controller,))
    ros_thread.daemon = True
    ros_thread.start()

    try:
        controller.run()
    except Exception as e:
        controller.get_logger().error(f"실행 에러: {e}")
    finally:
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()