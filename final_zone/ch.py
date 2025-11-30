#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import time

import cv2
import numpy as np
from ultralytics import YOLO

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
from std_msgs.msg import String, Bool
from cv_bridge import CvBridge


# ----------------- 설정 -----------------
# Realsense rs_launch.py 기준 토픽 (지금 depth 잘 나오던 세팅 그대로)
IMAGE_TOPIC       = '/camera/camera/color/image_raw'
DEPTH_TOPIC       = '/camera/camera/depth/image_rect_raw'
CAMERA_INFO_TOPIC = '/camera/camera/depth/camera_info'

ODOM_TOPIC   = '/Odometry'
STATE_TOPIC  = '/mission_state'

DOCK_MODEL_PATH = 'docking.pt'       # 도킹용 YOLO (blue_circle, green_rectangle, red_triangle)
LANE_MODEL_PATH = 'cone.pt'          # 항로/부표용 YOLO

CONF_THRESHOLD = 0.5

# 항로 추종용 모델(cone.pt)의 클래스 "이름" 기준
# 실제 names는 로그로 확인해서 필요하면 변경하면 됨.
LANE_RED_NAME   = 'red_cone'    # 예시: id=1 -> 'red_cone'
LANE_GREEN_NAME = 'green_cone'  # 예시: id=0 -> 'green_cone'

# 도킹에서 도킹 스팟으로 쓸 클래스 이름
DOCK_TARGET_NAME = 'red_triangle'   # 로그에서 확인한 그대로

# 도킹 시 마커 평면 법선 방향 오프셋 거리 (m)
DOCK_OFFSET_DISTANCE = 1.5  # 마커에서 1.5m 앞쪽에 목표점 설정

# depth 없을 때 임시로 쓸 기본 거리 (m)
DEFAULT_DEPTH = 3.0
# 카메라 → 보트(body) 오프셋 (필요하면 값 수정)
CAM_TO_BASE_X = 0.0
CAM_TO_BASE_Y = 0.0
CAM_TO_BASE_Z = 0.0
# ---------------------------------------


class YoloPerceptionNode(Node):
    def __init__(self):
        super().__init__('yolo_perception_node')

        # YOLO 모델 로드
        self.get_logger().info("YOLO 모델 로드 중...")
        self.model_dock = YOLO(DOCK_MODEL_PATH)
        self.model_lane = YOLO(LANE_MODEL_PATH)
        self.device = 'cpu'
        self.get_logger().info("모델 로드 완료 ✅")
        self.get_logger().info(f"Docking model classes: {self.model_dock.names}")
        self.get_logger().info(f"Lane model classes   : {self.model_lane.names}")

        self.bridge = CvBridge()

        # 상태 / 오도메트리
        self.current_state = 'NONE'

        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_z = 0.0
        self.odom_yaw = 0.0

        # depth / camera info
        self.depth_image = None       # depth frame (depth 카메라 해상도)
        self.fx = 600.0
        self.fy = 600.0
        self.cx = 320.0
        self.cy = 240.0
        self.camera_info_received = False

        # 컬러 이미지 해상도 (depth와 다를 수 있어서 스케일링용)
        self.color_width = None
        self.color_height = None

        # FPS
        self.last_time = time.time()
        self.frame_count = 0

        # ---- 구독자 (핸들 꼭 변수에 저장) ----
        self.image_sub = self.create_subscription(
            Image, IMAGE_TOPIC, self.image_callback, 10
        )
        self.depth_sub = self.create_subscription(
            Image, DEPTH_TOPIC, self.depth_callback, 10
        )
        self.caminfo_sub = self.create_subscription(
            CameraInfo, CAMERA_INFO_TOPIC, self.camera_info_callback, 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, ODOM_TOPIC, self.odom_callback, 10
        )
        self.state_sub = self.create_subscription(
            String, STATE_TOPIC, self.state_callback, 10
        )

        # ---- 퍼블리셔 ----
        self.pub_red_buoy   = self.create_publisher(Point, '/red_buoy_xyz', 10)
        self.pub_green_buoy = self.create_publisher(Point, '/green_buoy_xyz', 10)
        self.pub_other_buoy = self.create_publisher(Point, '/detected_buoy_global', 10)
        self.pub_turn_right = self.create_publisher(Bool,  '/buoy_turn_right', 10)
        self.pub_dock_spot  = self.create_publisher(Point, '/docking_spot_global', 10)

        self.get_logger().info("YOLO Perception 노드 시작 ✅")
        self.get_logger().info(f"IMAGE_TOPIC       = {IMAGE_TOPIC}")
        self.get_logger().info(f"DEPTH_TOPIC       = {DEPTH_TOPIC}")
        self.get_logger().info(f"CAMERA_INFO_TOPIC = {CAMERA_INFO_TOPIC}")
        self.get_logger().info(f"ODOM_TOPIC        = {ODOM_TOPIC}")
        self.get_logger().info(f"STATE_TOPIC       = {STATE_TOPIC}")
        self.get_logger().info(f"DOCK_OFFSET_DISTANCE = {DOCK_OFFSET_DISTANCE}m")

    # ---------------- 콜백들 ----------------
    def state_callback(self, msg: String):
        self.current_state = msg.data
        self.get_logger().info(f"[STATE] {self.current_state}")

    def odom_callback(self, msg: Odometry):
        self.odom_x = msg.pose.pose.position.x
        self.odom_y = msg.pose.pose.position.y
        self.odom_z = msg.pose.pose.position.z

        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.odom_yaw = math.atan2(siny_cosp, cosy_cosp)

    def depth_callback(self, msg: Image):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding='passthrough'
            )
        except Exception as e:
            self.get_logger().warn(f"Depth convert error: {e}")

    def camera_info_callback(self, msg: CameraInfo):
        # Depth 카메라 K 행렬에서 내부 파라미터 추출
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]
        self.camera_info_received = True

    # ---------------- 이미지 콜백 ----------------
    def image_callback(self, msg: Image):
        mode = self.get_mode_from_state()

        if mode == 'OFF':
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f"CvBridge 변환 오류: {e}")
            return

        # 컬러 해상도 저장
        self.color_height, self.color_width = frame.shape[:2]

        # 상태에 따라 모델 선택
        if mode == 'DOCK':
            model = self.model_dock
        elif mode == 'LANE':
            model = self.model_lane
        else:
            return

        results = model.predict(
            frame,
            device=self.device,
            conf=CONF_THRESHOLD,
            verbose=False
        )
        result = results[0]

        self.process_detections(result, mode)

        # 화면 표시
        annotated = result.plot()
        cv2.imshow("YOLO Perception", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.get_logger().info("사용자 입력('q')으로 종료합니다.")
            rclpy.shutdown()

        # FPS 계산
        self.frame_count += 1
        now = time.time()
        if now - self.last_time >= 1.0:
            fps = self.frame_count / (now - self.last_time)
            self.get_logger().info(f"FPS: {fps:.2f}")
            self.last_time = now
            self.frame_count = 0

    # ---------------- 상태 → 모드 ----------------
    def get_mode_from_state(self) -> str:
        """
        DOCK_APPROACH / DOCK_BACK_OUT : 도킹 모델
        NAV_GATE / BUOY_CIRCLE        : 항로/게이트 모델
        나머지                          : OFF
        """
        s = self.current_state

        if s in ('DOCK_APPROACH', 'DOCK_BACK_OUT'):
            return 'DOCK'
        if s in ('NAV_GATE', 'BUOY_CIRCLE'):
            return 'LANE'
        return 'OFF'

    # ---------------- detection 처리 ----------------
    def process_detections(self, result, mode: str):
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return

        names = result.names  # {cls_id: class_name}

        # ===== 1) 도킹 모드: red_triangle 하나만 선택해서 /docking_spot_global =====
        if mode == 'DOCK':
            best_box = None
            best_conf = -1.0

            cls_ids = result.boxes.cls.cpu().numpy()
            confs   = result.boxes.conf.cpu().numpy()
            xyxy    = result.boxes.xyxy.cpu().numpy()

            for i, cid in enumerate(cls_ids):
                cls_name = names[int(cid)]
                conf = float(confs[i])
                x1, y1, x2, y2 = xyxy[i]

                self.get_logger().info(
                    f"[DET] mode=DOCK, cls_id={int(cid)}, name={cls_name}, conf={conf:.2f}"
                )

                if cls_name != DOCK_TARGET_NAME:
                    continue

                if conf > best_conf:
                    best_conf = conf
                    best_box = (x1, y1, x2, y2)

            if best_box is None:
                self.get_logger().info("[DOCK] red_triangle 감지 안됨")
                return

            # 바운딩박스 중심
            x1, y1, x2, y2 = best_box
            u_color = 0.5 * (x1 + x2)
            v_color = 0.5 * (y1 + y2)

            # Depth 획득
            depth = self.get_depth_for_color_pixel(u_color, v_color)
            if depth is None:
                depth = DEFAULT_DEPTH

            # 좌표 변환: pixel → camera → body
            Xc, Yc, Zc = self.pixel_to_camera(u_color, v_color, depth)
            xb_marker, yb_marker, zb_marker = self.camera_to_body(Xc, Yc, Zc)

            # ===== 법선벡터 방향 오프셋 적용 =====
            # 마커 → 카메라 방향 벡터 (body frame)
            norm = math.sqrt(xb_marker**2 + yb_marker**2 + zb_marker**2)
            
            if norm > 0.1:  # 너무 가까우면 계산 불안정
                # 정규화된 방향벡터
                nx = xb_marker / norm
                ny = yb_marker / norm
                nz = zb_marker / norm
                
                # 마커 평면 법선 방향(카메라 쪽)으로 OFFSET만큼 이동
                # 즉, 마커에서 카메라 방향으로 떨어진 지점
                xb_target = xb_marker - nx * DOCK_OFFSET_DISTANCE
                yb_target = yb_marker - ny * DOCK_OFFSET_DISTANCE
                zb_target = zb_marker - nz * DOCK_OFFSET_DISTANCE
            else:
                # fallback: 그냥 X축(전방) 방향으로 오프셋
                xb_target = xb_marker - DOCK_OFFSET_DISTANCE
                yb_target = yb_marker
                zb_target = zb_marker

            # 글로벌 좌표로 변환
            xg, yg, zg = self.body_to_odom(xb_target, yb_target, zb_target)

            p_msg = Point()
            p_msg.x = xg
            p_msg.y = yg
            p_msg.z = zg

            self.get_logger().info(
                f"[DOCK] Marker(body)=({xb_marker:.2f}, {yb_marker:.2f}, {zb_marker:.2f}), "
                f"Target(body)=({xb_target:.2f}, {yb_target:.2f}, {zb_target:.2f}), "
                f"Global=({xg:.2f}, {yg:.2f}, {zg:.2f}), conf={best_conf:.2f}"
            )
            self.pub_dock_spot.publish(p_msg)
            return  # 도킹 모드는 여기서 끝

        # ===== 2) 항로 모드: 부표 + 회전 방향 =====
        red_pts_body = []
        green_pts_body = []

        cls_ids = result.boxes.cls.cpu().numpy()
        xyxy    = result.boxes.xyxy.cpu().numpy()

        for i, cid in enumerate(cls_ids):
            cls_name = names[int(cid)]
            x1, y1, x2, y2 = xyxy[i]
            u_color = 0.5 * (x1 + x2)
            v_color = 0.5 * (y1 + y2)

            self.get_logger().info(
                f"[DET] mode=LANE, cls_id={int(cid)}, name={cls_name}"
            )

            depth = self.get_depth_for_color_pixel(u_color, v_color)
            if depth is None:
                depth = DEFAULT_DEPTH

            Xc, Yc, Zc = self.pixel_to_camera(u_color, v_color, depth)
            xb, yb, zb = self.camera_to_body(Xc, Yc, Zc)
            xg, yg, zg = self.body_to_odom(xb, yb, zb)

            p_msg = Point()
            p_msg.x = xg
            p_msg.y = yg
            p_msg.z = zg

            if cls_name == LANE_RED_NAME:
                self.pub_red_buoy.publish(p_msg)
                red_pts_body.append((xb, yb, zb))
            elif cls_name == LANE_GREEN_NAME:
                self.pub_green_buoy.publish(p_msg)
                green_pts_body.append((xb, yb, zb))
            else:
                self.pub_other_buoy.publish(p_msg)

        if mode == 'LANE':
            msg_turn = Bool()
            msg_turn.data = bool(self.calc_turn_right(red_pts_body, green_pts_body))
            self.pub_turn_right.publish(msg_turn)

    # ---------------- depth & 좌표 변환 ----------------
    def get_depth_for_color_pixel(self, u_color: float, v_color: float):
        """
        컬러 이미지 픽셀(u_color, v_color)에 대응하는 depth 픽셀에서 깊이[m]을 얻는다.
        color(1280x720)와 depth(848x480) 해상도가 다르므로 비율로 스케일링.
        """
        if self.depth_image is None:
            return None
        if self.color_width is None or self.color_height is None:
            return None

        depth_h, depth_w = self.depth_image.shape[:2]

        # color → depth 좌표 스케일링
        u_depth = int(u_color * depth_w / self.color_width)
        v_depth = int(v_color * depth_h / self.color_height)

        if not (0 <= u_depth < depth_w and 0 <= v_depth < depth_h):
            return None

        d_raw = self.depth_image[v_depth, u_depth]

        # 16UC1(mm) 또는 32FC1(m) 처리
        if np.issubdtype(type(d_raw), np.integer):
            if d_raw == 0:
                return None
            depth_m = float(d_raw) * 0.001
        else:
            depth_m = float(d_raw)
            if depth_m <= 0.0:
                return None

        return depth_m

    def pixel_to_camera(self, u_color: float, v_color: float, depth: float):
        """
        컬러 픽셀 (u_color, v_color) + depth[m] → depth 카메라 optical frame (Xc,Yc,Zc)
        여기서는 depth 카메라 camera_info를 사용하므로,
        color에서 depth로 스케일링된 u_depth/v_depth를 이용한 것과 동일 좌표계라고 가정.
        """
        Xc = (u_color - self.cx) * depth / self.fx
        Yc = (v_color - self.cy) * depth / self.fy
        Zc = depth
        return Xc, Yc, Zc

    def camera_to_body(self, Xc: float, Yc: float, Zc: float):
        """
        depth optical frame → 보트 body frame (X앞, Y좌, Z위)
        예시 변환:
          Xb = Zc
          Yb = -Xc
          Zb = -Yc
        """
        Xb = Zc
        Yb = -Xc
        Zb = -Yc

        Xb += CAM_TO_BASE_X
        Yb += CAM_TO_BASE_Y
        Zb += CAM_TO_BASE_Z

        return Xb, Yb, Zb

    def body_to_odom(self, xb: float, yb: float, zb: float):
        cos_yaw = math.cos(self.odom_yaw)
        sin_yaw = math.sin(self.odom_yaw)

        xg = self.odom_x + cos_yaw * xb - sin_yaw * yb
        yg = self.odom_y + sin_yaw * xb + cos_yaw * yb
        zg = self.odom_z + zb
        return xg, yg, zg

    def calc_turn_right(self, red_pts_body, green_pts_body) -> bool:
        pts = red_pts_body + green_pts_body
        if not pts:
            return False

        ys = [float(p[1]) for p in pts]
        mean_y = sum(ys) / len(ys)
        # numpy.bool_ 방지용으로 bool() 씌움
        return bool(mean_y < 0.0)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = YoloPerceptionNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node:
            node.get_logger().info("KeyboardInterrupt → shutdown")
    finally:
        if node and rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()