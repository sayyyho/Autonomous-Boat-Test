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
# Realsense rs_launch.py 기준 토픽
IMAGE_TOPIC       = '/camera/camera/color/image_raw'
DEPTH_TOPIC       = '/camera/camera/depth/image_rect_raw'
CAMERA_INFO_TOPIC = '/camera/camera/depth/camera_info'

ODOM_TOPIC   = '/Odometry'
STATE_TOPIC  = '/mission_state'

DOCK_MODEL_PATH = 'docking.pt'       # (blue_circle, green_rectangle, red_triangle)
LANE_MODEL_PATH = 'cone.pt'          # (red_cone, green_cone, ...)

CONF_THRESHOLD = 0.5

# cone.pt 클래스 이름 (로그 보고 필요시 수정)
LANE_RED_NAME   = 'red_cone'
LANE_GREEN_NAME = 'green_cone'

# 도킹 타겟 클래스
DOCK_TARGET_NAME = 'red_triangle'

# depth 없을 때 임시로 사용할 거리 (m)
DEFAULT_DEPTH = 3.0

# 카메라 → 보트(body) 오프셋 (m)
CAM_TO_BASE_X = 0.0
CAM_TO_BASE_Y = 0.0
CAM_TO_BASE_Z = 0.0

# ---- 도킹 평면 법선 기반 지점 설정 ----
DOCK_STANDOFF_M   = 1.0   # 평면에서 떨어질 거리 (법선 방향, 양수면 카메라 쪽으로 1m)
PLANE_PATCH_SIZE  = 21    # depth 패치 한 변 픽셀 수(홀수 권장)
PLANE_Z_OUTLIER_M = 0.30  # 패치 내부 Z(m) 이상치 제거 허용치(중앙값 대비)
MIN_PLANE_POINTS  = 50    # 평면 추정 최소 유효 포인트 수
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

        # 컬러 이미지 해상도 (depth와 다를 수 있어 스케일링용)
        self.color_width = None
        self.color_height = None

        # FPS
        self.last_time = time.time()
        self.frame_count = 0

        # ---- 구독자 ----
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

        # OFF 모드: 카메라 프레임은 받지만 YOLO/표시/토픽 모두 안 함(유지)
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

        # 시각화 (DOCK/LANE에서만)
        annotated = result.plot()
        cv2.imshow("YOLO Perception", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.get_logger().info("사용자 입력('q')으로 종료합니다.")
            rclpy.shutdown()

        # FPS
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

        # ===== 1) 도킹 모드: red_triangle 평면 법선 기반 도킹 지점 =====
        if mode == 'DOCK':
            cls_ids = boxes.cls.cpu().numpy()
            confs   = boxes.conf.cpu().numpy()
            xyxy    = boxes.xyxy.cpu().numpy()

            best_box = None
            best_conf = -1.0

            for i, cid in enumerate(cls_ids):
                cls_name = names[int(cid)]
                conf = float(confs[i])
                x1, y1, x2, y2 = xyxy[i]

                self.get_logger().info(
                    f"[DET] DOCK, cls_id={int(cid)}, name={cls_name}, conf={conf:.2f}"
                )

                if cls_name != DOCK_TARGET_NAME:  # red_triangle
                    continue

                if conf > best_conf:
                    best_conf = conf
                    best_box = (x1, y1, x2, y2)

            if best_box is None:
                self.get_logger().info("[DOCK] red_triangle 감지 안됨")
                return

            x1, y1, x2, y2 = best_box
            u_color = 0.5 * (x1 + x2)
            v_color = 0.5 * (y1 + y2)

            # 1) 중심 컬러 픽셀 → depth 픽셀
            uv_depth = self.color_to_depth_uv(u_color, v_color)
            if uv_depth is None:
                self.get_logger().warn("[DOCK] depth 좌표 변환 실패 → 중심좌표 fallback")
                target_cam = self.center_point_fallback(u_color, v_color)
            else:
                u_d, v_d = uv_depth
                # 2) depth 패치로 평면 법선/중심 추정
                plane = self.estimate_plane_from_patch(u_d, v_d)
                if plane is None:
                    self.get_logger().warn("[DOCK] 평면 추정 실패 → 중심좌표 fallback")
                    target_cam = self.center_point_fallback(u_color, v_color)
                else:
                    Pc, n_cam = plane  # 카메라 좌표계 기준: 중심, 법선(정규화)
                    # 3) 법선 방향을 '카메라를 향하도록' 정방향화 (n·Pc < 0 되도록)
                    if np.dot(n_cam, Pc) > 0.0:
                        n_cam = -n_cam
                    # 4) 법선 방향으로 standoff 이동
                    target_cam = Pc + n_cam * DOCK_STANDOFF_M

            # 카메라 → 보트 → /Odometry 글로벌
            xb, yb, zb = self.camera_to_body(*target_cam)
            xg, yg, zg = self.body_to_odom(xb, yb, zb)

            p_msg = Point()
            p_msg.x = xg
            p_msg.y = yg
            p_msg.z = zg

            self.get_logger().info(
                f"[DOCK] 도킹지점(법선기반): global=({xg:.2f}, {yg:.2f}, {zg:.2f}), conf={best_conf:.2f}"
            )
            self.pub_dock_spot.publish(p_msg)
            return  # 도킹 모드는 여기서 종료

        # ===== 2) LANE 모드: NAV_GATE / BUOY_CIRCLE 재분기 =====
        cls_ids = boxes.cls.cpu().numpy()
        xyxy    = boxes.xyxy.cpu().numpy()

        # --- NAV_GATE: 빨/초 각각 퍼블리시 + 회전 방향 ---
        if self.current_state == 'NAV_GATE':
            red_pts_body = []
            green_pts_body = []

            for i, cid in enumerate(cls_ids):
                cls_name = names[int(cid)]
                x1, y1, x2, y2 = xyxy[i]
                u_color = 0.5 * (x1 + x2)
                v_color = 0.5 * (y1 + y2)

                self.get_logger().info(
                    f"[DET] NAV_GATE, cls_id={int(cid)}, name={cls_name}"
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

            msg_turn = Bool()
            msg_turn.data = bool(self.calc_turn_right(red_pts_body, green_pts_body))
            self.pub_turn_right.publish(msg_turn)
            return

        # --- BUOY_CIRCLE: 한 개만 선택 → /detected_buoy_global + /buoy_turn_right ---
        if self.current_state == 'BUOY_CIRCLE':
            candidates = []  # (cls_name, xb, yb, zb, xg, yg, zg)

            for i, cid in enumerate(cls_ids):
                cls_name = names[int(cid)]
                if cls_name not in (LANE_RED_NAME, LANE_GREEN_NAME):
                    continue

                x1, y1, x2, y2 = xyxy[i]
                u_color = 0.5 * (x1 + x2)
                v_color = 0.5 * (y1 + y2)

                self.get_logger().info(
                    f"[DET] BUOY_CIRCLE 후보, cls_id={int(cid)}, name={cls_name}"
                )

                depth = self.get_depth_for_color_pixel(u_color, v_color)
                if depth is None:
                    depth = DEFAULT_DEPTH

                Xc, Yc, Zc = self.pixel_to_camera(u_color, v_color, depth)
                xb, yb, zb = self.camera_to_body(Xc, Yc, Zc)
                xg, yg, zg = self.body_to_odom(xb, yb, zb)

                candidates.append((cls_name, xb, yb, zb, xg, yg, zg))

            if not candidates:
                self.get_logger().info("[BUOY_CIRCLE] 빨간/초록 콘 후보 없음")
                return

            # 가장 가까운 콘 하나 선택 (body 기준 거리)
            def dist_body(c):
                _, xb, yb, _, _, _, _ = c
                return math.sqrt(xb*xb + yb*yb)

            best = min(candidates, key=dist_body)
            best_name, xb, yb, zb, xg, yg, zg = best

            p_msg = Point()
            p_msg.x = xg
            p_msg.y = yg
            p_msg.z = zg

            self.pub_other_buoy.publish(p_msg)  # 탐색/원그리기용 글로벌 부표 위치

            msg_turn = Bool()
            msg_turn.data = True if best_name == LANE_RED_NAME else False
            self.pub_turn_right.publish(msg_turn)

            self.get_logger().info(
                f"[BUOY_CIRCLE] 선택: {best_name}, "
                f"global=({xg:.2f}, {yg:.2f}, {zg:.2f}), turn_right={msg_turn.data}"
            )
            return

        # 그 외 LANE 상태는 아무 것도 안 함
        return

    # ---------------- depth & 좌표 변환 유틸 ----------------
    def color_to_depth_uv(self, u_color: float, v_color: float):
        """color 픽셀 → depth 픽셀 좌표 스케일링"""
        if self.depth_image is None or self.color_width is None or self.color_height is None:
            return None
        h, w = self.depth_image.shape[:2]
        u_d = int(u_color * w / self.color_width)
        v_d = int(v_color * h / self.color_height)
        if 0 <= u_d < w and 0 <= v_d < h:
            return (u_d, v_d)
        return None

    def get_depth_for_color_pixel(self, u_color: float, v_color: float):
        """color 픽셀에 대응하는 depth 픽셀에서 깊이[m] 리턴"""
        uv = self.color_to_depth_uv(u_color, v_color)
        if uv is None or self.depth_image is None:
            return None
        u_d, v_d = uv
        d_raw = self.depth_image[v_d, u_d]
        if np.issubdtype(type(d_raw), np.integer):
            if d_raw == 0:
                return None
            return float(d_raw) * 0.001
        else:
            d_m = float(d_raw)
            return d_m if d_m > 0.0 else None

    def backproject_depth_pixel(self, u_d: int, v_d: int, depth_m: float):
        """depth 픽셀(u_d, v_d) + 깊이[m] → 카메라(Depth optical) 좌표계 (Xc,Yc,Zc)"""
        Xc = (u_d - self.cx) * depth_m / self.fx
        Yc = (v_d - self.cy) * depth_m / self.fy
        Zc = depth_m
        return np.array([Xc, Yc, Zc], dtype=np.float32)

    def estimate_plane_from_patch(self, u_d: int, v_d: int):
        """
        depth 패치에서 평면 중심(Pc)과 법선(n_cam) 추정 (카메라 좌표계).
        실패 시 None.
        """
        if self.depth_image is None or not self.camera_info_received:
            return None

        h, w = self.depth_image.shape[:2]
        r = PLANE_PATCH_SIZE // 2
        u0 = max(0, u_d - r)
        u1 = min(w - 1, u_d + r)
        v0 = max(0, v_d - r)
        v1 = min(h - 1, v_d + r)

        pts = []
        for vv in range(v0, v1 + 1):
            row = self.depth_image[vv, u0:u1 + 1]
            # 가능한 빠른 필터링
            for uu_offset, d_raw in enumerate(row):
                uu = u0 + uu_offset
                if np.issubdtype(row.dtype, np.integer):
                    if d_raw == 0:
                        continue
                    depth_m = float(d_raw) * 0.001
                else:
                    depth_m = float(d_raw)
                    if depth_m <= 0.0:
                        continue
                pts.append(self.backproject_depth_pixel(uu, vv, depth_m))

        if len(pts) < MIN_PLANE_POINTS:
            return None

        pts = np.stack(pts, axis=0)   # (N,3)

        # Z 이상치 제거 (중앙값 기준)
        z_med = np.median(pts[:, 2])
        mask = np.abs(pts[:, 2] - z_med) < PLANE_Z_OUTLIER_M
        pts = pts[mask]
        if pts.shape[0] < MIN_PLANE_POINTS:
            return None

        # 중심, 공분산, 고유벡터(법선)
        Pc = np.mean(pts, axis=0)  # (3,)
        Q = pts - Pc
        H = (Q.T @ Q) / float(Q.shape[0])
        # H는 대칭행렬 -> eigh
        eigvals, eigvecs = np.linalg.eigh(H)
        n_cam = eigvecs[:, 0]  # 가장 작은 고유값의 고유벡터 = 법선 (3,)

        # 정규화
        n_norm = np.linalg.norm(n_cam)
        if n_norm < 1e-6:
            return None
        n_cam = n_cam / n_norm

        return (Pc, n_cam)

    def center_point_fallback(self, u_color: float, v_color: float):
        """평면 추정 실패 시: 중심 픽셀 좌표로 3D 복원해서 카메라 좌표계 점 반환"""
        d = self.get_depth_for_color_pixel(u_color, v_color)
        if d is None:
            d = DEFAULT_DEPTH
        Xc = (u_color - self.cx) * d / self.fx
        Yc = (v_color - self.cy) * d / self.fy
        Zc = d
        return np.array([Xc, Yc, Zc], dtype=np.float32)

    def pixel_to_camera(self, u_color: float, v_color: float, depth: float):
        """(근사) 컬러 픽셀(u_color, v_color) + depth[m] → 카메라 좌표계 (Xc,Yc,Zc)"""
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
