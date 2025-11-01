#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from std_msgs.msg import String
import pyrealsense2 as rs
import time

# =============== 캡처/표시 설정 ===============
WIDTH, HEIGHT, FPS = 848, 480, 30
# 화면 표시 배율 (이미지만 키워서 보여줌: 성능 안전)
DISPLAY_SCALE = 1.0    # 1.0 = 원본, 1.5~2.0 권장
WINDOW_NAME   = "Docking: Detect 3, Select 1 → /dock/target_xyz"

# HSV (필요시 조정)
HSV_RED_1  = ((0, 120, 70), (10, 255, 255))
HSV_RED_2  = ((170,120,70), (180,255,255))
HSV_GREEN  = ((35, 70, 70), (85, 255, 255))
HSV_BLUE   = ((90, 80, 50), (130,255,255))

MIN_AREA   = 400
KERNEL     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
DEPTH_WIN  = 5
MIN_DEPTH_M, MAX_DEPTH_M = 0.2, 15.0

# 우선순위 (fallback)
PRIORITY_ORDER = ["green_rect", "blue_circle", "red_triangle"]


# =============== 유틸 ===============
def mask_color(hsv, lo, hi):
    mask = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL, iterations=2)
    return mask

def detect_shape(cnt):
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
    v = len(approx)

    if v == 3:
        return "triangle"
    elif v == 4:
        x, y, w, h = cv2.boundingRect(approx)
        ratio = w / float(h) if h > 0 else 0
        return "rectangle" if 0.5 < ratio < 2.0 else "other"
    else:
        area = cv2.contourArea(cnt)
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        circle_area = math.pi * (radius ** 2)
        return "circle" if circle_area > 0 and area / circle_area > 0.70 else "other"

def median_depth_from_window(depth_frame: rs.depth_frame, x: int, y: int, win: int):
    h, w = depth_frame.get_height(), depth_frame.get_width()
    r = max(1, win // 2)
    vals = []
    xs, xe = max(0, x - r), min(w - 1, x + r)
    ys, ye = max(0, y - r), min(h - 1, y + r)
    for yy in range(ys, ye + 1):
        for xx in range(xs, xe + 1):
            d = depth_frame.get_distance(xx, yy)
            if MIN_DEPTH_M <= d <= MAX_DEPTH_M and d > 0.0:
                vals.append(d)
    return float(np.median(vals)) if vals else None


# =============== 노드 ===============
class DockingTargetPublisher(Node):
    def __init__(self):
        super().__init__('docking_target_pub')

        # 파라미터: 원하는 타깃 (green_rect / blue_circle / red_triangle)
        self.declare_parameter('desired_target', 'green_rect')
        self.desired_target = self.get_parameter('desired_target').get_parameter_value().string_value

        # Publisher
        self.pub_target_xyz = self.create_publisher(Point, '/dock/target_xyz', 10)
        self.pub_target_name = self.create_publisher(String, '/dock/target_name', 10)
        self.get_logger().info("✅ /dock/target_xyz, /dock/target_name 퍼블리시 시작")

        # RealSense 초기화
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
        cfg.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
        self.profile = self.pipeline.start(cfg)
        self.align = rs.align(rs.stream.color)
        self.intr = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

        self.get_logger().info(f"🎥 D435i 시작 ( {WIDTH}x{HEIGHT}@{FPS} )")

        # OpenCV 윈도 설정 (크기 조절/전체화면 토글 가능)
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        # 초기 크기 설정
        cv2.resizeWindow(WINDOW_NAME, int(WIDTH * DISPLAY_SCALE), int(HEIGHT * DISPLAY_SCALE))
        self.fullscreen = False

        # FPS 디스플레이용
        self._t_prev = time.time()
        self._fps_smooth = None

    def _update_fps(self):
        t = time.time()
        dt = t - self._t_prev
        self._t_prev = t
        fps = 1.0 / dt if dt > 1e-6 else 0.0
        # 지수평활로 부드럽게
        if self._fps_smooth is None:
            self._fps_smooth = fps
        else:
            self._fps_smooth = 0.9 * self._fps_smooth + 0.1 * fps
        return self._fps_smooth

    def deproject(self, u, v, d):
        x, y, z = rs.rs2_deproject_pixel_to_point(self.intr, [float(u), float(v)], float(d))
        return float(x), float(y), float(z)

    def detect_color_shape(self, hsv, depth, color_img, color_name, hsv_ranges, want_shape, draw_color):
        # 색 마스크
        if color_name == "red":
            mask = mask_color(hsv, *HSV_RED_1) | mask_color(hsv, *HSV_RED_2)
        else:
            (lo, hi) = hsv_ranges
            mask = mask_color(hsv, lo, hi)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []  # (name, (X,Y,Z), (cx,cy), contour)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_AREA:
                continue
            shp = detect_shape(cnt)
            if shp != want_shape:
                continue

            M = cv2.moments(cnt)
            if M["m00"] <= 1e-6:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            d = median_depth_from_window(depth, cx, cy, DEPTH_WIN)
            if d is None:
                continue

            X, Y, Z = self.deproject(cx, cy, d)
            name = f"{color_name}_{want_shape}"
            candidates.append((name, (X, Y, Z), (cx, cy), cnt))

            # 가시화(일반)
            cv2.drawContours(color_img, [cnt], -1, draw_color, 2)
            cv2.circle(color_img, (cx, cy), 6, draw_color, -1)
            cv2.putText(color_img, name, (cx - 40, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, draw_color, 2)

        return candidates

    def spin_once(self):
        frames = self.pipeline.wait_for_frames()
        frames = self.align.process(frames)
        depth = frames.get_depth_frame()
        color = frames.get_color_frame()
        if not depth or not color:
            return

        color_img = np.asanyarray(color.get_data())
        hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)

        # 세 색+도형 모두 후보 수집
        green_rect   = self.detect_color_shape(hsv, depth, color_img, "green", HSV_GREEN, "rectangle", (0, 255, 0))
        blue_circle  = self.detect_color_shape(hsv, depth, color_img, "blue",  HSV_BLUE,  "circle",    (255, 0, 0))
        red_triangle = self.detect_color_shape(hsv, depth, color_img, "red",   (None, None), "triangle", (0, 0, 255))

        # 후보 병합
        all_candidates = green_rect + blue_circle + red_triangle  # (name, (X,Y,Z), (cx,cy), cnt)

        # 1) desired_target 우선
        target_pick = None
        desired = self.desired_target
        for cand in all_candidates:
            if cand[0] == desired:
                target_pick = cand
                break

        # 2) fallback: PRIORITY_ORDER 순서대로
        if target_pick is None:
            for key in PRIORITY_ORDER:
                for cand in all_candidates:
                    if cand[0] == key:
                        target_pick = cand
                        break
                if target_pick:
                    break

        # 3) 그래도 없으면: Z(거리) 가장 가까운 것
        if target_pick is None and all_candidates:
            target_pick = min(all_candidates, key=lambda c: c[1][2])

        # 선택 결과 퍼블리시 & 강조 표시
        if target_pick is not None:
            name, (X, Y, Z), (cx, cy), cnt = target_pick

            # 페이로드 publish
            self.pub_target_xyz.publish(Point(x=X, y=Y, z=Z))
            self.pub_target_name.publish(String(data=name))
            self.get_logger().info(f"🎯 SELECT [{name}]  XYZ=({X:.2f},{Y:.2f},{Z:.2f})")

            # 화면 강조(두꺼운 테두리/큰 점/HUD)
            cv2.drawContours(color_img, [cnt], -1, (0, 255, 255), 3)
            cv2.circle(color_img, (cx, cy), 9, (0, 255, 255), -1)
            hud_main = f"TARGET={name} | XYZ=({X:.2f},{Y:.2f},{Z:.2f})"
            cv2.putText(color_img, hud_main, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 220, 220), 2)
        else:
            cv2.putText(color_img, "No target found", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # FPS 표시
        fps = self._update_fps()
        cv2.putText(color_img, f"{fps:5.1f} FPS", (10, HEIGHT-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,180,180), 2)

        # ===== 화면 표시 (배율 적용) =====
        if DISPLAY_SCALE != 1.0:
            resized = cv2.resize(color_img,
                                 (int(color_img.shape[1]*DISPLAY_SCALE), int(color_img.shape[0]*DISPLAY_SCALE)),
                                 interpolation=cv2.INTER_NEAREST)
            cv2.imshow(WINDOW_NAME, resized)
        else:
            cv2.imshow(WINDOW_NAME, color_img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('f'):
            # 전체화면 토글
            self.fullscreen = not self.fullscreen
            flag = cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, flag)
            if not self.fullscreen:
                cv2.resizeWindow(WINDOW_NAME, int(WIDTH * DISPLAY_SCALE), int(HEIGHT * DISPLAY_SCALE))
        elif key == ord('q'):
            # 안전 종료
            raise KeyboardInterrupt

    def destroy(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = DockingTargetPublisher()
    try:
        while rclpy.ok():
            node.spin_once()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
