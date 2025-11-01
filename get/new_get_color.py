#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point

try:
    import pyrealsense2 as rs
except Exception as e:
    print("[ERR] pyrealsense2 로드 실패. (pip install pyrealsense2)")
    raise

# ================= 기본 설정 =================
WIDTH, HEIGHT, FPS = 848, 480, 15

# HSV 범위 (환경에 맞게 조정 가능)
HSV_RED_1  = ((0,  70, 40), (10, 255, 255))
HSV_RED_2  = ((170,70, 40), (180,255, 255))
HSV_GREEN  = ((35, 70, 40), (85, 255, 255))

MIN_AREA   = 300
KERNEL     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
DEPTH_WIN  = 5
MIN_DEPTH_M, MAX_DEPTH_M = 0.2, 20.0


def mask_color(hsv, lo, hi):
    return cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))


def find_largest_centroid(mask):
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_area = None, 0
    for cnt in contours:
        a = cv2.contourArea(cnt)
        if a < MIN_AREA:
            continue
        if a > best_area:
            M = cv2.moments(cnt)
            if M["m00"] > 1e-6:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                best = (cx, cy, cnt)
                best_area = a
    return best  # (cx, cy, cnt) or None


def median_depth_from_window(depth_frame: rs.depth_frame, x: int, y: int, win: int):
    h, w = depth_frame.get_height(), depth_frame.get_width()
    r = max(1, win // 2)
    vals = []
    xs, xe = max(0, x - r), min(w - 1, x + r)
    ys, ye = max(0, y - r), min(h - 1, y + r)
    for yy in range(ys, ye + 1):
        for xx in range(xs, xe + 1):
            d = depth_frame.get_distance(xx, yy)
            if MIN_DEPTH_M <= d <= MAX_DEPTH_M:
                vals.append(d)
    return float(np.median(vals)) if vals else None


class GateCenterPublisher(Node):
    def __init__(self):
        super().__init__('gate_center_pub')

        # ROS2 publisher (geometry_msgs/Point: float64 x,y,z)
        self.pub_xyz = self.create_publisher(Point, '/gate_center_xyz', 10)
        self.get_logger().info("✅ Publishing /gate_center_xyz  (geometry_msgs/Point)")

        # RealSense init
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
        cfg.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
        self.profile = self.pipeline.start(cfg)
        # depth→color 정렬
        self.align = rs.align(rs.stream.color)
        self.intr = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    def deproject_to_3d(self, u, v, depth_m):
        pt = rs.rs2_deproject_pixel_to_point(self.intr, [float(u), float(v)], float(depth_m))
        return float(pt[0]), float(pt[1]), float(pt[2])  # X,Y,Z (m), camera frame

    def spin_once(self):
        frames = self.pipeline.wait_for_frames()
        frames = self.align.process(frames)
        depth = frames.get_depth_frame()
        color = frames.get_color_frame()
        if not depth or not color:
            return

        color_img = np.asanyarray(color.get_data())
        hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)

        red_mask   = mask_color(hsv, *HSV_RED_1) | mask_color(hsv, *HSV_RED_2)
        green_mask = mask_color(hsv, *HSV_GREEN)

        red   = find_largest_centroid(red_mask)
        green = find_largest_centroid(green_mask)

        vis = color_img.copy()

        if red is not None:
            cv2.circle(vis, (red[0], red[1]), 6, (0, 0, 255), -1)
            cv2.drawContours(vis, [red[2]], -1, (0, 0, 255), 2)
        if green is not None:
            cv2.circle(vis, (green[0], green[1]), 6, (0, 255, 0), -1)
            cv2.drawContours(vis, [green[2]], -1, (0, 255, 0), 2)

        if red and green:
            rx, ry, _ = red
            gx, gy, _ = green

            # 각 부표 중심에서 깊이 취득
            d_red   = median_depth_from_window(depth, rx, ry, DEPTH_WIN)
            d_green = median_depth_from_window(depth, gx, gy, DEPTH_WIN)
            if (d_red is None) or (d_green is None):
                cv2.putText(vis, "Depth missing", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow("Gate Center (RGB+Depth)", vis)
                cv2.waitKey(1)
                return

            # 3D 좌표 (camera frame)
            red3d   = self.deproject_to_3d(rx, ry, d_red)
            green3d = self.deproject_to_3d(gx, gy, d_green)

            # 두 3D 점의 중점 (게이트 중심의 3D)
            cx = (red3d[0] + green3d[0]) * 0.5
            cy = (red3d[1] + green3d[1]) * 0.5
            cz = (red3d[2] + green3d[2]) * 0.5

            # 카메라(원점)과의 실제 3D 거리
            dist = math.sqrt(cx*cx + cy*cy + cz*cz)

            # ===== 퍼블리시: geometry_msgs/Point (x,y,z) =====
            msg = Point()
            msg.x, msg.y, msg.z = cx, cy, cz
            self.pub_xyz.publish(msg)

            # ===== 화면 HUD =====
            mx, my = int((rx + gx) * 0.5), int((ry + gy) * 0.5)
            cv2.line(vis, (rx, ry), (gx, gy), (0, 255, 255), 2)
            cv2.circle(vis, (mx, my), 6, (0, 255, 255), -1)
            hud = f"GateDist={dist:.2f}m | XYZ=({cx:.2f},{cy:.2f},{cz:.2f})"
            cv2.putText(vis, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
            self.get_logger().info(f"📤 /gate_center_xyz  x={cx:.2f}, y={cy:.2f}, z={cz:.2f} (dist={dist:.2f}m)")

        else:
            cv2.putText(vis, "Gate Not Found", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Gate Center (RGB+Depth)", vis)
        cv2.waitKey(1)

    def destroy(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = GateCenterPublisher()
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
