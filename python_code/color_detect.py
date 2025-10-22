# color_gate_demo.py
import cv2
import numpy as np
import time
import argparse

# --------------------------
# 공통 유틸
# --------------------------
def connected_component_center(mask, min_area=150):
    """라벨링으로 가장 그럴듯한 블롭 1개 선택."""
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return {'detected': False, 'area': 0, 'coverage': 0.0}

    H, W = mask.shape[:2]
    best = None
    best_score = -1
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        aspect = w / (h + 1e-6)          # 세장비
        circ = 1.0 - abs(np.log(aspect)) # aspect=1에 가까울수록 큼
        score = area * circ
        if score > best_score:
            best_score = score
            best = (centroids[i], area)

    if best is None:
        return {'detected': False, 'area': 0, 'coverage': 0.0}

    (cx, cy), area = best
    coverage = (area / float(H * W)) * 100.0
    return {'detected': True, 'center': (int(cx), int(cy)), 'area': int(area), 'coverage': float(coverage)}

def draw_detection_debug(frame, det, color_bgr, label):
    if det.get('detected'):
        cv2.circle(frame, det['center'], 12, color_bgr, -1)
        cv2.putText(frame, f"{label}:{det['coverage']:.1f}%",
                    (det['center'][0]-40, det['center'][1]-18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)

# --------------------------
# 모드 1: HSV 방식
# --------------------------
def detect_by_hsv(bgr, green_hsv=None, red_hsv=None):
    """
    green_hsv, red_hsv: ((low1, up1), (low2, up2)) 형태도 허용(빨강처럼 두 구간)
    초록은 기본 한 구간만 사용.
    """
    # 성능/안정성: BGR -> blur -> HSV
    blur = cv2.GaussianBlur(bgr, (5,5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # 초록: 보통 H 35~85 권장
    if green_hsv is None:
        green_low = np.array([35, 80, 60], np.uint8)
        green_up  = np.array([85, 255, 255], np.uint8)
        mask_g = cv2.inRange(hsv, green_low, green_up)
    else:
        (low, up) = green_hsv
        mask_g = cv2.inRange(hsv, np.array(low, np.uint8), np.array(up, np.uint8))

    # 빨강: 2구간(0~10, 170~180) 합치기
    if red_hsv is None:
        red_low1, red_up1 = np.array([0, 120, 80], np.uint8),  np.array([10, 255, 255], np.uint8)
        red_low2, red_up2 = np.array([170,120, 80], np.uint8), np.array([180,255,255], np.uint8)
    else:
        (red_low1, red_up1), (red_low2, red_up2) = red_hsv

    mask_r1 = cv2.inRange(hsv, red_low1, red_up1)
    mask_r2 = cv2.inRange(hsv, red_low2, red_up2)
    mask_r  = cv2.bitwise_or(mask_r1, mask_r2)

    kernel = np.ones((5,5), np.uint8)
    mask_g = cv2.morphologyEx(mask_g, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_g = cv2.morphologyEx(mask_g, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask_r = cv2.morphologyEx(mask_r, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_r = cv2.morphologyEx(mask_r, cv2.MORPH_CLOSE, kernel, iterations=1)

    det_g = connected_component_center(mask_g, min_area=150)
    det_r = connected_component_center(mask_r, min_area=150)
    return det_g, det_r, mask_g, mask_r

# --------------------------
# 모드 2: 색 강조 + 밝기 정규화
# --------------------------
def detect_by_emphasis(bgr, use_otsu=True):
    """
    1) 밝기 정규화(gray-normalization)로 조명 영향 최소화
    2) 초록: G - (R+B)/2, 빨강: R - (G+B)/2
    3) Otsu 또는 상대 임계로 이진화 후 모폴로지
    """
    bgr32 = bgr.astype(np.float32)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) + 1e-6
    gray3 = np.stack([gray, gray, gray], axis=-1)
    norm = np.clip(bgr32 / gray3 * 128.0, 0, 255)  # 평균 밝기 근처로 스케일 고정

    B, G, R = cv2.split(norm)
    green_enh = G - 0.5*(R + B)   # 초록 강조
    red_enh   = R - 0.5*(G + B)   # 빨강 강조
    green_enh[green_enh < 0] = 0
    red_enh[red_enh < 0] = 0

    green_u8 = cv2.normalize(green_enh, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    red_u8   = cv2.normalize(red_enh,   None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if use_otsu:
        thr_g, mask_g = cv2.threshold(green_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr_r, mask_r = cv2.threshold(red_u8,   0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # 약간 임계 올려서 반사 노이즈 더 제거하고 싶으면:
        # mask_g = cv2.threshold(green_u8, int(thr_g*1.05), 255, cv2.THRESH_BINARY)[1]
        # mask_r = cv2.threshold(red_u8,   int(thr_r*1.05), 255, cv2.THRESH_BINARY)[1]
    else:
        # 수동 임계 테스트용
        _, mask_g = cv2.threshold(green_u8, 60, 255, cv2.THRESH_BINARY)
        _, mask_r = cv2.threshold(red_u8,   60, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5,5), np.uint8)
    mask_g = cv2.morphologyEx(mask_g, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_g = cv2.morphologyEx(mask_g, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask_r = cv2.morphologyEx(mask_r, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_r = cv2.morphologyEx(mask_r, cv2.MORPH_CLOSE, kernel, iterations=1)

    det_g = connected_component_center(mask_g, min_area=150)
    det_r = connected_component_center(mask_r, min_area=150)
    return det_g, det_r, mask_g, mask_r, green_u8, red_u8

# --------------------------
# 메인 루프
# --------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0, help="OpenCV VideoCapture camera index")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--start_mode", type=int, default=1, choices=[1,2], help="1=HSV, 2=Emphasis")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERR] 카메라 {args.camera} 열기 실패")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    mode = args.start_mode
    prev_time = time.time()
    fps = 0.0

    print("[i] 실행 중: 1=HSV 모드, 2=색강조 모드, q=종료")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] 프레임 읽기 실패")
            break

        # FPS 계산
        now = time.time()
        dt = now - prev_time
        prev_time = now
        fps = 0.9*fps + 0.1*(1.0/dt) if dt > 0 else fps

        vis = frame.copy()
        h, w = vis.shape[:2]
        cx = w // 2

        if mode == 1:
            det_g, det_r, mask_g, mask_r = detect_by_hsv(frame)
            mode_name = "Mode 1: HSV"
        else:
            det_g, det_r, mask_g, mask_r, gmap, rmap = detect_by_emphasis(frame, use_otsu=True)
            mode_name = "Mode 2: Emphasis"

        # 디버그 그리기
        draw_detection_debug(vis, det_g, (0,255,0), "G")
        draw_detection_debug(vis, det_r, (0,0,255), "R")

        # 두 개 다 있으면 중간점
        if det_g.get('detected') and det_r.get('detected'):
            mid_x = (det_g['center'][0] + det_r['center'][0]) // 2
            mid_y = (det_g['center'][1] + det_r['center'][1]) // 2
            cv2.line(vis, det_g['center'], det_r['center'], (255,255,255), 2)
            cv2.circle(vis, (mid_x, mid_y), 10, (255,255,0), -1)
            # 좌/우 오프셋 시각화
            cv2.line(vis, (cx, 0), (cx, h), (128,128,128), 2)
            off = (mid_x - cx) / float(cx)
            cv2.putText(vis, f"OFFSET: {off:+.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        else:
            cv2.line(vis, (cx, 0), (cx, h), (128,128,128), 2)

        # 텍스트
        cv2.putText(vis, f"{mode_name} | FPS {fps:4.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.imshow("Color Gate - View", vis)
        cv2.imshow("Mask Green", mask_g)
        cv2.imshow("Mask Red", mask_r)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('1'):
            mode = 1
        elif key == ord('2'):
            mode = 2
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
