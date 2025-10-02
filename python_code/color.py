import cv2
import numpy as np
import time
from collections import deque

class StrictColorNavigator:
    def __init__(self, camera_index=None):
        print("🚢 Initializing Strict Color Navigator...")

        # 자동 카메라 검색 또는 수동 인덱스 사용
        if camera_index is None:
            self.cap = self.find_camera()
        else:
            self.cap = cv2.VideoCapture(camera_index)
        
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
            'green': None,
            'red': None
        }

        self.navigation_active = False
        self.target_path = None

        # 화면 설정
        self.display_width = 640
        self.display_height = 480

        print("✅ Navigator initialized with STRICT color detection!")

    def find_camera(self):
        """사용 가능한 카메라 자동 검색"""
        print("🔍 Searching for available cameras...")
        for index in range(5):
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    print(f"✅ Camera found at index {index}")
                    return cap
                cap.release()
        
        print("❌ No available camera found!")
        exit(1)

    def setup_strict_color_ranges(self):
        """매우 엄격한 색상 범위 설정"""
        # 초록색: 연두~진녹색 모든 초록 계열 포함
        self.green_lower = np.array([30, 40, 40])
        self.green_upper = np.array([90, 255, 255])

        # 빨간색: 범위 축소 (더 정확하게)
        self.red_lower1 = np.array([0, 100, 100])
        self.red_upper1 = np.array([12, 255, 255])
        self.red_lower2 = np.array([168, 100, 100])
        self.red_upper2 = np.array([180, 255, 255])

        print("🎨 현재 색상 설정: STRICT (순수 색상만 검출)")
        print(f"   초록색 범위: H[{self.green_lower[0]}-{self.green_upper[0]}] S[{self.green_lower[1]}-255] V[{self.green_lower[2]}-255]")
        print(f"   빨간색 범위: H[0-{self.red_upper1[0]}|{self.red_lower2[0]}-180] S[{self.red_lower1[1]}-255] V[{self.red_lower1[2]}-255]")

    def detect_cones(self, color_image, color_type):
        """매우 엄격한 콘 검출"""
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

        # 가우시안 블러로 노이즈 제거
        hsv = cv2.GaussianBlur(hsv, (5, 5), 0)

        # 색상 마스크 생성
        if color_type == 'green':
            mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
        else:  # red
            mask1 = cv2.inRange(hsv, self.red_lower1, self.red_upper1)
            mask2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)
            mask = cv2.bitwise_or(mask1, mask2)

        # 강력한 노이즈 제거
        kernel_small = np.ones((3, 3), np.uint8)
        kernel_large = np.ones((7, 7), np.uint8)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)

        # 컨투어 찾기
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cones = []

        for contour in contours:
            area = cv2.contourArea(contour)

            if area > 800:
                x, y, w, h = cv2.boundingRect(contour)

                aspect_ratio = h / w if w > 0 else 0
                if 0.8 < aspect_ratio < 3.0:

                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        if circularity > 0.3:

                            center_x = x + w // 2
                            center_y = y + h // 2

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

    def update_stable_zone(self, cone, color):
        """안정화 영역 업데이트"""
        zone = self.stable_zones.get(color)
        
        if zone is None:
            # 새로운 안정화 영역 생성
            self.stable_zones[color] = {
                'center': cone['pixel_pos'],
                'radius': 50,
                'confidence': 1
            }
        else:
            # 기존 영역과의 거리 계산
            distance = np.sqrt(
                (cone['pixel_pos'][0] - zone['center'][0]) ** 2 +
                (cone['pixel_pos'][1] - zone['center'][1]) ** 2
            )
            
            if distance <= zone['radius']:
                # 영역 내부: 신뢰도 증가
                zone['confidence'] = min(10, zone['confidence'] + 1)
                # 중심 점진적 업데이트
                alpha = 0.3
                zone['center'] = (
                    int(zone['center'][0] * (1 - alpha) + cone['pixel_pos'][0] * alpha),
                    int(zone['center'][1] * (1 - alpha) + cone['pixel_pos'][1] * alpha)
                )
            else:
                # 영역 외부: 신뢰도 감소
                zone['confidence'] = max(0, zone['confidence'] - 1)
                if zone['confidence'] == 0:
                    # 신뢰도 0이면 새 영역으로 리셋
                    self.stable_zones[color] = {
                        'center': cone['pixel_pos'],
                        'radius': 50,
                        'confidence': 1
                    }

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
                distance = np.sqrt((cone_pos[0] - zone_center[0]) ** 2 +
                                   (cone_pos[1] - zone_center[1]) ** 2)

                if distance <= stable_zone['radius']:
                    stability_bonus = 0.5 * (stable_zone['confidence'] / 10.0)
                    return base_score + stability_bonus

            return base_score

        return max(cones, key=score_cone)

    def run(self):
        frame_count = 0
        fps_start = time.time()
        
        # 최근 검출 결과 저장
        best_green = None
        best_red = None

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ 프레임을 읽을 수 없습니다!")
                    break

                # 색상 검출 (10프레임마다)
                if frame_count % 10 == 0:
                    green_cones = self.detect_cones(frame, 'green')
                    red_cones = self.detect_cones(frame, 'red')

                    best_green = self.get_best_cone(green_cones)
                    best_red = self.get_best_cone(red_cones)

                    if best_green:
                        print(f"🟢 초록색 인식됨! 거리: {best_green['distance']:.1f}m, 면적: {best_green['area']:.0f}")
                        self.cone_history['green'].append(best_green)
                        self.update_stable_zone(best_green, 'green')
                    else:
                        if len(self.cone_history['green']) > 0:
                            print("🟢 초록색 사라짐!")
                        self.cone_history['green'].clear()

                    if best_red:
                        print(f"🔴 빨간색 인식됨! 거리: {best_red['distance']:.1f}m, 면적: {best_red['area']:.0f}")
                        self.cone_history['red'].append(best_red)
                        self.update_stable_zone(best_red, 'red')
                    else:
                        if len(self.cone_history['red']) > 0:
                            print("🔴 빨간색 사라짐!")
                        self.cone_history['red'].clear()

                    # 네비게이션 로직
                    if best_green and best_red:
                        green_pixel = best_green['pixel_pos']
                        red_pixel = best_red['pixel_pos']

                        mid_pixel_x = (green_pixel[0] + red_pixel[0]) // 2
                        screen_center = self.display_width // 2
                        offset = mid_pixel_x - screen_center

                        if offset < -50:
                            print(f"⬅️ 왼쪽으로 이동! (오프셋: {offset}px)")
                        elif offset > 50:
                            print(f"➡️ 오른쪽으로 이동! (오프셋: {offset}px)")
                        else:
                            print(f"✅ 중앙 유지! (오프셋: {offset}px)")
                    elif best_green:
                        print("⚠️ 빨간색 콘 없음 - 초록색만 추적 중")
                    elif best_red:
                        print("⚠️ 초록색 콘 없음 - 빨간색만 추적 중")
                    else:
                        print("⚠️ 콘 미검출 - 대기 중...")

                # FPS 계산
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
                    
        except KeyboardInterrupt:
            print("\n🛑 사용자에 의해 중단됨")
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            print("✅ Navigator stopped")


def main():
    print("🚢 Starting Strict Color Cone Navigator...")
    print("💡 Tip: 'q' 키를 눌러 종료할 수 있습니다")
    
    # 카메라 인덱스를 지정하려면: navigator = StrictColorNavigator(camera_index=0)
    navigator = StrictColorNavigator()  # 자동 검색
    navigator.run()


if __name__ == '__main__':
    main()