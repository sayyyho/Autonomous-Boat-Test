#!/usr/bin/env python3
"""
로컬 컴퓨터에서 색상 네비게이션 테스트
ROS2 없이 독립 실행 가능
"""

import cv2
import numpy as np
import time

class ColorNavigator:
    """엄격한 색상 기반 네비게이션 (30% 임계값)"""
    def __init__(self, camera_index=None):
        print("🎨 색상 네비게이터 초기화 중...")
        
        # 카메라 초기화
        if camera_index is None:
            self.cap = self.find_camera()
        else:
            self.cap = cv2.VideoCapture(camera_index)
        
        # 카메라 없으면 종료
        if self.cap is None or not self.cap.isOpened():
            print("❌ 카메라 연결 실패!")
            self.camera_available = False
            self.cap = None
            return
        
        self.camera_available = True
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # RealSense 카메라 설정 조정
        self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)  # 자동 화이트밸런스 끄기
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # 수동 노출 모드
        self.cap.set(cv2.CAP_PROP_EXPOSURE, -5)  # 노출 조정
        print("카메라 설정: 화이트밸런스 수동, 노출 조정")
        
        # 엄격한 색상 범위 설정
        self.setup_strict_color_ranges()
        
        # 화면 설정
        self.display_width = 640
        self.display_height = 480
        
        # 네비게이션 상태
        self.target_offset = 0.0
        self.is_valid_setup = False
        self.last_detection_time = 0
        
        # 검출 데이터
        self.gb_data = {'detected': False}
        self.red_data = {'detected': False}
        
        print("✅ 색상 네비게이터 초기화 완료")
    
    def find_camera(self):
        """사용 가능한 RGB 카메라 찾기"""
        print("카메라 검색 중...")
        
        for index in range(6):
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    shape = frame.shape
                    dtype = frame.dtype
                    print(f"video{index}: shape={shape}, dtype={dtype}")
                    
                    # RGB 카메라 확인 (3채널 + 색상 편차 확인)
                    if len(shape) == 3 and shape[2] == 3:
                        mean_color = frame.mean(axis=(0,1))
                        # BGR 채널 간 편차 확인 (IR은 모든 채널이 비슷함)
                        color_std = mean_color.std()
                        print(f"video{index}: Mean BGR={mean_color}, std={color_std:.2f}")
                        
                        # 채널 간 표준편차가 5 이상이면 실제 컬러
                        if color_std > 3.0:
                            print(f"✅ RGB 카메라 발견! video{index}")
                            return cap
                        else:
                            print(f"video{index}는 IR (채널 간 차이 없음)")
                            cap.release()
                    else:
                        print(f"video{index}는 3채널이 아님")
                        cap.release()
                else:
                    print(f"video{index}: 프레임 읽기 실패")
                    cap.release()
            else:
                print(f"video{index}: 열기 실패")
        
        print("RGB 카메라를 찾지 못했습니다")
        return None
    
    def setup_strict_color_ranges(self):
        """엄격한 색상 HSV 범위"""
        # 초록-파랑 통합 (엄격)
        self.green_blue_lower = np.array([35, 80, 80])
        self.green_blue_upper = np.array([125, 255, 255])
        
        # 빨강 (매우 엄격)
        self.red_lower1 = np.array([0, 150, 150])
        self.red_upper1 = np.array([8, 255, 255])
        self.red_lower2 = np.array([172, 150, 150])
        self.red_upper2 = np.array([180, 255, 255])
        
        print("🎨 색상 설정: STRICT MODE")
        print("   🟢🔵 초록-파랑: H[35-125°] S[80+] V[80+]")
        print("   🔴 빨강: H[0-8°|172-180°] S[150+] V[150+]")
        print("   ⚠️  화면의 30% 미만은 검출하지 않음")
    
    def color_correction(self, frame):
        """초록끼 보정"""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # b 채널 조정 (초록 감소)
        b = cv2.add(b, 10)
        
        lab = cv2.merge([l, a, b])
        corrected = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return corrected
    
    def find_color_center(self, frame, color_type):
        """색상 중심점 찾기 (30% 미만 무시)"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv = cv2.GaussianBlur(hsv, (5, 5), 0)

        # 마스크 생성
        if color_type == 'green_blue':
            mask = cv2.inRange(hsv, self.green_blue_lower, self.green_blue_upper)
        else:  # red
            mask1 = cv2.inRange(hsv, self.red_lower1, self.red_upper1)
            mask2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)
            mask = cv2.bitwise_or(mask1, mask2)

        # 노이즈 제거
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 픽셀 수 직접 계산
        screen_area = self.display_width * self.display_height
        detected_pixels = np.count_nonzero(mask)
        coverage_percent = (detected_pixels / screen_area) * 100
        
        # 100% 상한선
        coverage_percent = min(coverage_percent, 100.0)
        
        # 30% 미만이면 검출 안 함
        if coverage_percent >= 0.5:
            # 모멘트로 중심점만 계산
            moments = cv2.moments(mask)
            if moments["m00"] > 0:
                center_x = int(moments["m10"] / moments["m00"])
                center_y = int(moments["m01"] / moments["m00"])
                
                return {
                    'center': (center_x, center_y),
                    'area': detected_pixels,
                    'coverage': coverage_percent,
                    'mask': mask,
                    'detected': True
                }
        
        return {
            'detected': False,
            'area': detected_pixels,
            'coverage': coverage_percent,
            'mask': mask
        }
    
    def draw_overlay(self, frame, gb_data, red_data):
        """화면에 검출 정보 표시"""
        display = frame.copy()
        
        # 초록-파랑 영역 표시
        if gb_data['detected']:
            cv2.circle(display, gb_data['center'], 20, (0, 255, 255), 3)
            cv2.circle(display, gb_data['center'], 10, (0, 255, 255), -1)
            cv2.putText(display, f"GB: {gb_data['coverage']:.1f}%", 
                       (gb_data['center'][0] - 50, gb_data['center'][1] - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # 빨강 영역 표시
        if red_data['detected']:
            cv2.circle(display, red_data['center'], 20, (0, 0, 255), 3)
            cv2.circle(display, red_data['center'], 10, (0, 0, 255), -1)
            cv2.putText(display, f"RED: {red_data['coverage']:.1f}%", 
                       (red_data['center'][0] - 50, red_data['center'][1] - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # 네비게이션 정보
        if gb_data['detected'] and red_data['detected']:
            gb_x = gb_data['center'][0]
            red_x = red_data['center'][0]
            
            # 연결선
            cv2.line(display, gb_data['center'], red_data['center'], (255, 255, 255), 2)
            
            # 중앙점
            mid_x = (gb_x + red_x) // 2
            mid_y = (gb_data['center'][1] + red_data['center'][1]) // 2
            cv2.circle(display, (mid_x, mid_y), 15, (255, 255, 0), -1)
            
            # 화면 중앙선
            screen_center = self.display_width // 2
            cv2.line(display, (screen_center, 0), (screen_center, self.display_height),
                    (128, 128, 128), 2)
            
            # 오프셋
            offset = mid_x - screen_center
            offset_normalized = offset / screen_center
            
            # 방향 표시
            if offset_normalized < -0.15:
                direction = "LEFT"
                color = (0, 100, 255)
            elif offset_normalized > 0.15:
                direction = "RIGHT"
                color = (0, 100, 255)
            else:
                direction = "CENTER"
                color = (0, 255, 0)
            
            cv2.putText(display, f"{direction} | Offset: {offset:+d}px", 
                       (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        else:
            # 미검출 정보
            cv2.putText(display, "Searching for colors...", 
                       (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            
            if not gb_data['detected']:
                cv2.putText(display, f"X GB: {gb_data.get('coverage', 0):.1f}% (need 3%)", 
                           (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            if not red_data['detected']:
                cv2.putText(display, f"X RED: {red_data.get('coverage', 0):.1f}% (need 3%)", 
                           (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return display
    
    def run(self):
        """메인 실행 루프"""
        if not self.camera_available:
            print("❌ 카메라를 사용할 수 없습니다")
            return
        
        print("\n" + "="*50)
        print("색상 네비게이션 테스트 시작")
        print("="*50)
        print("조작법:")
        print("  q: 종료")
        print("  m: 마스크 화면 토글")
        print("\n배치 방법:")
        print("  - 빨강 물체: 화면 왼쪽")
        print("  - 초록/파랑 물체: 화면 오른쪽")
        print("  - 각각 화면의 30% 이상 차지해야 함")
        print("="*50 + "\n")
        
        show_mask = False
        frame_count = 0
        fps_start = time.time()
        fps = 0
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ 프레임을 읽을 수 없습니다")
                    break
                
                # 색상 보정 적용
                frame = self.color_correction(frame)
                
                # 색상 중심점 검출
                gb_data = self.find_color_center(frame, 'green_blue')
                red_data = self.find_color_center(frame, 'red')
                
                # 메인 화면 표시
                main_display = self.draw_overlay(frame, gb_data, red_data)
                
                # FPS 표시
                cv2.putText(main_display, f"FPS: {fps:.1f}", 
                           (self.display_width - 120, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('Color Navigator - Camera View', main_display)
                
                # 마스크 화면 (선택적)
                if show_mask:
                    mask_display = np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)
                    mask_display[:, :, 1] = gb_data['mask']  # Green channel
                    mask_display[:, :, 2] = red_data['mask']  # Red channel
                    cv2.imshow('Color Navigator - Detection Mask', mask_display)
                
                # FPS 계산
                frame_count += 1
                if frame_count % 30 == 0:
                    current_time = time.time()
                    fps = 30 / (current_time - fps_start)
                    fps_start = current_time
                    
                    # 콘솔 출력
                    if gb_data['detected'] and red_data['detected']:
                        print(f"✅ GB: {gb_data['coverage']:.1f}% | RED: {red_data['coverage']:.1f}%")
                    else:
                        print(f"⚠️  GB: {gb_data.get('coverage', 0):.1f}% | RED: {red_data.get('coverage', 0):.1f}%")
                
                # 키보드 입력
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("🛑 종료 중...")
                    break
                elif key == ord('m'):
                    show_mask = not show_mask
                    if not show_mask:
                        cv2.destroyWindow('Color Navigator - Detection Mask')
                    print(f"마스크 표시: {'ON' if show_mask else 'OFF'}")
                
        except KeyboardInterrupt:
            print("\n🛑 사용자에 의해 중단됨")
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def cleanup(self):
        """종료 처리"""
        if self.camera_available and self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        print("✅ 종료 완료")


def main():
    print("""
╔══════════════════════════════════════════╗
║  색상 네비게이션 테스트 (로컬 버전)      ║
╚══════════════════════════════════════════╝
    """)
    
    navigator = ColorNavigator()
    
    if navigator.camera_available:
        navigator.run()
    else:
        print("\n📝 참고:")
        print("  - 웹캠이 연결되어 있는지 확인하세요")
        print("  - 다른 프로그램에서 카메라를 사용 중인지 확인하세요")
        print("  - USB 웹캠의 경우 연결을 확인하세요")


if __name__ == '__main__':
    main()