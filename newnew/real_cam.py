#!/usr/bin/env python3
"""
KABOAT 실시간 카메라 부표 검출
웹캠 또는 USB 카메라로 실시간 검출
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
import glob
import os


def find_latest_model():
    """훈련된 모델 자동 탐색"""
    patterns = [
        './runs/detect/**/weights/best.pt',
        './best.pt',
        './yolov8n.pt',
    ]
    
    for pattern in patterns:
        models = glob.glob(pattern, recursive=True)
        if models:
            latest = max(models, key=os.path.getmtime)
            print(f"✅ 모델 발견: {latest}")
            return latest
    
    print("❌ 모델을 찾을 수 없습니다.")
    print("💡 먼저 훈련을 실행해주세요:")
    print("   python3 cpu_gpu_train_auto.py --mode train")
    return None


class RealtimeBuoyDetector:
    """실시간 부표 검출기"""
    
    def __init__(self, model_path=None, conf_threshold=0.3, camera_id=0):
        """
        Args:
            model_path: YOLO 모델 경로 (None이면 자동 탐색)
            conf_threshold: 신뢰도 임계값
            camera_id: 카메라 ID (0=기본 웹캠, 1=외장 카메라)
        """
        # 모델 로드
        if model_path is None:
            model_path = find_latest_model()
            if model_path is None:
                raise FileNotFoundError("모델을 찾을 수 없습니다")
        
        print(f"🤖 모델 로딩: {model_path}")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        
        # 카메라 초기화
        print(f"📷 카메라 연결 시도: ID {camera_id}")
        self.cap = cv2.VideoCapture(camera_id)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"카메라를 열 수 없습니다 (ID: {camera_id})")
        
        # 카메라 설정 최적화
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # 성능 측정
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # 통계
        self.detection_history = []
        
        print("✅ 초기화 완료!")
        print(f"   카메라 해상도: {int(self.cap.get(3))}x{int(self.cap.get(4))}")
        print(f"   신뢰도 임계값: {self.conf_threshold}")
    
    def process_frame(self, frame):
        """
        프레임 처리 및 검출
        
        Returns:
            annotated_frame, detections
        """
        # YOLO 추론
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        
        # 결과 분석
        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = r.names[cls_id]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                detections.append({
                    'class': cls_name,
                    'confidence': conf,
                    'bbox': (x1, y1, x2, y2)
                })
            
            # 결과 시각화
            annotated = r.plot()
        
        return annotated, detections
    
    def draw_info_panel(self, frame, detections):
        """정보 패널 그리기"""
        h, w = frame.shape[:2]
        
        # 반투명 배경
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 200), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # FPS 표시
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # 검출 개수
        cv2.putText(frame, f"Detected: {len(detections)}", (20, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # 클래스별 개수
        class_counts = {}
        for det in detections:
            cls = det['class']
            class_counts[cls] = class_counts.get(cls, 0) + 1
        
        y_offset = 110
        for cls_name, count in class_counts.items():
            color = (0, 0, 255) if 'Red' in cls_name else (0, 255, 0)
            cv2.putText(frame, f"{cls_name}: {count}", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 30
        
        # 사용법
        y_offset += 10
        cv2.putText(frame, "Press 'q' to quit", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        cv2.putText(frame, "Press 's' to save", (20, y_offset + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        return frame
    
    def update_fps(self):
        """FPS 계산"""
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        
        if elapsed > 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.start_time = time.time()
    
    def run(self):
        """메인 루프"""
        print("\n" + "=" * 60)
        print("🎥 실시간 검출 시작!")
        print("=" * 60)
        print("사용법:")
        print("  'q' - 종료")
        print("  's' - 현재 프레임 저장")
        print("  '+' - 신뢰도 올리기")
        print("  '-' - 신뢰도 내리기")
        print("=" * 60)
        
        save_count = 0
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ 프레임을 읽을 수 없습니다")
                    break
                
                # 검출
                annotated, detections = self.process_frame(frame)
                
                # 정보 패널
                display = self.draw_info_panel(annotated, detections)
                
                # FPS 업데이트
                self.update_fps()
                
                # 통계 저장
                self.detection_history.append(len(detections))
                if len(self.detection_history) > 100:
                    self.detection_history.pop(0)
                
                # 화면 표시
                cv2.imshow('KABOAT Real-time Detection', display)
                
                # 키보드 입력
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\n👋 종료합니다...")
                    break
                
                elif key == ord('s'):
                    save_count += 1
                    filename = f'capture_{save_count:03d}.jpg'
                    cv2.imwrite(filename, display)
                    print(f"💾 저장: {filename}")
                
                elif key == ord('+') or key == ord('='):
                    self.conf_threshold = min(0.9, self.conf_threshold + 0.05)
                    print(f"🔺 신뢰도: {self.conf_threshold:.2f}")
                
                elif key == ord('-') or key == ord('_'):
                    self.conf_threshold = max(0.1, self.conf_threshold - 0.05)
                    print(f"🔻 신뢰도: {self.conf_threshold:.2f}")
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Ctrl+C 감지. 종료합니다...")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """리소스 정리"""
        self.cap.release()
        cv2.destroyAllWindows()
        
        # 통계 출력
        if self.detection_history:
            avg_det = sum(self.detection_history) / len(self.detection_history)
            print("\n" + "=" * 60)
            print("📊 세션 통계")
            print("=" * 60)
            print(f"평균 FPS: {self.fps:.1f}")
            print(f"평균 검출: {avg_det:.1f}개/프레임")
            print(f"최대 검출: {max(self.detection_history)}개")
            print("=" * 60)


def list_cameras():
    """사용 가능한 카메라 나열"""
    print("\n" + "=" * 60)
    print("📷 카메라 검색 중...")
    print("=" * 60)
    
    available = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                available.append((i, w, h))
                print(f"✅ 카메라 {i}: {w}x{h}")
            cap.release()
    
    if not available:
        print("❌ 사용 가능한 카메라가 없습니다")
    
    print("=" * 60)
    return available


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='KABOAT 실시간 부표 검출')
    
    parser.add_argument('--model', type=str, default=None,
                       help='모델 경로 (기본: 자동 탐색)')
    parser.add_argument('--camera', type=int, default=0,
                       help='카메라 ID (기본: 0)')
    parser.add_argument('--conf', type=float, default=0.3,
                       help='신뢰도 임계값 (기본: 0.3)')
    parser.add_argument('--list-cameras', action='store_true',
                       help='사용 가능한 카메라 나열')
    
    args = parser.parse_args()
    
    # 카메라 나열
    if args.list_cameras:
        list_cameras()
        return
    
    # 검출기 시작
    try:
        detector = RealtimeBuoyDetector(
            model_path=args.model,
            conf_threshold=args.conf,
            camera_id=args.camera
        )
        detector.run()
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        print("\n💡 문제 해결:")
        print("1. 카메라가 연결되어 있나요?")
        print("2. 다른 프로그램이 카메라를 사용 중인가요?")
        print("3. 카메라 목록 확인: python3 realtime_camera.py --list-cameras")
        print("4. 다른 카메라 시도: python3 realtime_camera.py --camera 1")


if __name__ == '__main__':
    main()