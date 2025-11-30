"""
KABOAT 콘 검출 테스트
학습된 모델로 이미지/비디오/카메라 테스트
"""

from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import argparse


def test_on_image(model_path: str, image_path: str, conf: float = 0.5, save: bool = True):
    """이미지에서 콘 검출"""
    print("\n" + "=" * 60)
    print("🖼️  이미지 테스트")
    print("=" * 60)
    print(f"모델: {model_path}")
    print(f"이미지: {image_path}")
    print(f"신뢰도 임계값: {conf}")
    print("=" * 60)
    
    if not Path(image_path).exists():
        print(f"❌ 이미지 없음: {image_path}")
        return
    
    # 모델 로드
    model = YOLO(model_path)
    
    # 추론
    results = model(image_path, conf=conf)
    
    # 결과 처리
    for r in results:
        img = r.orig_img.copy()
        
        print(f"\n🎯 검출 결과:")
        print(f"   총 {len(r.boxes)}개 검출")
        
        if len(r.boxes) == 0:
            print("   ⚠️  콘 검출 안 됨")
            print(f"   💡 신뢰도를 낮춰보세요: --conf 0.3")
        else:
            # 각 검출 결과 처리
            for i, box in enumerate(r.boxes):
                cls_idx = int(box.cls[0])
                cls_name = r.names[cls_idx]
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                print(f"   {i+1}. {cls_name}: {confidence:.3f} at ({x1},{y1})-({x2},{y2})")
                
                # 색상 설정
                if cls_name == 'green_cone':
                    color = (0, 255, 0)  # 초록
                elif cls_name == 'red_cone':
                    color = (0, 0, 255)  # 빨강
                else:
                    color = (255, 255, 255)  # 흰색
                
                # 바운딩 박스 그리기
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                
                # 라벨 배경
                label = f'{cls_name} {confidence:.2f}'
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(img, (x1, y1-label_size[1]-10), (x1+label_size[0], y1), color, -1)
                
                # 라벨 텍스트
                cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (255, 255, 255), 2)
        
        # 저장
        if save:
            output_path = 'cone_detection_result.jpg'
            cv2.imwrite(output_path, img)
            print(f"\n💾 저장: {output_path}")
        
        # 표시
        cv2.imshow('Cone Detection', img)
        print("\n⌨️  아무 키나 누르면 종료...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    print("=" * 60)


def test_on_video(model_path: str, video_path: str, conf: float = 0.5, save: bool = False):
    """비디오에서 콘 검출"""
    print("\n" + "=" * 60)
    print("🎥 비디오 테스트")
    print("=" * 60)
    print(f"모델: {model_path}")
    print(f"비디오: {video_path}")
    print(f"신뢰도: {conf}")
    print("=" * 60)
    
    if not Path(video_path).exists():
        print(f"❌ 비디오 없음: {video_path}")
        return
    
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    
    # 비디오 정보
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n📊 비디오 정보:")
    print(f"   해상도: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   총 프레임: {total_frames}")
    
    # 저장 설정
    if save:
        output_path = 'cone_detection_video.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"   저장: {output_path}")
    
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # 추론
        results = model(frame, conf=conf, verbose=False)
        
        for r in results:
            # 검출 결과 그리기
            for box in r.boxes:
                cls_name = r.names[int(box.cls[0])]
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                color = (0, 255, 0) if cls_name == 'green_cone' else (0, 0, 255)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f'{cls_name} {confidence:.2f}'
                cv2.putText(frame, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # FPS 표시
        cv2.putText(frame, f'Frame: {frame_count}/{total_frames}', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        if save:
            out.write(frame)
        
        cv2.imshow('Cone Detection - Video', frame)
        
        # 'q' 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n⚠️  사용자 중단")
            break
    
    cap.release()
    if save:
        out.release()
        print(f"\n✅ 저장 완료: {output_path}")
    cv2.destroyAllWindows()
    print("=" * 60)


def test_on_webcam(model_path: str, conf: float = 0.5, camera_id: int = 0):
    """웹캠 실시간 콘 검출"""
    print("\n" + "=" * 60)
    print("📹 웹캠 실시간 테스트")
    print("=" * 60)
    print(f"모델: {model_path}")
    print(f"카메라 ID: {camera_id}")
    print(f"신뢰도: {conf}")
    print("=" * 60)
    print("\n💡 조작법:")
    print("   - 'q': 종료")
    print("   - 'c': 스크린샷 저장")
    print("=" * 60)
    
    model = YOLO(model_path)
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"❌ 카메라 열기 실패 (ID: {camera_id})")
        return
    
    frame_count = 0
    screenshot_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ 프레임 읽기 실패")
            break
        
        frame_count += 1
        
        # 추론 (매 프레임)
        results = model(frame, conf=conf, verbose=False)
        
        green_count = 0
        red_count = 0
        
        for r in results:
            for box in r.boxes:
                cls_idx = int(box.cls[0])
                cls_name = r.names[cls_idx]
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                if cls_name == 'green_cone':
                    color = (0, 255, 0)
                    green_count += 1
                else:
                    color = (0, 0, 255)
                    red_count += 1
                
                # 바운딩 박스
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # 라벨
                label = f'{cls_name} {confidence:.2f}'
                cv2.putText(frame, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 통계 표시
        cv2.putText(frame, f'Green: {green_count}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Red: {red_count}', (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f'Frame: {frame_count}', (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imshow('Cone Detection - Webcam', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n✅ 종료")
            break
        elif key == ord('c'):
            screenshot_count += 1
            filename = f'screenshot_{screenshot_count}.jpg'
            cv2.imwrite(filename, frame)
            print(f"📸 스크린샷 저장: {filename}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("=" * 60)


def validate_model(model_path: str, data_yaml: str):
    """모델 검증"""
    print("\n" + "=" * 60)
    print("📊 모델 검증")
    print("=" * 60)
    
    model = YOLO(model_path)
    results = model.val(data=data_yaml, verbose=True)
    
    print(f"\n📈 전체 성능:")
    print(f"   mAP50: {results.box.map50:.3f}")
    print(f"   mAP50-95: {results.box.map:.3f}")
    print(f"   Precision: {results.box.p:.3f}")
    print(f"   Recall: {results.box.r:.3f}")
    
    print(f"\n📊 클래스별 mAP50:")
    for i, name in enumerate(results.names.values()):
        print(f"   {name}: {results.box.maps[i]:.3f}")
    
    if results.box.map50 >= 0.7:
        print("\n   ✅ 성능 좋음!")
    elif results.box.map50 >= 0.5:
        print("\n   ⚠️  개선 필요")
    else:
        print("\n   ❌ 재훈련 권장")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='KABOAT 콘 검출 테스트')
    
    parser.add_argument('--weights', type=str, 
                       default='runs/detect/kaboat_cone_only/weights/best.pt',
                       help='모델 경로')
    parser.add_argument('--mode', type=str, default='image',
                       choices=['image', 'video', 'webcam', 'validate'],
                       help='테스트 모드')
    parser.add_argument('--source', type=str, default='test.jpg',
                       help='이미지/비디오 경로')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='신뢰도 임계값 (0.0-1.0)')
    parser.add_argument('--save', action='store_true',
                       help='결과 저장')
    parser.add_argument('--camera', type=int, default=0,
                       help='카메라 ID (0, 1, ...)')
    parser.add_argument('--data', type=str, 
                       default='./cone_only/data_cone_only.yaml',
                       help='검증용 data.yaml 경로')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("🚢 KABOAT 콘 검출 테스트")
    print("=" * 60)
    
    if args.mode == 'image':
        test_on_image(args.weights, args.source, args.conf, args.save)
    elif args.mode == 'video':
        test_on_video(args.weights, args.source, args.conf, args.save)
    elif args.mode == 'webcam':
        test_on_webcam(args.weights, args.conf, args.camera)
    elif args.mode == 'validate':
        validate_model(args.weights, args.data)


if __name__ == '__main__':
    main()