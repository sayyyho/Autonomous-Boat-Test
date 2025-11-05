from ultralytics import YOLO
import yaml
import os
from pathlib import Path


def check_dataset_structure(dataset_path: str):
    """데이터셋 구조 검증"""
    dataset_path = Path(dataset_path)
    
    print("=" * 60)
    print("📁 데이터셋 구조 확인")
    print("=" * 60)
    
    # 필수 파일/폴더 확인
    required_items = {
        'data.yaml': dataset_path / 'data.yaml',
        'train': dataset_path / 'train',
        'valid': dataset_path / 'valid',
    }
    
    all_exist = True
    for name, path in required_items.items():
        exists = path.exists()
        status = "✅" if exists else "❌"
        print(f"{status} {name}: {path}")
        all_exist = all_exist and exists
    
    if not all_exist:
        print("\n⚠️  필수 파일/폴더가 없습니다!")
        return False
    
    # 데이터 개수 확인
    try:
        train_images = list((dataset_path / 'train' / 'images').glob('*.jpg')) + \
                      list((dataset_path / 'train' / 'images').glob('*.png'))
        valid_images = list((dataset_path / 'valid' / 'images').glob('*.jpg')) + \
                      list((dataset_path / 'valid' / 'images').glob('*.png'))
        
        print(f"\n📊 데이터 개수:")
        print(f"   Train: {len(train_images)} 이미지")
        print(f"   Valid: {len(valid_images)} 이미지")
        print(f"   Total: {len(train_images) + len(valid_images)} 이미지")
        
        if len(train_images) < 10:
            print("\n⚠️  훈련 데이터가 너무 적습니다 (최소 50개 권장)")
        
    except Exception as e:
        print(f"\n⚠️  데이터 확인 중 오류: {e}")
    
    # data.yaml 내용 확인
    try:
        with open(dataset_path / 'data.yaml', 'r') as f:
            config = yaml.safe_load(f)
            print(f"\n📋 data.yaml 내용:")
            print(f"   클래스 수: {config.get('nc', 'N/A')}")
            print(f"   클래스 이름: {config.get('names', 'N/A')}")
    except Exception as e:
        print(f"\n⚠️  data.yaml 읽기 오류: {e}")
    
    print("=" * 60)
    return True


def train_buoy_detector(
    dataset_path: str = './Red Buoy.v1i.yolov8',
    model_size: str = 'n',      # n, s, m, l, x
    epochs: int = 100,
    img_size: int = 640,
    batch_size: int = 16,
    project_name: str = 'kaboat_red_buoy',
    device: str = '0'           # '0' = GPU, 'cpu' = CPU
):
    """
    빨간 부표 검출기 훈련
    
    Args:
        dataset_path: 데이터셋 경로
        model_size: 모델 크기 (n=nano, s=small, m=medium)
        epochs: 훈련 에포크
        img_size: 입력 이미지 크기
        batch_size: 배치 크기
        project_name: 프로젝트 이름
        device: 사용할 디바이스
    """
    
    # 데이터셋 검증
    if not check_dataset_structure(dataset_path):
        return None
    
    # data.yaml 경로
    data_yaml = os.path.join(dataset_path, 'data.yaml')
    
    print("\n" + "=" * 60)
    print("🚀 YOLO 훈련 시작")
    print("=" * 60)
    print(f"모델: YOLOv8{model_size}")
    print(f"데이터셋: {dataset_path}")
    print(f"에포크: {epochs}")
    print(f"이미지 크기: {img_size}")
    print(f"배치 크기: {batch_size}")
    print(f"디바이스: {device}")
    print("=" * 60)
    
    # 사전훈련 모델 로드
    model = YOLO(f'yolov8{model_size}.pt')
    
    # 훈련 시작
    try:
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            name=project_name,
            device=device,
            
            # 최적화 설정
            patience=50,        # Early stopping
            save=True,
            save_period=10,     # 10 에포크마다 저장
            
            # Augmentation (해상 환경 최적화)
            hsv_h=0.015,        # 색조 변화 (빨간색 유지)
            hsv_s=0.7,          # 채도 변화
            hsv_v=0.4,          # 명도 변화 (조명)
            degrees=15,         # 회전 (파도)
            translate=0.1,      # 이동
            scale=0.5,          # 스케일 (거리 변화)
            shear=0.0,
            perspective=0.0,
            flipud=0.0,         # 상하반전 X (해상)
            fliplr=0.5,         # 좌우반전 O
            mosaic=1.0,
            mixup=0.0,
            
            # 성능 설정
            optimizer='AdamW',
            lr0=0.01,           # 초기 학습률
            lrf=0.01,           # 최종 학습률
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            warmup_momentum=0.8,
            
            # 기타
            cos_lr=True,        # Cosine LR scheduler
            close_mosaic=10,    # 마지막 10 에포크는 mosaic X
            verbose=True,
            seed=0,
            deterministic=True,
        )
        
        print("\n" + "=" * 60)
        print("✅ 훈련 완료!")
        print("=" * 60)
        print(f"📁 결과 저장 위치: runs/detect/{project_name}/")
        print(f"🏆 최고 모델: runs/detect/{project_name}/weights/best.pt")
        print(f"📊 마지막 모델: runs/detect/{project_name}/weights/last.pt")
        print("=" * 60)
        
        return results
        
    except Exception as e:
        print(f"\n❌ 훈련 중 오류 발생: {e}")
        return None


def validate_trained_model(model_path: str, data_yaml: str):
    """훈련된 모델 검증"""
    print("\n" + "=" * 60)
    print("📊 모델 검증")
    print("=" * 60)
    
    model = YOLO(model_path)
    
    # 검증 데이터셋으로 평가
    results = model.val(data=data_yaml)
    
    print(f"\n📈 성능 지표:")
    print(f"   mAP50: {results.box.map50:.3f}")
    print(f"   mAP50-95: {results.box.map:.3f}")
    print(f"   Precision: {results.box.p:.3f}")
    print(f"   Recall: {results.box.r:.3f}")
    print("=" * 60)
    
    return results


def test_on_image(model_path: str, image_path: str):
    """테스트 이미지로 추론"""
    import cv2
    
    print("\n" + "=" * 60)
    print("🖼️  테스트 이미지 추론")
    print("=" * 60)
    
    model = YOLO(model_path)
    
    # 추론
    results = model(image_path, conf=0.5)
    
    # 결과 출력
    for r in results:
        print(f"\n검출된 부표: {len(r.boxes)} 개")
        for i, box in enumerate(r.boxes):
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            print(f"  {i+1}. Class: {cls}, Confidence: {conf:.3f}")
        
        # 시각화
        img = r.plot()
        cv2.imshow('Detection Result', img)
        print("\n⌨️  아무 키나 누르면 종료...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    print("=" * 60)


def export_to_optimized_format(model_path: str, format: str = 'onnx'):
    """
    추론 최적화를 위한 모델 변환
    
    Args:
        model_path: PyTorch 모델 경로
        format: 변환 형식 ('onnx', 'engine', 'tflite' 등)
    """
    print("\n" + "=" * 60)
    print(f"🔄 모델 변환: {format.upper()}")
    print("=" * 60)
    
    model = YOLO(model_path)
    
    try:
        export_path = model.export(format=format)
        print(f"\n✅ 변환 완료: {export_path}")
        print(f"📁 변환된 모델 사용법:")
        print(f"   model = YOLO('{export_path}')")
        print("=" * 60)
        return export_path
    except Exception as e:
        print(f"\n❌ 변환 실패: {e}")
        return None


def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='KABOAT Red Buoy YOLO 훈련')
    
    # 기본 설정
    parser.add_argument('--dataset', type=str, default='./Red Buoy.v1i.yolov8',
                       help='데이터셋 경로')
    parser.add_argument('--model', type=str, default='n',
                       choices=['n', 's', 'm', 'l', 'x'],
                       help='모델 크기 (n=nano, s=small, m=medium, l=large, x=xlarge)')
    
    # 훈련 파라미터
    parser.add_argument('--epochs', type=int, default=100,
                       help='훈련 에포크 수')
    parser.add_argument('--img-size', type=int, default=640,
                       help='입력 이미지 크기')
    parser.add_argument('--batch', type=int, default=16,
                       help='배치 크기')
    parser.add_argument('--device', type=str, default='0',
                       help='디바이스 (0=GPU, cpu=CPU)')
    
    # 실행 모드
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'validate', 'test', 'export', 'check'],
                       help='실행 모드')
    
    # 추가 옵션
    parser.add_argument('--weights', type=str, default='runs/detect/kaboat_red_buoy/weights/best.pt',
                       help='검증/테스트용 모델 경로')
    parser.add_argument('--test-image', type=str, default='test.jpg',
                       help='테스트 이미지 경로')
    parser.add_argument('--export-format', type=str, default='onnx',
                       choices=['onnx', 'engine', 'tflite', 'saved_model'],
                       help='변환 형식')
    
    args = parser.parse_args()
    
    # 모드별 실행
    if args.mode == 'check':
        # 데이터셋 구조만 확인
        check_dataset_structure(args.dataset)
        
    elif args.mode == 'train':
        # 훈련
        train_buoy_detector(
            dataset_path=args.dataset,
            model_size=args.model,
            epochs=args.epochs,
            img_size=args.img_size,
            batch_size=args.batch,
            device=args.device
        )
        
    elif args.mode == 'validate':
        # 검증
        data_yaml = os.path.join(args.dataset, 'data.yaml')
        validate_trained_model(args.weights, data_yaml)
        
    elif args.mode == 'test':
        # 테스트
        test_on_image(args.weights, args.test_image)
        
    elif args.mode == 'export':
        # 변환
        export_to_optimized_format(args.weights, args.export_format)


if __name__ == '__main__':
    main()