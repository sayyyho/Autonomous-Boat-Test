"""
KABOAT 콘 검출기 훈련
green_cone, red_cone만 집중 학습
"""

from ultralytics import YOLO
import yaml
import os
from pathlib import Path
import torch
import shutil


def detect_device():
    if torch.cuda.is_available():
        device = '0'
        device_name = f"GPU ({torch.cuda.get_device_name(0)})"
        print(f"✅ GPU: {device_name}")
    else:
        device = 'cpu'
        device_name = "CPU"
        print(f"⚠️  CPU 모드")
    return device, device_name


def create_cone_only_yaml(original_dataset_path: str, output_path: str = './cone_only'):
    """
    green_cone, red_cone만 포함하는 data.yaml 생성
    """
    original_path = Path(original_dataset_path)
    output_path = Path(output_path)
    
    print("=" * 60)
    print("🔧 콘 전용 데이터셋 생성")
    print("=" * 60)
    
    # 출력 폴더 생성
    output_path.mkdir(exist_ok=True)
    
    # 원본 data.yaml 읽기
    with open(original_path / 'data.yaml', 'r') as f:
        original_config = yaml.safe_load(f)
    
    print(f"원본 클래스: {original_config['names']}")
    print(f"원본 클래스 수: {original_config['nc']}")
    
    # green_cone, red_cone의 인덱스 찾기
    all_classes = original_config['names']
    green_cone_idx = all_classes.index('green_cone')
    red_cone_idx = all_classes.index('red_cone')
    
    print(f"\n🎯 선택된 클래스:")
    print(f"   green_cone (원본 인덱스: {green_cone_idx})")
    print(f"   red_cone (원본 인덱스: {red_cone_idx})")
    
    # 새 data.yaml 생성
    new_config = {
        'path': str(original_path.absolute()),  # 절대 경로 사용
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images' if (original_path / 'test').exists() else None,
        'nc': 2,  # green_cone, red_cone
        'names': ['green_cone', 'red_cone'],  # 0: green_cone, 1: red_cone
        'original_indices': {
            'green_cone': green_cone_idx,
            'red_cone': red_cone_idx
        }
    }
    
    # test 없으면 제거
    if new_config['test'] is None:
        del new_config['test']
    
    # 저장
    yaml_path = output_path / 'data_cone_only.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(new_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"\n✅ 생성 완료: {yaml_path}")
    print(f"\n📋 새 data.yaml 내용:")
    print(f"   path: {new_config['path']}")
    print(f"   train: {new_config['train']}")
    print(f"   val: {new_config['val']}")
    print(f"   nc: {new_config['nc']}")
    print(f"   names: {new_config['names']}")
    print("=" * 60)
    
    return str(yaml_path), (green_cone_idx, red_cone_idx)


def filter_labels_for_cones(dataset_path: str, green_idx: int, red_idx: int):
    """
    라벨 파일에서 green_cone, red_cone만 필터링
    (실제로는 YOLO가 알아서 처리하므로 선택사항)
    """
    dataset_path = Path(dataset_path)
    
    print("\n💡 팁: YOLO는 지정된 클래스만 자동으로 학습합니다")
    print("   라벨 파일 수정 불필요!")
    
    # 통계만 출력
    splits = ['train', 'valid']
    for split in splits:
        label_dir = dataset_path / split / 'labels'
        if not label_dir.exists():
            continue
        
        total_objects = 0
        cone_objects = 0
        
        for label_file in label_dir.glob('*.txt'):
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) > 0:
                        cls_idx = int(parts[0])
                        total_objects += 1
                        if cls_idx in [green_idx, red_idx]:
                            cone_objects += 1
        
        print(f"\n📊 {split}:")
        print(f"   전체 객체: {total_objects}")
        print(f"   콘 객체: {cone_objects} ({cone_objects/total_objects*100:.1f}%)")


def check_dataset(dataset_path: str):
    dataset_path = Path(dataset_path)
    
    print("=" * 60)
    print("📁 데이터셋 확인")
    print("=" * 60)
    
    required = {
        'data.yaml': dataset_path / 'data.yaml',
        'train': dataset_path / 'train',
        'valid': dataset_path / 'valid',
    }
    
    all_exist = True
    for name, path in required.items():
        exists = path.exists()
        status = "✅" if exists else "❌"
        print(f"{status} {name}: {path}")
        all_exist = all_exist and exists
    
    if not all_exist:
        print("\n⚠️  필수 파일/폴더 없음!")
        return False
    
    try:
        train_imgs = list((dataset_path / 'train' / 'images').glob('*.jpg')) + \
                     list((dataset_path / 'train' / 'images').glob('*.png'))
        valid_imgs = list((dataset_path / 'valid' / 'images').glob('*.jpg')) + \
                     list((dataset_path / 'valid' / 'images').glob('*.png'))
        
        print(f"\n📊 이미지:")
        print(f"   Train: {len(train_imgs)}")
        print(f"   Valid: {len(valid_imgs)}")
        print(f"   Total: {len(train_imgs) + len(valid_imgs)}")
    except Exception as e:
        print(f"\n⚠️  오류: {e}")
    
    try:
        with open(dataset_path / 'data.yaml', 'r') as f:
            config = yaml.safe_load(f)
            print(f"\n📋 원본 클래스: {config.get('names', 'N/A')}")
    except Exception as e:
        print(f"⚠️  yaml 오류: {e}")
    
    print("=" * 60)
    return True


def train_cone_detector(
    dataset_path: str = './docking',
    model_size: str = 'n',
    epochs: int = 100,
    img_size: int = 640,
    batch_size: int = 16,
    project_name: str = 'kaboat_cone_only',
    device: str = 'auto'
):
    """
    콘 검출기 훈련 (green_cone, red_cone만)
    """
    
    # 데이터셋 확인
    if not check_dataset(dataset_path):
        return None
    
    # 콘 전용 yaml 생성
    cone_yaml_path, (green_idx, red_idx) = create_cone_only_yaml(dataset_path)
    
    # 통계 출력
    filter_labels_for_cones(dataset_path, green_idx, red_idx)
    
    # 디바이스 감지
    if device == 'auto':
        device, device_name = detect_device()
    else:
        device_name = device
    
    # CPU 배치 조정
    if device == 'cpu' and batch_size > 8:
        batch_size = 8
        print(f"\n⚠️  CPU: 배치 → {batch_size}")
    
    print("\n" + "=" * 60)
    print("🚀 콘 검출기 훈련 시작")
    print("=" * 60)
    print(f"프로젝트: {project_name}")
    print(f"대상 클래스: green_cone, red_cone")
    print(f"모델: YOLOv8{model_size}")
    print(f"에포크: {epochs}")
    print(f"배치: {batch_size}")
    print(f"디바이스: {device_name}")
    print("=" * 60)
    
    model = YOLO(f'yolov8{model_size}.pt')
    
    try:
        results = model.train(
            data=cone_yaml_path,  # 콘 전용 yaml 사용!
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            name=project_name,
            device=device,
            
            # 최적화
            patience=50,
            save=True,
            save_period=10,
            
            # Augmentation (콘 특화)
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=20,      # 회전 증강
            translate=0.1,
            scale=0.5,
            fliplr=0.5,      # 좌우 반전
            mosaic=1.0,
            
            # 성능
            optimizer='AdamW',
            lr0=0.01,
            lrf=0.01,
            cos_lr=True,
            workers=4 if device == 'cpu' else 8,
        )
        
        print("\n" + "=" * 60)
        print("✅ 훈련 완료!")
        print("=" * 60)
        print(f"📁 결과: runs/detect/{project_name}/")
        print(f"🏆 best.pt: runs/detect/{project_name}/weights/best.pt")
        print(f"📊 그래프: runs/detect/{project_name}/results.png")
        print("=" * 60)
        
        print("\n🎯 실전 사용:")
        print("```python")
        print(f"model = YOLO('runs/detect/{project_name}/weights/best.pt')")
        print("results = model('test.jpg')")
        print("for r in results:")
        print("    for box in r.boxes:")
        print("        cls = r.names[int(box.cls[0])]")
        print("        if cls == 'green_cone':")
        print("            print('초록 콘 발견!')")
        print("        elif cls == 'red_cone':")
        print("            print('빨간 콘 발견!')")
        print("```")
        
        return results
        
    except KeyboardInterrupt:
        print("\n⚠️  중단")
        return None
    except Exception as e:
        print(f"\n❌ 오류: {e}")
        import traceback
        traceback.print_exc()
        return None


def validate_cone_model(model_path: str, data_yaml: str):
    """콘 모델 검증"""
    print("\n" + "=" * 60)
    print("📊 콘 모델 검증")
    print("=" * 60)
    
    try:
        model = YOLO(model_path)
        results = model.val(data=data_yaml)
        
        print(f"\n📈 성능:")
        print(f"   mAP50: {results.box.map50:.3f}")
        print(f"   mAP50-95: {results.box.map:.3f}")
        print(f"   Precision: {results.box.p:.3f}")
        print(f"   Recall: {results.box.r:.3f}")
        
        # 클래스별 성능
        print(f"\n📊 클래스별:")
        for i, name in enumerate(results.names.values()):
            print(f"   {name}: mAP50 = {results.box.maps[i]:.3f}")
        
        map50 = results.box.map50
        if map50 >= 0.9:
            print("\n   🌟 훌륭함!")
        elif map50 >= 0.7:
            print("\n   ✅ 양호함")
        elif map50 >= 0.5:
            print("\n   ⚠️  개선 필요")
        else:
            print("\n   ❌ 재훈련 권장")
        
        print("=" * 60)
        return results
        
    except Exception as e:
        print(f"❌ 오류: {e}")
        return None


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='KABOAT 콘 검출기 (green_cone, red_cone)')
    parser.add_argument('--dataset', type=str, default='./yl')
    parser.add_argument('--model', type=str, default='n', choices=['n', 's', 'm'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--name', type=str, default='kaboat_cone_only')
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'check', 'validate'])
    parser.add_argument('--weights', type=str, 
                       default='runs/detect/kaboat_cone_only/weights/best.pt')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("🖥️  시스템")
    print("=" * 60)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    print("=" * 60)
    
    if args.mode == 'check':
        check_dataset(args.dataset)
        # 콘 전용 yaml도 생성
        create_cone_only_yaml(args.dataset)
        
    elif args.mode == 'train':
        train_cone_detector(
            dataset_path=args.dataset,
            model_size=args.model,
            epochs=args.epochs,
            img_size=args.img_size,
            batch_size=args.batch,
            project_name=args.name,
            device=args.device
        )
        
    elif args.mode == 'validate':
        yaml_path = './cone_only/data_cone_only.yaml'
        validate_cone_model(args.weights, yaml_path)


if __name__ == '__main__':
    main()