import os
import random
import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# ==========================================
# 1. 설정값 정의
# ==========================================
RAW_DATA_DIR = Path("./raw_dataset")
FINAL_DATA_DIR = Path("./dataset_dvc")  # DVC로 관리할 최종 폴더
TARGET_SIZE = (224, 224)
SAMPLES_PER_CLASS = 200

# 분할 비율 (70%, 15%, 15%)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ==========================================
# 2. 캐글 API를 통한 데이터 다운로드 및 압축 해제
# ==========================================
def download_kaggle_dataset():
    print("🚀 캐글 API를 사용하여 Fruits and Vegetables 데이터셋 다운로드 중...")
    # 캐글 API를 사용하여 지정된 디렉토리에 다운로드 및 자동 압축 해제
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.system(f"kaggle datasets download -d kritikseth/fruit-and-vegetable-image-recognition --unzip -p {RAW_DATA_DIR}")
    print("✅ 데이터셋 다운로드 완료!")

# ==========================================
# 3. 데이터셋 가공 및 분할 (리사이즈 + 랜덤 샘플링 + 70/15/15 저장)
# ==========================================
def process_and_split_dataset():
    print("🛠️ 이미지 전처리 및 데이터셋 분할을 시작합니다...")
    
    # 최종 DVC용 디렉토리 트리 생성
    for split in ['train', 'validation', 'test']:
        os.makedirs(FINAL_DATA_DIR / split, exist_ok=True)

    # 원본 데이터셋 구조(train/validation/test 안에 클래스 폴더가 있음)에서 고유 클래스들을 추출
    # 여러 서브폴더에 흩어진 동일 클래스의 이미지들을 통합 관리하기 위해 딕셔너리에 매핑합니다.
    class_images = {}
    
    # 캐글 원본 폴더 내 모든 클래스별 이미지 경로 탐색하여 모으기
    for root, dirs, files in os.walk(RAW_DATA_DIR):
        root_path = Path(root)
        if root_path.parent.name in ['train', 'validation', 'test']:
            class_name = root_path.name
            if class_name not in class_images:
                class_images[class_name] = []
            
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    class_images[class_name].append(root_path / f)

    # 클래스명과 라벨링 매핑 처리 진행
    for class_name, img_paths in tqdm(class_images.items(), desc="클래스별 처리 진행률"):
        if len(img_paths) == 0:
            continue
            
        # 1. 200장 무작위(랜덤) 샘플링 (원본 이미지가 200장보다 적을 경우 확보된 최대치 사용)
        sampled_paths = random.sample(img_paths, min(len(img_paths), SAMPLES_PER_CLASS))
        
        # 섞기
        random.shuffle(sampled_paths)
        
        # 분할 지점 계산
        total_sampled = len(sampled_paths)
        train_end = int(total_sampled * TRAIN_RATIO)
        val_end = train_end + int(total_sampled * VAL_RATIO)
        
        splits = {
            'train': sampled_paths[:train_end],
            'validation': sampled_paths[train_end:val_end],
            'test': sampled_paths[val_end:]
        }
        
        # 2. 이미지 224 리사이즈 적용 및 최종 저장
        for split_name, paths in splits.items():
            # 최종 목적지 클래스 폴더 생성 (예: dataset_dvc/train/apple/)
            dest_class_dir = FINAL_DATA_DIR / split_name / class_name
            os.makedirs(dest_class_dir, exist_ok=True)
            
            for idx, src_path in enumerate(paths):
                try:
                    # 이미지 로드 및 리사이즈
                    with Image.open(src_path) as img:
                        # RGB 변환 (PNG 투명도 채널 등의 호환성 이슈 방지)
                        img_rgb = img.convert('RGB')
                        img_resized = img_rgb.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
                        
                        # 저장 파일명 명명 규칙: "클래스명_인덱스.jpg" 
                        new_filename = f"{class_name}_{idx:04d}.jpg"
                        img_resized.save(dest_class_dir / new_filename, "JPEG")
                except Exception as e:
                    print(f"⚠️ 파일 에러 스킵 ({src_path}): {e}")

    # 리소스 절약을 위해 원본 raw_dataset 폴더 삭제 (선택 사항)
    shutil.rmtree(RAW_DATA_DIR, ignore_errors=True)
    print(f"🎉 성공! {FINAL_DATA_DIR} 위치에 224 사이즈 정제 및 분할이 완료되었습니다.")

if __name__ == "__main__":
    # 실행하기 전에 kaggle.json 설정이 완료되어 있어야 함
    download_kaggle_dataset()
    process_and_split_dataset()
