# 육인수 - Kaggle 과일·채소 데이터 로드 스크립트
import os
import zipfile

# 데이터 저장 폴더 생성
os.makedirs("data/raw", exist_ok=True)

# Kaggle API로 데이터 다운로드
os.system("kaggle datasets download -d kritikseth/fruit-and-vegetable-image-recognition -p data/raw")

# 압축 해제
zip_path = "data/raw/fruit-and-vegetable-image-recognition.zip"
if os.path.exists(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("data/raw")
    print("압축 해제 완료!")
else:
    print("zip 파일을 찾을 수 없어요.")

print("데이터 저장 위치: data/raw")