import os
import time

# 네트워크 경로
base_path = r"//192.168.10.69/개발팀/lay/dataset-yoloe-construction/dataset/Construction_8845_['Gloves', 'Helmet', 'Human', 'Safety Boot', 'Safety Vest', 'boots', 'glasses', 'gloves', 'hat', 'helmet', 'no boot', 'no boots', 'no gloves', 'no hat', 'no vest', 'vest']/train/images"

print(f"Path: {base_path}")
print("=" * 60)

# 파일 목록 가져오기 시간 측정
start = time.time()
try:
    files = os.listdir(base_path)
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = [f for f in files if os.path.splitext(f)[1].lower() in image_exts]
    elapsed = time.time() - start

    print(f"Total files: {len(files)}")
    print(f"Image files: {len(image_files)}")
    print(f"List time: {elapsed:.2f} sec")
except Exception as e:
    print(f"Error: {e}")
