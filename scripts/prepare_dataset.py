import os
import rasterio
import numpy as np
import cv2

INPUT_FOLDER = "dataset_raw"
OUTPUT_IMG = "dataset/images"
OUTPUT_LABEL = "dataset/labels"

os.makedirs(OUTPUT_IMG, exist_ok=True)
os.makedirs(OUTPUT_LABEL, exist_ok=True)

tile_size = 256
stride = 128

def normalize(img):
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img = (img * 255).astype(np.uint8)
    return img

count = 0

for file in os.listdir(INPUT_FOLDER):
    if not file.endswith(".tif"):
        continue

    path = os.path.join(INPUT_FOLDER, file)

    with rasterio.open(path) as src:
        img = src.read(1)

    img = normalize(img)

    h, w = img.shape

    for y in range(0, h - tile_size, stride):
        for x in range(0, w - tile_size, stride):

            tile = img[y:y+tile_size, x:x+tile_size]

            # 🔥 FILTER 1: terlalu gelap
            if tile.mean() < 10:
                continue

            # 🔥 FILTER 2: terlalu flat (tidak ada informasi)
            if tile.std() < 5:
                continue

            # 🔥 FILTER 3: terlalu banyak pixel hitam (noise garis)
            black_ratio = np.sum(tile < 5) / tile.size
            if black_ratio > 0.3:
                continue

            filename = f"tile_{count}.png"

            cv2.imwrite(os.path.join(OUTPUT_IMG, filename), tile)

            # label kosong (nanti diisi manual)
            open(os.path.join(OUTPUT_LABEL, filename.replace(".png", ".txt")), "w").close()

            count += 1

print("DONE:", count)