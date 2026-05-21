"""
======================================================
  EVALUASI MODEL YOLOv8s — CB DETECTION
  Untuk keperluan laporan Tugas Akhir BAB 4
======================================================
Jalankan di Google Colab setelah training selesai.
Hasil disimpan di folder /content/evaluasi_hasil/
"""

# ── INSTALASI ──────────────────────────────────────
# !pip install ultralytics roboflow -q

import os
import time
import json
import cv2
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from google.colab import files
from google.colab import drive

# ══════════════════════════════════════════════════
# KONFIGURASI — sesuaikan bagian ini
# ══════════════════════════════════════════════════

MODEL_PATH   = "/content/runs/detect/cb_detector_640/weights/best.pt"
DATASET_YAML = "/content/datasets/CB-1/data.yaml"   # sesuaikan path dataset Anda
OUTPUT_DIR   = "/content/evaluasi_hasil"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════
# 1. LOAD MODEL
# ══════════════════════════════════════════════════

print("=" * 55)
print("  EVALUASI MODEL YOLOv8s — DETEKSI AWAN CB")
print("=" * 55)

model = YOLO(MODEL_PATH)
device = 0 if torch.cuda.is_available() else 'cpu'
print(f"\nDevice  : {'GPU — ' + torch.cuda.get_device_name(0) if device == 0 else 'CPU'}")
print(f"Model   : {MODEL_PATH}")
print(f"Output  : {OUTPUT_DIR}")

# ══════════════════════════════════════════════════
# 2. VALIDASI PADA TEST SET
#    Menghasilkan: Precision, Recall, F1, mAP@0.5,
#    mAP@0.5-95, Confusion Matrix, kurva PR/F1/P/R
# ══════════════════════════════════════════════════

print("\n" + "─" * 55)
print("  TAHAP 1 — Validasi pada Test Set")
print("─" * 55)

val_results = model.val(
    data=DATASET_YAML,
    split="test",              # evaluasi pada test set
    imgsz=640,
    conf=0.20,                 # sesuai MIN_CONF_THRESHOLD sistem
    iou=0.5,
    device=device,
    save_json=True,
    plots=True,                # simpan semua kurva otomatis
    project=OUTPUT_DIR,
    name="validasi_test_set",
    verbose=True,
)

# Ekstraksi nilai metrik
precision = float(val_results.box.mp)      # mean Precision
recall    = float(val_results.box.mr)      # mean Recall
map50     = float(val_results.box.map50)   # mAP@0.5
map5095   = float(val_results.box.map)     # mAP@0.5-0.95
f1        = 2 * (precision * recall) / (precision + recall + 1e-9)

print(f"\n{'─'*35}")
print(f"  Precision   : {precision:.4f}  ({precision*100:.2f}%)")
print(f"  Recall      : {recall:.4f}  ({recall*100:.2f}%)")
print(f"  F1-Score    : {f1:.4f}  ({f1*100:.2f}%)")
print(f"  mAP@0.5     : {map50:.4f}  ({map50*100:.2f}%)")
print(f"  mAP@0.5-0.95: {map5095:.4f}  ({map5095*100:.2f}%)")
print(f"{'─'*35}")

# ══════════════════════════════════════════════════
# 3. PENGUKURAN WAKTU INFERENSI
#    Diulang 20× pada 1 gambar untuk rata-rata stabil
# ══════════════════════════════════════════════════

print("\n" + "─" * 55)
print("  TAHAP 2 — Pengukuran Waktu Inferensi")
print("─" * 55)

# Ambil satu gambar dari test set sebagai sampel
import glob, random
test_images = glob.glob("/content/datasets/**/test/images/*.jpg", recursive=True)
test_images += glob.glob("/content/datasets/**/test/images/*.png", recursive=True)

waktu_list = []

if test_images:
    sample_img = random.choice(test_images)
    print(f"\nGambar uji : {os.path.basename(sample_img)}")
    print(f"Jumlah run : 20 kali (untuk rata-rata stabil)")

    # Warmup 3x (tidak dihitung)
    for _ in range(3):
        model(sample_img, conf=0.20, iou=0.5, device=device, verbose=False)

    # Pengukuran 20x
    for i in range(20):
        t_start = time.perf_counter()
        model(sample_img, conf=0.20, iou=0.5, device=device, verbose=False)
        t_end   = time.perf_counter()
        waktu_ms = (t_end - t_start) * 1000
        waktu_list.append(waktu_ms)
        print(f"  Run {i+1:02d}: {waktu_ms:.2f} ms")

    waktu_rata   = np.mean(waktu_list)
    waktu_min    = np.min(waktu_list)
    waktu_max    = np.max(waktu_list)
    waktu_std    = np.std(waktu_list)

    print(f"\n{'─'*35}")
    print(f"  Rata-rata : {waktu_rata:.2f} ms")
    print(f"  Minimum   : {waktu_min:.2f} ms")
    print(f"  Maksimum  : {waktu_max:.2f} ms")
    print(f"  Std dev   : {waktu_std:.2f} ms")
    print(f"{'─'*35}")
else:
    print("  Tidak ada gambar test set ditemukan.")
    waktu_rata = waktu_min = waktu_max = waktu_std = 0

# ══════════════════════════════════════════════════
# 4. DETEKSI VISUAL PADA 10 SAMPEL GAMBAR TEST
#    Simpan citra dengan bounding box untuk laporan
# ══════════════════════════════════════════════════

print("\n" + "─" * 55)
print("  TAHAP 3 — Deteksi Visual pada Sampel Test Set")
print("─" * 55)

WARNA = {
    "STRONG"   : (0, 0, 255),    # merah
    "MODERATE" : (0, 255, 255),  # kuning
    "WEAK"     : (0, 255, 0),    # hijau
}

def kategori(conf):
    if conf >= 0.32: return "STRONG"
    if conf >= 0.20: return "MODERATE"
    return "WEAK"

output_visual_dir = os.path.join(OUTPUT_DIR, "deteksi_visual")
os.makedirs(output_visual_dir, exist_ok=True)

if test_images:
    # Ambil 10 gambar acak (atau semua kalau < 10)
    samples = random.sample(test_images, min(10, len(test_images)))

    for idx, img_path in enumerate(samples, start=1):
        img     = cv2.imread(img_path)
        results = model(img_path, conf=0.20, iou=0.5, device=device, verbose=False)
        overlay = img.copy()
        cb_count = 0

        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf  = float(box.conf[0])
                kat   = kategori(conf)
                warna = WARNA[kat]
                cb_count += 1

                cv2.rectangle(overlay, (x1, y1), (x2, y2), warna, 3)
                label = f"CB{cb_count} {kat} {conf*100:.1f}%"
                cv2.putText(overlay, label, (x1, max(y1-8, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        out_name = f"sampel_{idx:02d}_cb{cb_count}_{os.path.basename(img_path)}"
        out_path = os.path.join(output_visual_dir, out_name)
        cv2.imwrite(out_path, overlay)
        print(f"  [{idx:02d}] {os.path.basename(img_path)} → {cb_count} CB terdeteksi → disimpan")

    print(f"\n  Hasil visual disimpan di: {output_visual_dir}")

# ══════════════════════════════════════════════════
# 5. TABEL RINGKASAN — Untuk copas ke laporan
# ══════════════════════════════════════════════════

print("\n" + "═" * 55)
print("  RINGKASAN HASIL EVALUASI MODEL")
print("═" * 55)
print(f"  Model             : YOLOv8s")
print(f"  imgsz             : 640 × 640 piksel")
print(f"  Confidence filter : 0.20 (MIN_CONF_THRESHOLD)")
print(f"  IoU threshold     : 0.50")
print(f"  Device            : {'GPU — ' + torch.cuda.get_device_name(0) if device == 0 else 'CPU'}")
print(f"{'─'*55}")
print(f"  Precision         : {precision:.4f}  ({precision*100:.2f}%)")
print(f"  Recall            : {recall:.4f}  ({recall*100:.2f}%)")
print(f"  F1-Score          : {f1:.4f}  ({f1*100:.2f}%)")
print(f"  mAP@0.5           : {map50:.4f}  ({map50*100:.2f}%)")
print(f"  mAP@0.5-0.95      : {map5095:.4f}  ({map5095*100:.2f}%)")
print(f"{'─'*55}")
if waktu_rata > 0:
    print(f"  Waktu inferensi   : {waktu_rata:.2f} ms / citra (rata-rata)")
    print(f"  Waktu min/maks    : {waktu_min:.2f} ms / {waktu_max:.2f} ms")
print(f"{'─'*55}")

# ══════════════════════════════════════════════════
# 6. SIMPAN RINGKASAN KE FILE JSON
# ══════════════════════════════════════════════════

ringkasan = {
    "model"              : "YOLOv8s",
    "imgsz"              : 640,
    "conf_threshold"     : 0.20,
    "iou_threshold"      : 0.50,
    "device"             : str(device),
    "precision"          : round(precision, 4),
    "recall"             : round(recall, 4),
    "f1_score"           : round(f1, 4),
    "mAP50"              : round(map50, 4),
    "mAP50_95"           : round(map5095, 4),
    "waktu_rata_ms"      : round(waktu_rata, 2),
    "waktu_min_ms"       : round(waktu_min, 2),
    "waktu_maks_ms"      : round(waktu_max, 2),
}

json_path = os.path.join(OUTPUT_DIR, "ringkasan_evaluasi.json")
with open(json_path, "w") as f:
    json.dump(ringkasan, f, indent=4)

print(f"\n  Ringkasan disimpan: {json_path}")

# ══════════════════════════════════════════════════
# 7. DOWNLOAD SEMUA HASIL KE KOMPUTER
# ══════════════════════════════════════════════════

print("\n" + "─" * 55)
print("  TAHAP 4 — Simpan ke Google Drive + Download")
print("─" * 55)

# Salin semua hasil ke Google Drive
try:
    drive.mount('/content/drive')
    import shutil

    drive_output = "/content/drive/MyDrive/evaluasi_cb_detector"
    if os.path.exists(drive_output):
        shutil.rmtree(drive_output)
    shutil.copytree(OUTPUT_DIR, drive_output)
    print(f"  ✓ Semua hasil tersimpan di Google Drive: evaluasi_cb_detector/")
except Exception as e:
    print(f"  Drive tidak tersambung: {e}")

# Download ringkasan JSON
print("\n  Mendownload file ringkasan...")
files.download(json_path)

print("\n" + "═" * 55)
print("  EVALUASI SELESAI")
print(f"  Semua hasil ada di folder: {OUTPUT_DIR}")
print(f"  Sub-folder penting:")
print(f"  • validasi_test_set/ → kurva PR, F1, confusion matrix")
print(f"  • deteksi_visual/    → 10 sampel gambar dengan bounding box")
print(f"  • ringkasan_evaluasi.json → data untuk tabel laporan")
print("═" * 55)