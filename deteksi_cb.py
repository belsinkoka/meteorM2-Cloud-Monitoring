import os
import time
import cv2
import numpy as np
import glob
import rasterio
from ultralytics import YOLO


# =====================================================
# KONFIGURASI
# =====================================================

SUPABASE_URL = "https://bkfudvtonbnnxlkbqiln.supabase.co"
SUPABASE_KEY = "sb_publishable_JEDoEXBdMaAIU36seFlaNQ_rfV6z_eE"

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER     = os.path.join(BASE_DIR, "data")
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "model", "best.pt")
RAW_BAND_PATH   = os.path.join(DATA_FOLDER, "raw_band.npy")

MAX_WIDTH = 2000

# =====================================================
# ★ THRESHOLD — ubah nilai ini sesuai kebutuhan
# =====================================================
# Hanya CB dengan confidence ≥ MIN_CONF_THRESHOLD yang
# akan ditampilkan di output gambar dan tabel.
#
#   0.20 → tampilkan WEAK, MODERATE, STRONG
#   0.20 → khusus MODERATE ke atas (default)
#   0.32 → hanya STRONG
#
MIN_CONF_THRESHOLD = 0.20   # ← ubah di sini

os.makedirs(DATA_FOLDER, exist_ok=True)


# =====================================================
# KALIBRASI SUHU
# =====================================================
#
# Formula divalidasi dari data TIF BMKG Himawari-9:
#   T(°C) = 60.0 - (DN_uint16 / 65280.0) * 160.0
#
# Anchor:
#   DN = 0      →  +60.0°C  (permukaan hangat, hitam di IR)
#   DN = 65280  → -100.0°C  (puncak CB ekstrem, putih terang)
#
# Validasi geografis:
#   Laut Jawa  (DN≈12385) → +29.6°C ✓ (SST tropis ~29°C)
#   Bali       (DN≈11955) → +30.7°C ✓ (SST pesisir)
#   CB kuat    (DN≈60000) →  ~-87°C ✓ (puncak CB tropis)
#
T_WARM = 60.0
T_COLD = -100.0
DN_MAX = 65280.0


def dn_to_celsius(dn_uint16: np.ndarray) -> np.ndarray:
    """Konversi nilai uint16 dari TIF BMKG → suhu °C."""
    return T_WARM + (dn_uint16.astype(np.float32) / DN_MAX) * (T_COLD - T_WARM)


# =====================================================
# LOAD MODEL
# =====================================================

print("Loading YOLOv8 model...")
model = YOLO(YOLO_MODEL_PATH)


# =====================================================
# INTENSITY CLASSIFICATION
# =====================================================

def classify_intensity(conf: float) -> str:
    if conf >= 0.32:
        return "STRONG"
    elif conf >= 0.2:
        return "MODERATE"
    else:
        return "WEAK"


# =====================================================
# GEO TIFF CONVERSION
# =====================================================

def convert_tif_to_png(tif_path: str) -> str:
    with rasterio.open(tif_path) as src:
        bounds   = src.bounds
        raw_band = src.read(1)   # uint16, shape (H, W)

        np.save(RAW_BAND_PATH, raw_band)

        img_bands = []
        for i in range(1, min(src.count, 3) + 1):
            band      = src.read(i)
            band_norm = cv2.normalize(band, None, 0, 255, cv2.NORM_MINMAX)
            img_bands.append(band_norm.astype(np.uint8))

        if len(img_bands) == 1:
            img_rgb = np.stack([img_bands[0]] * 3, axis=2)
        else:
            img_rgb = np.stack(img_bands, axis=2)

        img_bgr  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        png_path = os.path.join(DATA_FOLDER, "satellite_latest.png")
        cv2.imwrite(png_path, img_bgr)

        bounds_file = os.path.join(DATA_FOLDER, "map_bounds.txt")
        with open(bounds_file, "w") as f:
            f.write(f"{bounds.bottom},{bounds.left},{bounds.top},{bounds.right}")

    print(f"[TIF] Raw band shape: {raw_band.shape}, range: {raw_band.min()}–{raw_band.max()}")
    return png_path


# =====================================================
# EKSTRAKSI SUHU AKURAT PER BOUNDING BOX
# =====================================================

def extract_temperature_for_box(
    x: int, y: int, w_box: int, h_box: int,
    scale: float,
    img_h: int, img_w: int
) -> tuple[float, str]:
    if not os.path.exists(RAW_BAND_PATH):
        return -60.0, "no_raw_data"

    raw_band     = np.load(RAW_BAND_PATH)
    h_raw, w_raw = raw_band.shape

    img_h_orig = img_h / scale
    img_w_orig = img_w / scale

    sx = w_raw / img_w_orig
    sy = h_raw / img_h_orig

    x_orig = x     / scale
    y_orig = y     / scale
    w_orig = w_box / scale
    h_orig = h_box / scale

    x1_r = int(np.clip(x_orig * sx,             0, w_raw - 1))
    y1_r = int(np.clip(y_orig * sy,             0, h_raw - 1))
    x2_r = int(np.clip((x_orig + w_orig) * sx,  0, w_raw))
    y2_r = int(np.clip((y_orig + h_orig) * sy,  0, h_raw))

    roi_raw = raw_band[y1_r:y2_r, x1_r:x2_r]

    if roi_raw.size == 0:
        return -60.0, "empty_roi"

    valid = roi_raw[roi_raw > 0]
    if len(valid) == 0:
        valid = roi_raw.flatten()

    # Persentil ke-5 = area paling dingin = puncak CB
    dn_coldest = float(np.percentile(valid, 5))
    temp_c     = float(dn_to_celsius(np.array([dn_coldest]))[0])

    return round(temp_c, 1), "raw_ir"


# =====================================================
# AI DETECTION
# =====================================================

def detect_cb(image_path: str) -> int:
    image = cv2.imread(image_path)

    bounds_file = os.path.join(DATA_FOLDER, "map_bounds.txt")
    with open(bounds_file) as f:
        b, l, t, r = map(float, f.read().split(","))
    geo_bounds = (b, l, t, r)

    scale = 1.0
    if image.shape[1] > MAX_WIDTH:
        scale = MAX_WIDTH / image.shape[1]
        image = cv2.resize(image, None, fx=scale, fy=scale)

    img_h, img_w = image.shape[:2]

    table_path = os.path.join(DATA_FOLDER, "cb_table.txt")
    if os.path.exists(table_path):
        os.remove(table_path)

    overlay       = image.copy()
    results       = model(image, conf=0.02, iou=0.5)   # intentionally low, filter manual di bawah
    cluster_boxes = []

    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            print(f"RAW DETECTION — conf={conf:.3f}")
            cluster_boxes.append((x1, y1, x2 - x1, y2 - y1, conf))

    cb_count = 0
    cb_table = []
    cb_id    = 1

    for (x, y, w_box, h_box, conf) in cluster_boxes:

        # ★ FILTER THRESHOLD ★
        # Lewati deteksi yang tidak memenuhi minimum confidence
        if conf < MIN_CONF_THRESHOLD:
            print(f"  SKIP CB (conf={conf:.3f} < threshold={MIN_CONF_THRESHOLD})")
            continue

        lat1, lon1 = pixel_to_latlon(
            x / scale,             y / scale,
            img_w / scale, img_h / scale, geo_bounds
        )
        lat2, lon2 = pixel_to_latlon(
            (x + w_box) / scale,   (y + h_box) / scale,
            img_w / scale, img_h / scale, geo_bounds
        )

        confidence = conf * 100
        intensity  = classify_intensity(conf)

        temperature, temp_method = extract_temperature_for_box(
            x, y, w_box, h_box, scale, img_h, img_w
        )
        print(f"  CB{cb_id}: conf={confidence:.1f}%, T={temperature}°C ({temp_method}), {intensity}")

        cb_table.append({
            "cb"          : cb_id,
            "confidence"  : confidence,
            "temperature" : temperature,
            "bounds"      : [[lat1, lon1], [lat2, lon2]]
        })

        color = (0, 0, 255)   if intensity == "STRONG"   else \
                (0, 255, 255) if intensity == "MODERATE" else \
                (0, 255, 0)

        cv2.rectangle(overlay, (x, y), (x + w_box, y + h_box), color, 4)
        cv2.putText(
            overlay,
            f"CB{cb_id} {intensity}",
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.5, (255, 255, 255), 3
        )

        cb_count += 1
        cb_id    += 1

    output_path = os.path.join(DATA_FOLDER, "cb_latest.png")
    cv2.imwrite(output_path, overlay)

    count_path = os.path.join(DATA_FOLDER, "cb_count.txt")
    with open(count_path, "w") as f:
        f.write(str(cb_count))

    # ★ BUG FIX: kolom temperature sebelumnya hilang dari baris data ★
    table_path = os.path.join(DATA_FOLDER, "cb_table.txt")
    with open(table_path, "w") as f:
        f.write("CB,Confidence(%),Temperature(C),lat1,lon1,lat2,lon2\n")
        for cb in cb_table:
            lat1, lon1 = cb["bounds"][0]
            lat2, lon2 = cb["bounds"][1]
            f.write(
                f"{cb['cb']},"
                f"{cb['confidence']:.2f},"
                f"{cb['temperature']},"          # ← FIX: kolom ini sebelumnya tidak ada
                f"{lat1:.4f},{lon1:.4f},"
                f"{lat2:.4f},{lon2:.4f}\n"
            )

    print(f"CB clusters detected (above threshold): {cb_count}")
    return cb_count


# =====================================================
# HAPUS FILE OUTPUT LAMA
# =====================================================

def clean_old_files():
    for file in glob.glob(os.path.join(DATA_FOLDER, "cb_*.png")):
        if "history" not in file:
            os.remove(file)
    for file in glob.glob(os.path.join(DATA_FOLDER, "ir_*.png")):
        if "history" not in file:
            os.remove(file)


# =====================================================
# MONITOR FOLDER
# =====================================================

def run_detection():
    last_processed_path  = None
    last_processed_mtime = None
    print(f"Monitoring folder TIF... (threshold: conf ≥ {MIN_CONF_THRESHOLD})")

    while True:
        tif_files = [
            f for f in os.listdir(DATA_FOLDER)
            if f.lower().endswith((".tif", ".tiff"))
            and f != "reprojected.tif"
        ]

        if not tif_files:
            time.sleep(5)
            continue

        tif_files.sort(
            key=lambda x: os.path.getmtime(os.path.join(DATA_FOLDER, x)),
            reverse=True
        )

        latest_file  = tif_files[0]
        latest_path  = os.path.join(DATA_FOLDER, latest_file)
        latest_mtime = os.path.getmtime(latest_path)

        if latest_path != last_processed_path or latest_mtime != last_processed_mtime:
            print(f"\nNew TIF detected: {latest_file}")
            clean_old_files()
            png_path = convert_tif_to_png(latest_path)
            detect_cb(png_path)
            last_processed_path  = latest_path
            last_processed_mtime = latest_mtime
            print(f"Detection complete for: {latest_file}\n")

        time.sleep(5)


def pixel_to_latlon(x, y, width, height, bounds):
    lat_min, lon_min, lat_max, lon_max = bounds
    lon = lon_min + (x / width)  * (lon_max - lon_min)
    lat = lat_max - (y / height) * (lat_max - lat_min)
    return lat, lon


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    print("AI CB Detection Engine Started")
    run_detection()