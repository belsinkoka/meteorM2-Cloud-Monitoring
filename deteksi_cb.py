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

# >>> TANYA: koneksi-supabase / keamanan-kredensial / environment-variable
SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://bkfudvtonbnnxlkbqiln.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "sb_publishable_JEDoEXBdMaAIU36seFlaNQ_rfV6z_eE")

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER     = os.path.join(BASE_DIR, "data")
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "model", "best.pt")
RAW_BAND_PATH   = os.path.join(DATA_FOLDER, "raw_band.npy")

MAX_WIDTH = 2000

MIN_CONF_THRESHOLD = 0.20   # conf min yang diizinkan

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
# CATATAN PENGAMBILAN SUHU:
#   Citra LRPT Meteor M2 mengandung garis noise & tepi putih dengan
#   DN ekstrem (~65280). Piksel ber-DN di atas DN_NOISE_CUTOFF dibuang
#   dulu, lalu suhu diambil dari MEDIAN (persentil-50) piksel awan yang
#   tersisa. Median dipilih agar suhu merepresentasikan rata-rata area
#   awan dalam bounding box, bukan hanya puncak terdingin ekstrem.
#
T_WARM = 60.0
T_COLD = -100.0
DN_MAX = 65280.0

#TANYA: rumus-suhu
def dn_to_celsius(dn_uint16: np.ndarray) -> np.ndarray:
    """Konversi nilai uint16 dari TIF BMKG → suhu °C."""
    return T_WARM + (dn_uint16.astype(np.float32) / DN_MAX) * (T_COLD - T_WARM)


# =====================================================
# LOAD MODEL
# =====================================================

print("Loading YOLOv8 model...")
model = YOLO(YOLO_MODEL_PATH)

# ★ BARU: konfirmasi daftar kelas model (mis. {0: 'CB', 1: 'nonCB'})
print(f"Kelas model: {model.names}")

# ★ BARU: auto-deteksi index kelas 'CB' dari model.
#   Aman dari urutan data.yaml — tidak peduli CB di index 0 atau 1.
CB_CLASS_ID = None
for idx, name in model.names.items():
    if str(name).strip().upper() == "CB":
        CB_CLASS_ID = idx
        break

if CB_CLASS_ID is None:
    CB_CLASS_ID = 0   # fallback kalau nama kelas bukan persis "CB"
    print(f"PERINGATAN: kelas 'CB' tidak ditemukan di model.names, "
          f"memakai default index {CB_CLASS_ID}. "
          f"Cek kembali nama kelas di data.yaml!")

print(f"CB class ID terpilih = {CB_CLASS_ID}")


# =====================================================
# INTENSITY CLASSIFICATION
# =====================================================
#TANYA: klasifikasi-warna
def classify_intensity(conf: float) -> str:
    # STRONG  : conf >= 40%  (CB sangat kuat)
    # MODERATE: 30% - 39%    (CB sedang)
    # WEAK    : 20% - 29%    (CB lemah / berkembang)
    if conf >= 0.40:
        return "STRONG"
    elif conf >= 0.30:
        return "MODERATE"
    else:
        return "WEAK"


# =====================================================
# GEO TIFF CONVERSION
# =====================================================
#TANYA: konversi-png
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

        # File 1: untuk YOLO (solid, 3 channel)
        png_path = os.path.join(DATA_FOLDER, "satellite_latest.png")
        cv2.imwrite(png_path, img_bgr)

        # File 2: untuk overlay peta Leaflet (hitam → transparan)
        BLACK_THRESHOLD = 10
        bgra = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)
        black_mask = np.all(img_bgr < BLACK_THRESHOLD, axis=2)
        bgra[black_mask, 3] = 0
        overlay_path = os.path.join(DATA_FOLDER, "satellite_overlay.png")
        cv2.imwrite(overlay_path, bgra)

        bounds_file = os.path.join(DATA_FOLDER, "map_bounds.txt")
        with open(bounds_file, "w") as f:
            f.write(f"{bounds.bottom},{bounds.left},{bounds.top},{bounds.right}")

    print(f"[TIF] Raw band shape: {raw_band.shape}, range: {raw_band.min()}–{raw_band.max()}")
    return png_path


# =====================================================
# EKSTRAKSI SUHU AKURAT PER BOUNDING BOX
# =====================================================
#TANYA: median-suhu
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

    # ── Buang piksel noise ber-DN ekstrem ──────────────────────
    # Garis noise/tepi putih hasil dekoding LRPT punya DN sangat
    # tinggi (mendekati 65280) yang keliru terbaca sebagai -100°C.
    # Kita buang piksel di atas DN_NOISE_CUTOFF sebelum hitung suhu.
    DN_NOISE_CUTOFF = 62000           # ≈ -92°C ke atas dianggap noise
    cloud = valid[valid < DN_NOISE_CUTOFF]
    if len(cloud) == 0:               # kalau semua kepotong, pakai data asli
        cloud = valid

    # Dari piksel awan yang tersisa, ambil persentil ke-50 (median)
    # sebagai estimasi suhu rata-rata area awan. Median lebih
    # representatif terhadap keseluruhan awan dalam bounding box
    # dibanding persentil tinggi yang hanya menangkap puncak ekstrem.
    dn_coldest = float(np.percentile(cloud, 50))
    temp_c     = float(dn_to_celsius(np.array([dn_coldest]))[0])

    return round(temp_c, 1), "raw_ir"


# =====================================================
# AI DETECTION
# =====================================================

# =====================================================
# MERGE BOUNDING BOX OVERLAP
# =====================================================
#
# Box CB yang saling tumpang tindih (IoU di atas ambang) digabung
# menjadi satu box besar yang membungkus semuanya. Confidence yang
# dipakai adalah yang TERTINGGI di antara anggota cluster, sehingga
# level STRONG selalu diutamakan dibanding MODERATE/WEAK saat overlap.
#
#TANYA: merge-overlap-box / nms-iou
MERGE_IOU_THRESHOLD = 0.20

def _iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter_area

    return inter_area / union if union > 0 else 0.0


def merge_overlapping_boxes(boxes_with_conf, iou_thresh=MERGE_IOU_THRESHOLD):
    """
    boxes_with_conf: list of (x, y, w_box, h_box, conf)
    return: list of (x, y, w_box, h_box, conf) hasil merge,
            conf yang dipakai adalah conf TERTINGGI di tiap cluster.
    """
    # urutkan dari confidence tertinggi agar cluster terbentuk dari box terkuat
    items = sorted(boxes_with_conf, key=lambda d: d[4], reverse=True)
    used  = [False] * len(items)
    merged = []

    for i in range(len(items)):
        if used[i]:
            continue

        cluster = [items[i]]
        used[i] = True

        changed = True
        while changed:
            changed = False
            xs1 = [c[0] for c in cluster]
            ys1 = [c[1] for c in cluster]
            xs2 = [c[0] + c[2] for c in cluster]
            ys2 = [c[1] + c[3] for c in cluster]
            cluster_box = (min(xs1), min(ys1), max(xs2), max(ys2))

            for j in range(len(items)):
                if used[j]:
                    continue
                xj, yj, wj, hj, _ = items[j]
                box_j = (xj, yj, xj + wj, yj + hj)
                if _iou(cluster_box, box_j) >= iou_thresh:
                    cluster.append(items[j])
                    used[j] = True
                    changed = True

        xs1 = [c[0] for c in cluster]
        ys1 = [c[1] for c in cluster]
        xs2 = [c[0] + c[2] for c in cluster]
        ys2 = [c[1] + c[3] for c in cluster]

        x_merged = min(xs1)
        y_merged = min(ys1)
        w_merged = max(xs2) - x_merged
        h_merged = max(ys2) - y_merged

        # confidence tertinggi di cluster → STRONG selalu menang saat overlap
        conf_merged = max(c[4] for c in cluster)

        merged.append((x_merged, y_merged, w_merged, h_merged, conf_merged))

    return merged

#TANYA: panggil-model
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
            #TANYA: filter-kelas-cb (abaikan non-cb)
            cls = int(box.cls[0])
            if cls != CB_CLASS_ID:
                print(f"  SKIP (kelas={cls} '{model.names.get(cls, '?')}', bukan CB)")
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            print(f"RAW DETECTION — CB conf={conf:.3f}")
            cluster_boxes.append((x1, y1, x2 - x1, y2 - y1, conf))

    # >>> TANYA: merge-overlap-box
    # Gabungkan box yang saling tumpang tindih SEBELUM filter threshold,
    # supaya box STRONG yang overlap dengan WEAK/MODERATE tetap terpilih
    # sebagai representasi cluster tersebut.
    cluster_boxes = merge_overlapping_boxes(cluster_boxes)

    cb_count = 0
    cb_table = []
    cb_id    = 1

    for (x, y, w_box, h_box, conf) in cluster_boxes:

        # ★ FILTER THRESHOLD ★
        #TANYA: filter-threshold
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

#TANYA: konversi-koordinat
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