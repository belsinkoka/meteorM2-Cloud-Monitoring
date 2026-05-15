from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory
import os
from datetime import datetime
import cv2
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from supabase import create_client
import threading
import numpy as np
import shutil
import base64
import requests
import time


# =====================================================
# 1. KONFIGURASI & KONSTANTA
# =====================================================

SUPABASE_URL = "https://bkfudvtonbnnxlkbqiln.supabase.co"
SUPABASE_KEY = "sb_publishable_JEDoEXBdMaAIU36seFlaNQ_rfV6z_eE"

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER   = os.path.join(BASE_DIR, "data")
UPLOAD_FOLDER = DATA_FOLDER
HISTORY_FOLDER = os.path.join(DATA_FOLDER, "history")
BMKG_FOLDER   = "bmkg"

ALLOWED_EXTENSIONS = {"tif", "tiff"}

cities = [
    {"name": "Jakarta",  "lat": -6.2,  "lon": 106.8},
    {"name": "Surabaya", "lat": -7.25, "lon": 112.75},
    {"name": "Bandung",  "lat": -6.9,  "lon": 107.6},
    {"name": "Medan",    "lat":  3.6,  "lon":  98.6},
    {"name": "Makassar", "lat": -5.1,  "lon": 119.4},
]


# =====================================================
# 2. INISIALISASI APP & KLIEN EKSTERNAL
# =====================================================

app = Flask(__name__)
app.secret_key = "meteor_m2_secret_key"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

os.makedirs(HISTORY_FOLDER, exist_ok=True)
os.makedirs(BMKG_FOLDER,    exist_ok=True)


# =====================================================
# 3. HELPER / UTILITAS
# =====================================================

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_latest_ir_image():
    files = [
        f for f in os.listdir(DATA_FOLDER)
        if f.endswith(".png") and "satellite" in f
    ]
    if not files:
        return None
    files.sort(
        key=lambda x: os.path.getmtime(os.path.join(DATA_FOLDER, x)),
        reverse=True,
    )
    return files[0]


def read_cb_table():
    """Baca cb_table.txt dan kembalikan sebagai list of dict."""
    cb_table = []
    table_path = os.path.join(DATA_FOLDER, "cb_table.txt")
    if os.path.exists(table_path):
        with open(table_path) as f:
            next(f)  # skip header
            for line in f:
                parts = line.strip().split(",")
                if len(parts) != 7:
                    continue
                cb, conf, temp, x1, y1, x2, y2 = parts
                cb_table.append({
                    "cb":         int(cb),
                    "confidence": float(conf),
                    "temperature": float(temp),
                    "bounds": [
                        [float(x1), float(y1)],
                        [float(x2), float(y2)],
                    ],
                })
    return cb_table


def read_map_bounds(default=None):
    """Baca map_bounds.txt dan kembalikan sebagai dict."""
    if default is None:
        default = {"south": -12, "west": 94, "north": 12, "east": 142}
    bounds_path = os.path.join(DATA_FOLDER, "map_bounds.txt")
    if os.path.exists(bounds_path):
        try:
            with open(bounds_path) as f:
                b = f.read().split(",")
            if len(b) == 4:
                return {
                    "south": float(b[0]),
                    "west":  float(b[1]),
                    "north": float(b[2]),
                    "east":  float(b[3]),
                }
        except Exception:
            pass
    return default


def read_cb_count():
    """Baca cb_count.txt dan kembalikan sebagai int."""
    count_path = os.path.join(DATA_FOLDER, "cb_count.txt")
    if os.path.exists(count_path):
        try:
            with open(count_path) as f:
                return int(f.read().strip())
        except Exception:
            pass
    return 0


# =====================================================
# 4. LOGIKA BMKG
# =====================================================

def download_bmkg_image():
    timestamp = int(time.time())  # anti-cache
    url = (
        f"https://inderaja.bmkg.go.id/IMAGE/HIMA/H08_EH_Indonesia.png?t={timestamp}"
    )
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            file_path = os.path.join(BMKG_FOLDER, "bmkg_latest.png")
            with open(file_path, "wb") as f:
                f.write(response.content)
            print("BMKG image updated")
            return True
    except Exception as e:
        print("Error download BMKG:", e)
    return False


def auto_update_bmkg():
    while True:
        download_bmkg_image()
        time.sleep(60 * 5)  # update tiap 5 menit


# =====================================================
# 5. LOGIKA DETEKSI
# =====================================================

def detect_once():
    try:
        status_path = os.path.join(DATA_FOLDER, "status.txt")

        with open(status_path, "w") as f:
            f.write("processing")

        tif_path = os.path.join(DATA_FOLDER, "input.tif")

        import deteksi_cb
        png_path = deteksi_cb.convert_tif_to_png(tif_path)
        deteksi_cb.detect_cb(png_path)

        with open(status_path, "w") as f:
            f.write("done")

    except Exception as e:
        print("Error:", e)


# =====================================================
# 6. STARTUP  (jalankan sebelum routes didaftarkan)
# =====================================================

download_bmkg_image()
threading.Thread(target=auto_update_bmkg, daemon=True).start()


# =====================================================
# 7. AUTH ROUTES
# =====================================================

@app.route("/")
def root():
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        res = supabase.table("users").select("*").eq("username", username).execute()

        if res.data:
            user = res.data[0]
            if check_password_hash(user["password"], password):
                session["user"] = username
                return redirect(url_for("home"))

        return render_template("login.html", error="Username atau password salah")

    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if len(username) < 3:
            return render_template("register.html", error="Username minimal 3 karakter")

        if len(password) < 8:
            return render_template("register.html", error="Password minimal 8 karakter")

        hashed_pw = generate_password_hash(password)

        res = supabase.table("users").select("*").eq("username", username).execute()
        if res.data:
            return render_template("register.html", error="Username sudah digunakan")

        supabase.table("users").insert({
            "username": username,
            "password": hashed_pw,
        }).execute()

        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))


# =====================================================
# 8. PAGE ROUTES
# =====================================================

@app.route("/home")
def home():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("home.html")


@app.route("/cb-guide")
def cb_guide():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("cb_guide.html")


@app.route("/about")
def about():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("about.html")


@app.route("/about-system")
def about_system():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("about.html")


# =====================================================
# 9. STATIC FILE ROUTES
# =====================================================

@app.route("/tif/<filename>")
def tif_files(filename):
    return send_from_directory("data", filename)


@app.route("/data/<path:filename>")
def data_files(filename):
    return send_from_directory("data", filename)


@app.route("/bmkg/<filename>")
def bmkg_files(filename):
    return send_from_directory("bmkg", filename)


# =====================================================
# 10. DASHBOARD
# =====================================================

@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))

    count    = read_cb_count()
    cb_table = read_cb_table()
    bounds   = read_map_bounds()

    status = "idle"
    status_path = os.path.join(DATA_FOLDER, "status.txt")
    if os.path.exists(status_path):
        with open(status_path) as f:
            status = f.read().strip()

    # Cari file IR terbaru
    ir_file = None
    image_extensions = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    files = [
        f for f in os.listdir(DATA_FOLDER)
        if f.lower().endswith(image_extensions) and not f.startswith("cb_")
    ]
    if files:
        files.sort(
            key=lambda x: os.path.getmtime(os.path.join(DATA_FOLDER, x)),
            reverse=True,
        )
        ir_file = files[0]

    # Hasil deteksi CB
    cb_file = (
        "cb_latest.png"
        if os.path.exists(os.path.join(DATA_FOLDER, "cb_latest.png"))
        else None
    )

    # Last update time
    if ir_file:
        last_modified  = os.path.getmtime(os.path.join(DATA_FOLDER, ir_file))
        formatted_time = datetime.fromtimestamp(last_modified).strftime("%d-%m-%Y %H:%M:%S")
    else:
        formatted_time = "-"

    return render_template(
        "index.html",
        jumlah_cb=count,
        last_update=formatted_time,
        ir_file=ir_file.replace("\\", "/") if ir_file else None,
        cb_file=cb_file.replace("\\", "/") if cb_file else None,
        cb_table=cb_table,
        timestamp=int(time.time()),
        bounds=bounds,
        status=status,
    )


@app.route("/upload_tif", methods=["POST"])
def upload_tif():
    if "tif_file" not in request.files:
        return redirect(url_for("dashboard"))

    file = request.files["tif_file"]
    if file.filename == "":
        return redirect(url_for("dashboard"))

    if file and allowed_file(file.filename):
        # Hapus file TIF lama
        for f in os.listdir(UPLOAD_FOLDER):
            if f.endswith(".tif") or f.endswith(".tiff"):
                os.remove(os.path.join(UPLOAD_FOLDER, f))

        # Simpan dan rename file baru
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        os.rename(filepath, os.path.join(UPLOAD_FOLDER, "input.tif"))

        # Jalankan deteksi di thread terpisah
        threading.Thread(target=detect_once).start()

    return redirect(url_for("dashboard"))


# =====================================================
# 11. API ROUTES
# =====================================================

@app.route("/api/latest_cb")
def api_latest_cb():
    cb_table = read_cb_table()
    return {"cb_table": cb_table, "count": len(cb_table)}


# =====================================================
# 12. HISTORY
# =====================================================

@app.route("/history")
def history():
    if "user" not in session:
        return redirect(url_for("login"))

    limit = int(request.args.get("limit", 50))

    res = (
        supabase.table("detections")
        .select("*")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )

    return render_template("history.html", data=res.data)


@app.route("/load/<id>")
def load_history(id):
    res = supabase.table("detections").select("*").eq("id", id).execute()
    if not res.data:
        return "Data tidak ditemukan"

    detection = res.data[0]

    cb_res = supabase.table("cb_data").select("*").eq("detection_id", id).execute()

    cb_table = []
    for row in cb_res.data:
        cb_table.append({
            "cb":          row["cb"],
            "confidence":  row["confidence"],
            "temperature": row["temperature"],
            "bounds": [
                [row["lat1"], row["lon1"]],
                [row["lat2"], row["lon2"]],
            ],
        })

    return render_template(
        "index.html",
        jumlah_cb=detection["cb_count"],
        cb_table=cb_table,
        is_history=True,
        bounds=detection.get("bounds"),
        ir_file=detection.get("ir_file"),
        cb_file=detection.get("image_url"),
        timestamp=int(time.time()),
        last_update="-",
    )


@app.route("/save", methods=["POST"])
def save_data():
    if "user" not in session:
        return redirect(url_for("login"))

    ir_file  = get_latest_ir_image()
    cb_file  = "cb_latest.png" if os.path.exists(os.path.join(DATA_FOLDER, "cb_latest.png")) else None
    count    = read_cb_count()
    cb_table = read_cb_table()
    bounds   = read_map_bounds(default=None)

    timestamp = int(time.time())

    new_cb = None
    new_ir = None

    if cb_file:
        new_cb = f"history/cb_{timestamp}.png"
        shutil.copy(
            os.path.join(DATA_FOLDER, cb_file),
            os.path.join(HISTORY_FOLDER, f"cb_{timestamp}.png"),
        )

    if ir_file:
        new_ir = f"history/ir_{timestamp}.png"
        shutil.copy(
            os.path.join(DATA_FOLDER, ir_file),
            os.path.join(HISTORY_FOLDER, f"ir_{timestamp}.png"),
        )

    result = supabase.table("detections").insert({
        "image_url": new_cb,
        "ir_file":   new_ir,
        "cb_count":  count,
        "bounds":    bounds,
    }).execute()

    if not result.data:
        return "Gagal menyimpan"

    detection_id = result.data[0]["id"]

    for cb in cb_table:
        supabase.table("cb_data").insert({
            "detection_id": detection_id,
            "cb":           cb["cb"],
            "confidence":   cb["confidence"],
            "temperature":  cb["temperature"],
            "lat1": cb["bounds"][0][0],
            "lon1": cb["bounds"][0][1],
            "lat2": cb["bounds"][1][0],
            "lon2": cb["bounds"][1][1],
        }).execute()

    return redirect(url_for("dashboard"))


@app.route("/delete_selected", methods=["POST"])
def delete_selected():
    ids = request.form.getlist("selected_ids")
    for id in ids:
        supabase.table("cb_data").delete().eq("detection_id", id).execute()
        supabase.table("detections").delete().eq("id", id).execute()
    return redirect(url_for("history"))


# =====================================================
# 13. MAIN
# =====================================================

if __name__ == "__main__":
    app.run(debug=True)