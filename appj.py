import os
import json
import uuid
import base64
import traceback
from datetime import datetime, date
from typing import List, Dict, Tuple
from collections import deque
from io import BytesIO

from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import requests

# 画像圧縮用（インストールされていれば）
try:
    from PIL import Image
    PIL_OK = True
except Exception:
    PIL_OK = False

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

load_dotenv()

# =====================
# パス設定
# =====================
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
TEMPLATES  = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")
UPLOAD_DIR = os.path.join(STATIC_DIR, "uploads")
IMAGES_DIR = os.path.join(STATIC_DIR, "images")   # ★ seed→images
DATA_DIR   = os.path.join(BASE_DIR, "data")
DB_PATH    = os.path.join(DATA_DIR, "items.json")

ALLOWED_EXT = {"png", "jpg", "jpeg", "gif", "webp"}

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(DATA_DIR,   exist_ok=True)

# =====================
# LLM設定
# =====================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL   = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

app = Flask(__name__, template_folder=TEMPLATES, static_folder=STATIC_DIR)

# =====================
# HTTPセッション（リトライ付）
# =====================
def _make_session():
    s = requests.Session()
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.8,
        status_forcelist=[429, 500, 502, 503, 504],
        raise_on_status=False
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s

SESSION = _make_session()

# =====================
# DB
# =====================
def load_db() -> List[Dict]:
    if not os.path.exists(DB_PATH):
        return []
    try:
        with open(DB_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            for it in data:
                it.setdefault("quantity", 1)
                it.setdefault("unit", "個")
                it.setdefault("category", "")
                it.setdefault("location", "冷蔵")
                it.setdefault("is_archived", False)
            return data
    except Exception:
        return []

def save_db(items: List[Dict]):
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

# =====================
# ユーティリティ
# =====================
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def cookpad_search_url(keywords):
    from urllib.parse import quote_plus
    q = quote_plus(" ".join([k for k in keywords if k]))
    return f"https://cookpad.com/search/{q}"

def _image_to_small_data_url(path: str, max_side: int = 768, quality: int = 80, max_kb: int = 800) -> str:
    """画像を縮小＆圧縮して data URL に変換"""
    if not PIL_OK:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = min(1.0, float(max_side) / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    q = quality
    for _ in range(6):
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=q, optimize=True)
        data = buf.getvalue()
        if len(data) <= max_kb * 1024 or q <= 50:
            break
        q -= 5
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

# =====================
# Vision Autoname
# =====================
def vision_autoname(image_path: str) -> str:
    if not OPENROUTER_API_KEY or not OPENROUTER_MODEL:
        return ""
    try:
        image_data_url = _image_to_small_data_url(image_path)
        messages = [
            {"role": "system", "content":
             "画像から食材名を1語で推定。日本語10文字以内。ブランド/料理名は不可。"},
            {"role": "user", "content": [
                {"type": "text", "text": "この画像の主な食材名を1語（日本語10文字以内）で返してください。"},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ]},
        ]
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        body = {"model": OPENROUTER_MODEL, "messages": messages, "temperature": 0.2}
        resp = SESSION.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=body,
            timeout=(15, 240)
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        return content.splitlines()[0].strip(" 　「」[]()")
    except requests.exceptions.Timeout:
        print("[Vision Autoname Timeout]", image_path)
        return ""
    except requests.exceptions.ConnectionError as e:
        print("[Vision Autoname ConnError]", e)
        return ""
    except Exception:
        print("[Vision Autoname Error]", traceback.format_exc())
        return ""

# =====================
# レシピ（省略版: 同じ）
# =====================
def fallback_recipes(ingredients: List[str]) -> List[Dict]:
    return [{"title": "野菜炒め", "ingredients": ingredients, "steps": ["切る","炒める"], "time": "10分", "servings": 2}]

def llm_recipes(ings_with_qty: List[Dict], constraints: Dict) -> List[Dict]:
    if not OPENROUTER_API_KEY:
        return []
    sys_prompt = "あなたは家庭料理アシスタントです。JSON形式で5件返してください。"
    flat = [f"{i['name']}x{i['quantity']}{i['unit']}" for i in ings_with_qty if i.get("name")]
    user_prompt = f"食材: {', '.join(flat)} 制約: {constraints}"
    try:
        body = {
            "model": OPENROUTER_MODEL,
            "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
            "response_format": {"type": "json_object"},
        }
        resp = SESSION.post("https://openrouter.ai/api/v1/chat/completions",
                            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"},
                            json=body, timeout=(15, 240))
        resp.raise_for_status()
        return json.loads(resp.json()["choices"][0]["message"]["content"]).get("recipes", [])
    except Exception:
        return []

# =====================
# images スキャン
# =====================
def import_images_autoscan() -> Tuple[int, int]:
    if not os.path.isdir(IMAGES_DIR):
        return 0, 0
    items = load_db()
    existing_urls = {it.get("image_url") for it in items}
    found = 0
    added = 0
    for filename in os.listdir(IMAGES_DIR):
        if not allowed_file(filename):
            continue
        found += 1
        url = f"/static/images/{secure_filename(filename)}"
        if url in existing_urls:
            continue
        name = vision_autoname(os.path.join(IMAGES_DIR, filename)) or os.path.splitext(filename)[0]
        items.append({
            "id": str(uuid.uuid4()), "name": name,
            "expiry": "", "quantity": 1, "unit": "個",
            "category": "", "location": "冷蔵", "is_archived": False,
            "image_url": url, "created_at": datetime.utcnow().isoformat() + "Z"
        })
        added += 1
    if added:
        save_db(items)
    return added, found

# =====================
# APIルート（抜粋）
# =====================
@app.route("/")
def index():
    return render_template("indexj.html")

@app.route("/api/items", methods=["GET"])
def items_api():
    return jsonify(load_db())

@app.route("/api/recipes", methods=["POST"])
def recipes_api():
    data = request.get_json(force=True)
    items = load_db()
    ings = [{"name": it["name"], "quantity": it["quantity"], "unit": it["unit"]} for it in items]
    recs = llm_recipes(ings, data.get("constraints", {}))
    if not recs:
        recs = fallback_recipes([i["name"] for i in items])
    return jsonify({"recipes": recs})

@app.route("/api/images/rescan", methods=["POST"])
def images_rescan():
    added, found = import_images_autoscan()
    return jsonify({"added": added, "found": found})

@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(STATIC_DIR, filename)

# =====================
# 起動時
# =====================
if __name__ == "__main__":
    added, found = import_images_autoscan()
    print(f"[Images Autoscan] added {added} / found {found}")
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)

