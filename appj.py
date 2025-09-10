import os
import json
import uuid
import base64
import traceback
import logging
import sys, itertools, threading, time
from datetime import datetime, date
from typing import List, Dict, Tuple
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import requests

load_dotenv()

# =====================
# パス設定
# =====================
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
TEMPLATES  = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")
UPLOAD_DIR = os.path.join(STATIC_DIR, "uploads")
SEED_DIR   = os.path.join(STATIC_DIR, "images")
DATA_DIR   = os.path.join(BASE_DIR, "data")
DB_PATH    = os.path.join(DATA_DIR, "items.json")

ALLOWED_EXT = {"png", "jpg", "jpeg", "gif", "webp"}

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(SEED_DIR,   exist_ok=True)
os.makedirs(DATA_DIR,   exist_ok=True)

DEBUG = os.getenv("DEBUG", "0") == "1"
logging.basicConfig(level=logging.DEBUG if DEBUG else logging.WARNING, format="%(levelname)s: %(message)s")

# =====================
# LLM設定
# =====================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL   = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")  # 任意に変更可

app = Flask(__name__, template_folder=TEMPLATES, static_folder=STATIC_DIR)

# =====================
# 速度チューニング設定（追加）
# =====================
SEED_FAST = os.getenv("SEED_FAST", "0") == "1"     # 1でLLM自動命名をスキップ（ファイル名をそのまま食材名）
MAX_WORKERS = int(os.getenv("SEED_MAX_WORKERS", str(min(32, (os.cpu_count() or 4) * 5))))  # I/O主体なので多め
AI_CONCURRENCY = int(os.getenv("SEED_AI_CONCURRENCY", "3"))  # LLM同時呼び出し上限（レート制限回避）

SESSION = requests.Session()
SESSION.headers.update({"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"})
AI_SEM = threading.Semaphore(AI_CONCURRENCY)

def vision_autoname(image_path: str) -> str:
    if not OPENROUTER_API_KEY or not OPENROUTER_MODEL:
        return ""
    # 縮小してから送る
    try:
        image_data_url = _img_b64_downscaled(image_path, max_side=640, jpeg_quality=80)
    except Exception as e:
        logging.debug(f"Downscale failed: {e}")
        image_data_url = _image_to_data_url(image_path)  # フォールバック

    messages = [
        {"role": "system", "content": "画像から食材名を1語で推定。日本語10文字以内。ブランド/料理名は不可。"},
        {"role": "user", "content": [
            {"type": "text", "text": "この画像の主な食材名（日本語1語/10文字以内）を1つだけ返してください。"},
            {"type": "image_url", "image_url": {"url": image_data_url}},
        ]},
    ]
    body = {"model": OPENROUTER_MODEL, "messages": messages, "temperature": 0.2}

    try:
        with AI_SEM:  # 同時実行を抑制
            resp = SESSION.post("https://openrouter.ai/api/v1/chat/completions", json=body, timeout=(10, 60))
        resp.raise_for_status()
        data = resp.json()
        content = ""
        if isinstance(data, dict) and data.get("choices"):
            content = (data["choices"][0].get("message", {}) or {}).get("content", "") or ""
        content = (content or "").splitlines()[0].strip(" 　「」[]()")
        return content if 0 < len(content) <= 20 else ""
    except Exception as e:
        logging.debug(f"Vision autoname failed: {e}")
        return ""


class ConsoleSpinner:
    def __init__(self, text="実行中…"):
        self.text = text
        self._stop = threading.Event()
        self._thr = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        for ch in itertools.cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"):
            if self._stop.is_set(): break
            sys.stdout.write(f"\r{self.text} {ch}")
            sys.stdout.flush()
            time.sleep(0.09)
        # 消して行を確定
        sys.stdout.write("\r" + " " * (len(self.text) + 4) + "\r")
        sys.stdout.flush()

    def start(self): self._thr.start()
    def stop(self, done_text="✔ 完了"):
        self._stop.set()
        self._thr.join()
        print(done_text, flush=True)



# =====================
# 簡易DB
# =====================
def load_db() -> List[Dict]:
    if not os.path.exists(DB_PATH):
        return []
    try:
        with open(DB_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            # 後方互換：新フィールドの既定値を補完
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
def _fmt_dur(sec: float) -> str:
    m, s = divmod(sec, 60.0)
    h, m = divmod(int(m), 60)
    return f"{h:02d}:{m:02d}:{s:06.3f}"

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def cookpad_search_url(keywords):
    from urllib.parse import quote_plus
    q = quote_plus(" ".join([k for k in keywords if k]))
    return f"https://cookpad.com/search/{q}"

def _image_to_data_url(path: str) -> str:
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    mime = "image/jpeg" if ext in {"jpg", "jpeg"} else f"image/{ext}"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def vision_autoname(image_path: str) -> str:
    if not OPENROUTER_API_KEY or not OPENROUTER_MODEL:
        return ""
    try:
        image_data_url = _image_to_data_url(image_path)
        messages = [
            {"role": "system", "content": "画像から食材名を1語で推定。日本語10文字以内。ブランド/料理名は不可。"},
            {"role": "user", "content": [
                {"type": "text", "text": "この画像の主な食材名（日本語1語/10文字以内）を1つだけ返してください。"},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ]},
        ]
        headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
        body = {"model": OPENROUTER_MODEL, "messages": messages, "temperature": 0.2}
        resp = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=body, timeout=60)

        # HTTPエラーはここで例外に（下の except で静かに握りつぶす）
        resp.raise_for_status()

        data = resp.json()
        # choices が無い/空でも KeyError にしない
        content = ""
        if isinstance(data, dict):
            if "choices" in data and data["choices"]:
                content = (data["choices"][0].get("message", {}) or {}).get("content", "") or ""
            elif "error" in data:
                # エラーは DEBUG の時だけ短くログ
                logging.debug(f"OpenRouter error: {data.get('error')}")
        content = (content or "").splitlines()[0].strip(" 　「」[]()")
        return content if 0 < len(content) <= 20 else ""
    except Exception as e:
        # ここでは**何も print しない**（必要なら DEBUG の時だけ短く）
        logging.debug(f"Vision autoname failed: {e}")
        return ""


# ===== レシピ
def fallback_recipes(ingredients: List[str]) -> List[Dict]:
    ideas = [
        {
            "title": "野菜炒め",
            "ingredients": ingredients,
            "steps": ["材料を食べやすく切る","油で硬い順に炒める","塩・こしょう・醤油で調える","仕上げにごま油"],
            "time": "15分", "servings": 2, "tags": ["簡単","時短"]
        },
        {
            "title": "具だくさん味噌汁",
            "ingredients": ingredients + ["だし","味噌","ねぎ"],
            "steps": ["具材を切る","出汁で煮る","火を止めて味噌を溶く","ねぎを散らす"],
            "time": "12分", "servings": 2, "tags": ["和食","汁物"]
        },
        {
            "title": "ペペロン風パスタ",
            "ingredients": ingredients + ["パスタ","にんにく","唐辛子","オリーブオイル"],
            "steps": ["パスタを茹でる","にんにくを弱火で香り出し","具材を加えて炒める","茹で汁で乳化し塩で整える"],
            "time": "20分", "servings": 1, "tags": ["麺","簡単"]
        },
        {
            "title": "親子丼（鶏・卵があれば）",
            "ingredients": ["鶏肉","玉ねぎ","卵","だし","醤油","みりん"],
            "steps": ["割下を沸かす","玉ねぎと鶏肉を煮る","溶き卵を回し入れる","半熟で火を止めご飯にのせる"],
            "time": "15分", "servings": 1, "tags": ["丼"]
        },
        {
            "title": "和風サラダ",
            "ingredients": ingredients + ["レタス","トマト","きゅうり","ツナ/豆腐"],
            "steps": ["野菜を切る","タンパク質を加える","和風ドレッシングで和える"],
            "time": "10分", "servings": 2, "tags": ["サラダ","ヘルシー"]
        },
    ]
    for it in ideas:
        it["url"] = cookpad_search_url([it["title"]] + ingredients)
    return ideas

def llm_recipes(ings_with_qty: List[Dict], constraints: Dict) -> List[Dict]:
    if not OPENROUTER_API_KEY or not OPENROUTER_MODEL:
        return []
    sys_prompt = (
        "あなたは日本語の家庭料理アシスタントです。"
        "与えられた手元の食材（名前・個数・単位）と制約（最大時間/スタイル/辛さ/ダイエット）を踏まえ、5件のレシピ案を提案してください。"
        "出力は必ずJSONオブジェクトで、キー'recipes'に配列。"
        "各要素は {title, ingredients, steps, time, servings, tags, url}。"
        "stepsは6ステップ以内、ingredientsは家庭の基本調味料の追加可。"
        "JSON以外の文章は書かない。"
    )
    flat = [f"{i.get('name','')}x{i.get('quantity',1)}{i.get('unit','')}" for i in ings_with_qty if i.get("name")]
    ctext = json.dumps(constraints, ensure_ascii=False)
    user_prompt = f"手元の食材: {', '.join(flat) if flat else 'なし'}\n制約: {ctext}"
    try:
        headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
        body = {
            "model": OPENROUTER_MODEL,
            "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
            "response_format": {"type": "json_object"},
            "temperature": 0.7,
        }
        resp = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=body,timeout=(15, 240))
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        data = json.loads(content)
        recs = data.get("recipes", [])
        for r in recs:
            if not r.get("url"):
                r["url"] = cookpad_search_url([r.get("title","")] + [i["name"] for i in ings_with_qty if i.get("name")])
        return recs
    except Exception:
        print("[LLM Recipes Error]", traceback.format_exc())
        return []

# =====================
# seedスキャン（期限空/数量1/追加フィールド既定）
# =====================
def import_seed_autoscan(progress=None, max_workers=5) -> Tuple[int, int]:
    if not os.path.isdir(SEED_DIR):
        return 0, 0

    items = load_db()
    existing_urls = {it.get("image_url") for it in items}
    files = [f for f in os.listdir(SEED_DIR) if allowed_file(f)]
    total = len(files)

    # --- 並列で処理する関数 ---
    def process_file(filename):
        safe = secure_filename(filename)
        url = f"/static/images/{safe}"
        if url in existing_urls:
            return None

        auto_name = ""
        if OPENROUTER_API_KEY:
            try:
                auto_name = vision_autoname(os.path.join(SEED_DIR, safe))
            except Exception as e:
                logging.debug(f"Vision autoname failed: {e}")
                auto_name = ""

        item = {
            "id": str(uuid.uuid4()),
            "name": auto_name or os.path.splitext(filename)[0],
            "expiry": "",
            "quantity": 1,
            "unit": "個",
            "category": "",
            "location": "冷蔵",
            "is_archived": False,
            "image_url": url,
            "created_at": datetime.utcnow().isoformat() + "Z",
        }
        return item

    added = 0
    processed = 0
    ai_names = []  # AIスキャン名を一時保存

    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futures = {exe.submit(process_file, f): f for f in files}
        for fut in as_completed(futures):
            processed += 1
            res = fut.result()
            if res:
                items.append(res)
                added += 1
                ai_names.append(res["name"])

            # 10件ごとに進捗出力
            if progress and processed % 10 == 0:
                progress(processed, added, total, ai_names)
                ai_names = []  # 表示したらクリア

    if added:
        save_db(items)

    return added, total


# =====================
# 検索・並び替え・フィルタ
# =====================
def parse_date(yMd: str):
    try:
        y, m, d = map(int, yMd.split("-"))
        return date(y, m, d)
    except Exception:
        return None

def days_until(yMd: str):
    d = parse_date(yMd)
    if not d:
        return None
    today = date.today()
    return (d - today).days

def apply_query_sort_filter(items: List[Dict], q: str, sort: str, order: str, flt: str, include_zero: bool, include_archived: bool):
    res = items[:]

    # 検索
    if q:
        ql = q.lower()
        def hit(it):
            for key in ("name", "category", "unit", "location"):
                v = str(it.get(key, "")).lower()
                if ql in v:
                    return True
            return False
        res = [it for it in res if hit(it)]

    # フィルタ
    def visible(it):
        if not include_archived and it.get("is_archived"):
            return False
        if not include_zero and (it.get("quantity", 0) == 0):
            return False
        if not flt:
            return True
        d = days_until(it.get("expiry","")) if it.get("expiry") else None
        if flt == "expired":
            return d is not None and d < 0
        if flt == "near_due":
            return d is not None and 0 <= d <= 2
        if flt == "this_week":
            return d is not None and 0 <= d <= 7
        if flt == "out_of_stock":
            return (it.get("quantity",0) == 0)
        if flt == "archived":
            return it.get("is_archived", False)
        return True

    res = [it for it in res if visible(it)]

    # 並び替え
    keyfn = None
    if sort == "expiry":
        def keyfn(it):
            d = parse_date(it.get("expiry","")) if it.get("expiry") else None
            return (date.max if d is None else d, it.get("name",""))
    elif sort == "created":
        def keyfn(it):
            try:
                return (datetime.fromisoformat(it.get("created_at","").replace("Z","")), it.get("name",""))
            except Exception:
                return (datetime.min, it.get("name",""))
    elif sort == "quantity":
        def keyfn(it):
            return (it.get("quantity",0), it.get("name",""))
    elif sort == "name":
        def keyfn(it):
            return (it.get("name",""),)
    else:
        def keyfn(it):
            return (it.get("name",""),)

    res.sort(key=keyfn, reverse=(order == "desc"))
    return res

# =====================
# Undo用バッファ（直近の削除を保持）
# =====================
UNDO_BUFFER = deque(maxlen=200)  # 要素: {"ts": datetime, "items": [item,...]}
UNDO_TTL_SEC = 90

def push_undo(items: List[Dict]):
    UNDO_BUFFER.append({"ts": datetime.utcnow(), "items": items})

def take_undo(ids: List[str]) -> List[Dict]:
    now = datetime.utcnow()
    restored = []
    for bucket in list(UNDO_BUFFER):
        if (now - bucket["ts"]).total_seconds() > UNDO_TTL_SEC:
            continue
        remain = []
        for it in bucket["items"]:
            if it["id"] in ids:
                restored.append(it)
            else:
                remain.append(it)
        bucket["items"] = remain
    return restored

# =====================
# ルーティング
# =====================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/meta", methods=["GET"])
def meta():
    return jsonify({
        "units": ["個", "g", "ml", "本", "パック"],
        "categories": ["", "野菜", "肉", "魚", "乳製品", "調味料", "冷凍", "常温"],
        "locations": ["冷蔵", "冷凍", "常温"],
    })

@app.route("/api/items", methods=["GET", "POST"])
def items_api():
    if request.method == "GET":
        items = load_db()
        q      = request.args.get("query","").strip()
        sort   = request.args.get("sort","name")
        order  = request.args.get("order","asc")
        flt    = request.args.get("filter","")
        include_zero = request.args.get("include_zero","false").lower() == "true"
        include_archived = request.args.get("include_archived","false").lower() == "true"
        items = apply_query_sort_filter(items, q, sort, order, flt, include_zero, include_archived)
        return jsonify(items)

    # POST: 画像アップロード
    if "image" not in request.files:
        return jsonify({"error": "image file required"}), 400

    file = request.files["image"]
    expiry   = request.form.get("expiry", "").strip()
    name     = request.form.get("name", "").strip()
    qty_raw  = request.form.get("quantity", "").strip()
    unit     = request.form.get("unit", "個").strip() or "個"
    category = request.form.get("category", "").strip()
    location = request.form.get("location", "冷蔵").strip() or "冷蔵"

    if not file or not allowed_file(file.filename):
        return jsonify({"error": "invalid image"}), 400

    filename = secure_filename(file.filename)
    ext = filename.rsplit(".", 1)[1].lower()
    new_name = f"{uuid.uuid4()}.{ext}"
    save_path = os.path.join(UPLOAD_DIR, new_name)
    file.save(save_path)

    auto_name = ""
    if not name and OPENROUTER_API_KEY:
        try:
            auto_name = vision_autoname(save_path)
        except Exception:
            auto_name = ""

    try:
        quantity = int(qty_raw) if qty_raw else 1
    except Exception:
        quantity = 1
    quantity = max(0, min(quantity, 999))

    item = {
        "id": str(uuid.uuid4()),
        "name": name or auto_name or os.path.splitext(filename)[0],
        "expiry": expiry,            # 空＝未設定
        "quantity": quantity,
        "unit": unit,
        "category": category,
        "location": location,
        "is_archived": False,
        "image_url": f"/static/uploads/{new_name}",
        "created_at": datetime.utcnow().isoformat() + "Z",
    }

    items = load_db()
    items.append(item)
    save_db(items)
    return jsonify(item), 201

@app.route("/api/items/<item_id>", methods=["DELETE", "PATCH"])
def item_detail(item_id):
    items = load_db()
    idx = next((i for i, it in enumerate(items) if it["id"] == item_id), None)
    if idx is None:
        return jsonify({"error": "not found"}), 404

    if request.method == "PATCH":
        data = request.get_json(force=True)
        for key in ("expiry","name","unit","category","location","is_archived"):
            if key in data:
                items[idx][key] = str(data[key]).strip() if isinstance(data[key], str) else data[key]
        if "quantity" in data:
            try:
                q = int(data["quantity"])
                items[idx]["quantity"] = max(0, min(q, 999))
            except Exception:
                pass
        save_db(items)
        return jsonify(items[idx])

    # DELETE (物理削除 + Undo用退避)
    target = items[idx]
    push_undo([target.copy()])  # 退避
    if target.get("image_url","").startswith("/static/uploads/"):
        try:
            path = os.path.join(BASE_DIR, target["image_url"].lstrip("/"))
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass
    items.pop(idx)
    save_db(items)
    return jsonify({"ok": True})

# 置き換え：/api/items/bulk_delete の実装だけ差し替え
@app.route("/api/items/bulk_delete", methods=["POST"])
def bulk_delete():
    # JSONが無い/壊れてる時に落ちないように
    data = request.get_json(silent=True) or {}
    ids_raw = data.get("ids", [])

    # 型・中身チェック（フロントのバグや空送信をはねる）
    if not isinstance(ids_raw, list):
        return jsonify({"error": "ids must be an array"}), 400
    ids = {str(x) for x in ids_raw if isinstance(x, (str, int)) and str(x).strip()}
    if not ids:
        return jsonify({"error": "no ids to delete"}), 400

    items = load_db()
    keep, removed = [], []
    for it in items:
        if it.get("id") in ids:
            removed.append(it)
        else:
            keep.append(it)

    if removed:
        push_undo([x.copy() for x in removed])
        # 物理ファイルの削除（アップロード分のみ）
        for it in removed:
            url = it.get("image_url", "")
            if isinstance(url, str) and url.startswith("/static/uploads/"):
                try:
                    path = os.path.join(BASE_DIR, url.lstrip("/"))
                    if os.path.exists(path):
                        os.remove(path)
                except Exception as e:
                    logging.debug(f"file remove failed: {e}")
        save_db(keep)

    # 何も該当しなかった場合も200で空配列を返す（フロントで正常扱いできる）
    return jsonify({"deleted": [x["id"] for x in removed]}), 200

@app.route("/api/undo_delete", methods=["POST"])
def undo_delete():
    data = request.get_json(force=True)
    ids = set(data.get("ids", []))
    if not ids:
        return jsonify({"restored": []})
    restored = take_undo(list(ids))
    if not restored:
        return jsonify({"restored": []})
    items = load_db()
    existing = {it["id"] for it in items}
    for it in restored:
        if it["id"] not in existing:
            items.append(it)
    save_db(items)
    return jsonify({"restored": [it["id"] for it in restored]})

@app.route("/api/items/bulk_update", methods=["POST"])
def bulk_update():
    data = request.get_json(force=True)
    ids = set(data.get("ids", []))
    patch = data.get("patch", {})
    items = load_db()
    changed = []
    for it in items:
        if it["id"] in ids:
            for k in ("expiry","unit","category","location","is_archived"):
                if k in patch:
                    it[k] = patch[k]
            if "quantity" in patch:
                try:
                    q = int(patch["quantity"])
                    it["quantity"] = max(0, min(q, 999))
                except Exception:
                    pass
            changed.append(it["id"])
    save_db(items)
    return jsonify({"updated": changed})

@app.route("/api/items/consume", methods=["POST"])
def consume():
    """
    body: { items: [{id, amount}] }
    """
    data = request.get_json(force=True)
    ops = data.get("items", [])
    items = load_db()
    idx_by_id = {it["id"]: i for i, it in enumerate(items)}
    changed = []
    for op in ops:
        iid = op.get("id")
        amt = int(op.get("amount", 0))
        if iid in idx_by_id and amt > 0:
            i = idx_by_id[iid]
            items[i]["quantity"] = max(0, items[i].get("quantity",0) - amt)
            changed.append(iid)
    save_db(items)
    return jsonify({"consumed": changed})

@app.route("/api/recipes", methods=["POST"])
def recipes_api():
    data = request.get_json(force=True)
    selected_ids = data.get("item_ids", [])
    constraints   = data.get("constraints", {})  # max_time/style/spicy/diet など
    items = load_db()
    selected = [i for i in items if i["id"] in selected_ids] if selected_ids else items
    ings = [{"name": it.get("name",""), "quantity": it.get("quantity", 1), "unit": it.get("unit","個")} for it in selected if it.get("name")]
    names_for_url = [it["name"] for it in selected if it.get("name")]

    source = "ai"
    recs = llm_recipes(ings, constraints)
    if not recs:
        source = "fallback"
        recs = fallback_recipes(names_for_url or ["食材"])
    return jsonify({"ingredients": names_for_url, "recipes": recs, "source": source})

@app.route("/api/shopping_list", methods=["POST"])
def shopping_list():
    """
    body: { requirements: [{name, quantity, unit}], include_archived: bool }
    """
    data = request.get_json(force=True)
    reqs = data.get("requirements", [])
    include_archived = bool(data.get("include_archived", False))
    items = load_db()
    stock = {}
    for it in items:
        if not include_archived and it.get("is_archived"):
            continue
        name = it.get("name","")
        if not name: 
            continue
        stock[name] = stock.get(name, 0) + int(it.get("quantity",0))
    result = []
    for r in reqs:
        name = r.get("name","")
        need = int(r.get("quantity", 0))
        unit = r.get("unit","")
        have = stock.get(name, 0)
        short = max(0, need - have)
        if short > 0:
            result.append({"name": name, "shortage": short, "unit": unit})
    return jsonify({"list": result})

@app.route("/api/substitutes", methods=["GET"])
def substitutes():
    name = request.args.get("name","").strip()
    rules = {
        "長ねぎ": ["玉ねぎ", "青ねぎ"],
        "牛乳": ["豆乳", "アーモンドミルク"],
        "バター": ["オリーブオイル", "マーガリン"],
        "みりん": ["砂糖+酒"],
    }
    out = rules.get(name, [])
    # LLMで拡張（任意）
    if not out and OPENROUTER_API_KEY and name:
        try:
            headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
            body = {
                "model": OPENROUTER_MODEL,
                "messages": [
                    {"role":"system","content":"日本語で、料理の置き換え食材候補を3〜5件、配列JSONのみで返してください。"},
                    {"role":"user","content": f"{name} の代替案"},
                ],
                "response_format":{"type":"json_object"},
                "temperature":0.2
            }
            resp = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=body, timeout=40)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            data = json.loads(content)
            out = data.get("items", []) or data.get("list", []) or []
        except Exception:
            pass
    return jsonify({"name": name, "alternatives": out})

@app.route("/api/seed/rescan", methods=["POST"])
def seed_rescan():
    try:
        added, found = import_seed_autoscan()
        # PowerShell に実行結果を表示
        print(f"▶ Rescan 実行: {found} 件スキャン、{added} 件追加", flush=True)
        return jsonify({"added": added, "found": found})
    except Exception as e:
        print("[Rescan Error]", e, flush=True)
        return jsonify({"error": "rescan failed"}), 500


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(STATIC_DIR, filename)

# 例：seed 自動スキャンの起動時に出す
if __name__ == "__main__":
    try:
        def _progress(processed, added, total, ai_names):
            names_str = "、".join(ai_names[:5])
            print(f"▶ 実行中… {processed}/{total} 件処理済み・追加 {added} 件 | AI名: {names_str}", flush=True)

        # ▼▼▼ 追加：開始時刻と計測開始 ▼▼▼
        start_dt = datetime.now()
        t0 = time.perf_counter()
        print(f"▶ 実行開始（イメージ 自動スキャン）", flush=True)
        # ▲▲▲ 追加ここまで ▲▲▲

        added, found = import_seed_autoscan(progress=_progress, max_workers=5)

        print(f"✔ 完了: {found} 件スキャン、{added} 件追加", flush=True)

        # ▼▼▼ 追加：終了時刻と経過時間を出力 ▼▼▼
        end_dt = datetime.now()
        elapsed = time.perf_counter() - t0
        print(f"⏱ 実行時間: {_fmt_dur(elapsed)}（約 {elapsed:.3f} 秒）", flush=True)
        # ▲▲▲ 追加ここまで ▲▲▲

    except Exception:
        print("[Seed Autoscan Error]", traceback.format_exc(), flush=True)

    print("▶ Flask サーバー起動中…", flush=True)
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)

