import os
import json
import uuid
import base64
import traceback
from datetime import datetime, date
from typing import List, Dict, Tuple
from collections import deque

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

# =====================
# LLM設定
# =====================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL   = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")  # 任意に変更可

app = Flask(__name__, template_folder=TEMPLATES, static_folder=STATIC_DIR)

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
            {"role": "system", "content":
             "画像から食材名を1語で推定。日本語10文字以内。ブランド/料理名は不可。"},
            {"role": "user", "content": [
                {"type": "text", "text": "この画像の主な食材名（日本語1語/10文字以内）を1つだけ返してください。"},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ]},
        ]
        headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
        body = {"model": OPENROUTER_MODEL, "messages": messages, "temperature": 0.2}
        resp = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=body, timeout=60)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        content = content.splitlines()[0].strip(" 　「」[]()")
        return content if 0 < len(content) <= 20 else ""
    except Exception:
        print("[Vision Autoname Error]", traceback.format_exc())
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
        resp = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=body, timeout=60)
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
def import_seed_autoscan() -> Tuple[int, int]:
    if not os.path.isdir(SEED_DIR):
        return 0, 0
    items = load_db()
    existing_urls = {it.get("image_url") for it in items}
    found = 0
    added = 0
    for filename in os.listdir(SEED_DIR):
        if not allowed_file(filename):
            continue
        found += 1
        safe = secure_filename(filename)
        url = f"/static/seed/{safe}"
        if url in existing_urls:
            continue
        auto_name = ""
        if OPENROUTER_API_KEY:
            try:
                auto_name = vision_autoname(os.path.join(SEED_DIR, safe))
            except Exception:
                auto_name = ""
        items.append({
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
        })
        added += 1
    if added:
        save_db(items)
    return added, found

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

@app.route("/api/items/bulk_delete", methods=["POST"])
def bulk_delete():
    data = request.get_json(force=True)
    ids = set(data.get("ids", []))
    items = load_db()
    keep, removed = [], []
    for it in items:
        if it["id"] in ids:
            removed.append(it)
        else:
            keep.append(it)
    if removed:
        push_undo([x.copy() for x in removed])
        # 物理ファイル
        for it in removed:
            if it.get("image_url","").startswith("/static/uploads/"):
                try:
                    path = os.path.join(BASE_DIR, it["image_url"].lstrip("/"))
                    if os.path.exists(path):
                        os.remove(path)
                except Exception:
                    pass
        save_db(keep)
    return jsonify({"deleted": [x["id"] for x in removed]})

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
        return jsonify({"added": added, "found": found})
    except Exception:
        return jsonify({"error": "rescan failed"}), 500

@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(STATIC_DIR, filename)

if __name__ == "__main__":
    try:
        added, found = import_seed_autoscan()
        if added:
            print(f"[Seed Autoscan] added {added} / found {found}")
    except Exception:
        print("[Seed Autoscan Error]", traceback.format_exc())
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)
