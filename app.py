import os
import json
import traceback
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, static_folder="static", static_url_path="/static")
UPLOAD_FOLDER = os.path.join(app.static_folder, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 簡易メモリDB
fridge = []

# OpenRouter
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

# --- 食材API ---
@app.route("/add_food", methods=["POST"])
def add_food():
    try:
        food_name = request.form.get("name", "").strip()
        expiry = request.form.get("expiry", "").strip()
        file = request.files.get("image")

        if not food_name or not expiry or not file:
            return jsonify({"status": "error", "message": "食材名・消費期限・画像は必須です"}), 400

        filename = secure_filename(file.filename)
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(save_path)

        food_item = {
            "id": (max([f["id"] for f in fridge]) + 1) if fridge else 1,
            "name": food_name,
            "expiry": expiry,
            "image": f"/static/uploads/{filename}",
        }
        fridge.append(food_item)
        return jsonify({"status": "success", "food": food_item})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/get_foods", methods=["GET"])
def get_foods():
    return jsonify(fridge)

@app.route("/delete_food/<int:food_id>", methods=["DELETE"])
def delete_food(food_id):
    global fridge
    before = len(fridge)
    fridge = [f for f in fridge if f["id"] != food_id]
    return jsonify({"status": "success", "deleted": (before - len(fridge))})

# --- 献立提案 ---
@app.route("/suggest_menu", methods=["POST"])
def suggest_menu():
    try:
        if not fridge:
            return jsonify({"message": "冷蔵庫が空です。食材を追加してください。", "menu": []})

        food_names = [f["name"] for f in fridge]
        prompt = (
            "以下の食材を使って1〜5個の献立を提案してください。"
            "出力は必ずJSONで {\"menu\": [\"...\",\"...\"]} の形にしてください。\n"
            f"食材: {', '.join(food_names)}"
        )

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.7,
        )

        content = resp.choices[0].message.content  # JSON文字列
        obj = json.loads(content)                  # Pythonのdictに変換
        # 念のため整形
        if "menu" not in obj or not isinstance(obj["menu"], list):
            obj = {"menu": []}

        return jsonify(obj)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

# --- SPA fallback: どの未定義パスでも index.html を返す ---
@app.errorhandler(404)
def spa_fallback(_):
    return send_from_directory("static", "index.html"), 200

if __name__ == "__main__":
    # デバッグ用途: 0.0.0.0 で外部アクセス可
    app.run(host="0.0.0.0", port=5000, debug=True)
