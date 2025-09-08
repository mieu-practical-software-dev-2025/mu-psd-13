import os
import json
import traceback
from urllib.parse import quote
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, static_folder="static", static_url_path="/static")
UPLOAD_FOLDER = os.path.join(app.static_folder, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 簡易メモリDB（本番はDBへ）
fridge = []

# OpenRouter クライアント
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    print("WARNING: OPENROUTER_API_KEY is not set.")
client = OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")


@app.route("/")
def index():
    # ルートは index.html を返す
    return send_from_directory("static", "index.html")


# ---------- 食材API ----------
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


# ---------- 献立提案（作り方 + Cookpad検索URL） ----------
@app.route("/suggest_menu", methods=["POST"])
def suggest_menu():
    try:
        if not fridge:
            return jsonify({
                "message": "冷蔵庫が空です。食材を追加してください。",
                "recipes": []
            })

        food_names = [f["name"] for f in fridge]
        prompt = (
            "以下の食材をできるだけ活用して、日本語で家庭向けの簡単なレシピを1〜5件提案してください。"
            "必ず JSON で返し、スキーマは "
            "{\"recipes\":[{\"title\":\"...\",\"steps\":[\"...\"],\"notes\":\"(省略可)\"}]}"
            " のみ。各レシピは 3〜8 ステップの箇条書きで、材料や代替案があれば簡潔に書いてください。"
            "URLは返さないでください（URLはサーバ側で生成します）。\n"
            f"食材: {', '.join(food_names)}"
        )

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.7,
        )

        content = resp.choices[0].message.content  # JSON文字列
        obj = json.loads(content)                  # dictへ

        recipes = obj.get("recipes", [])
        cleaned = []
        # Cookpad 検索URLを自動付与
        # パスセグメント型: https://cookpad.com/search/<キーワード>
        for r in recipes:
            title = (r.get("title") or "").strip()
            steps = r.get("steps") or []
            steps = [s.strip() for s in steps if isinstance(s, str) and s.strip()]
            if not title or not steps:
                continue

            # 検索精度を上げるために主要食材も加味（上位3件）
            main_ings = " ".join([f["name"] for f in fridge][:3])
            q = quote(f"{title} {main_ings}".strip())
            search_url = f"https://cookpad.com/search/{q}"

            cleaned.append({
                "title": title,
                "steps": steps,
                "search_url": search_url,
                "notes": r.get("notes", "")
            })

        if not cleaned:
            return jsonify({"message": "献立が生成できませんでした。", "recipes": []})

        return jsonify({"message": "", "recipes": cleaned})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


# ---------- SPA fallback（存在しないパスでも index.html を返す） ----------
@app.errorhandler(404)
def spa_fallback(_):
    return send_from_directory("static", "index.html"), 200


if __name__ == "__main__":
    # 開発用: 0.0.0.0 で外部アクセス可
    app.run(host="0.0.0.0", port=5000, debug=True)
