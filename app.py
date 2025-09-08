import os
import requests
from flask import Flask, request, jsonify, send_from_directory
from openai import OpenAI
from dotenv import load_dotenv
import traceback
from werkzeug.exceptions import HTTPException

load_dotenv()

app = Flask(__name__, static_folder="static", static_url_path="")

# ===== OpenRouter (Claudeなど) =====
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY")   # Unsplash APIキー
GOOGLE_MAPS_JS_KEY  = os.getenv("GOOGLE_MAPS_JS_KEY")    # Google Maps JavaScript APIキー

@app.route("/")
def index():
    """static/index.html を返す"""
    return send_from_directory(app.static_folder, "index.html")

@app.get("/config")
def get_config():
    """フロントに公開設定を返す（Google Mapsキー）"""
    return jsonify({
        "GOOGLE_MAPS_JS_KEY": GOOGLE_MAPS_JS_KEY or ""
    })

@app.get("/config/health")
def config_health():
    """デバッグ用の設定確認"""
    key = os.getenv("GOOGLE_MAPS_JS_KEY", "")
    return jsonify({
        "has_key": bool(key),
        "masked_key_tail": key[-6:] if key else None
    })

@app.route("/search", methods=["POST"])
def search():
    """
    入力: { "query": "京都" }
    出力: {
      "result": <プレーンテキスト>,
      "image": <Unsplashの画像URL or None>
    }
    """
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()

    if not query:
        return jsonify({"error": "地名が入力されていません"}), 400

    try:
        # ---- AIに観光案内を依頼（日本国内限定 & ホテルはURL付き形式） ----
        prompt_sys = (
            "あなたは日本観光の専門家です。"
            "必ず日本国内に存在する観光地・料理・宿泊施設のみを回答してください。"
            "出力形式はプレーンテキストで、以下の3見出しをこの順に必ず入れる：\n"
            "観光地\n"
            "名物グルメ\n"
            "おすすめホテル\n"
            "観光地・グルメは「名前 - 説明」の形式。"
            "おすすめホテルは「ホテル名 - 説明 - 公式URL」の形式にしてください。"
            "最後にモデルルートを「n日目：〜」形式で示す。"
        )
        prompt_user = (
            f"日本の{query}について、上記フォーマットのプレーンテキストで回答してください。"
            "対象は日本国内に限定してください。"
        )

        completion = client.chat.completions.create(
            model="anthropic/claude-3.5-sonnet",
            messages=[
                {"role": "system", "content": prompt_sys},
                {"role": "user", "content": prompt_user},
            ],
            max_tokens=600,
            temperature=0.7,
        )
        result_text = completion.choices[0].message.content if completion and completion.choices else ""

    except Exception as e:
        print("[AI ERROR]", repr(e))
        traceback.print_exc()
        result_text = (
            "観光情報の生成に失敗しました。時間をおいて再試行してください。"
            "\n（原因候補: APIキー/クォータ/ネットワーク）"
        )

    # ---- Unsplash でイメージを1枚検索（日本寄せ）----
    img_url = None
    if UNSPLASH_ACCESS_KEY:
        try:
            res = requests.get(
                "https://api.unsplash.com/search/photos",
                params={
                    "query": f"{query} Japan",
                    "per_page": 1,
                    "orientation": "landscape"
                },
                headers={"Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"},
                timeout=8,
            )
            if res.ok:
                j = res.json()
                if j.get("results"):
                    img_url = j["results"][0]["urls"]["regular"]
        except Exception as e:
            print("[UNSPLASH WARN]", repr(e))

    return jsonify({"result": result_text, "image": img_url})


# ===== 共通エラーハンドラ =====
@app.errorhandler(Exception)
def handle_ex(e):
    code = 500
    if isinstance(e, HTTPException):
        code = e.code
    print("[ERROR]", repr(e))
    traceback.print_exc()
    return jsonify({"error": f"server_error_{code}", "message": str(e)}), code


if __name__ == "__main__":
    def mask(k): 
        return ("****" + k[-6:]) if k else "(empty)"
    print("[BOOT] OPENROUTER_API_KEY:", mask(os.getenv("OPENROUTER_API_KEY")))
    print("[BOOT] UNSPLASH_ACCESS_KEY:", mask(os.getenv("UNSPLASH_ACCESS_KEY")))
    print("[BOOT] GOOGLE_MAPS_JS_KEY :", mask(os.getenv("GOOGLE_MAPS_JS_KEY")))
    app.run(debug=True)
