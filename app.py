import os
import requests
from flask import Flask, request, jsonify, send_from_directory
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, static_folder="static", static_url_path="")

# OpenRouter
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY")  # Unsplash APIキー

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/search", methods=["POST"])
def search():
    data = request.get_json()
    query = data.get("query")

    if not query:
        return jsonify({"error": "地名が入力されていません"})

    try:
        # AIで観光案内を生成
        completion = client.chat.completions.create(
            model="anthropic/claude-3.5-sonnet",
            messages=[
                {"role": "system", "content": "あなたは観光案内の専門家です。"},
                {"role": "user", "content": f"日本の{query}について、観光地・有名な食べ物・おすすめホテルを教えてください。"}
            ],
            max_tokens=300
        )
        result_text = completion.choices[0].message.content

        # Unsplashで画像検索
        img_url = None
        if UNSPLASH_ACCESS_KEY:
            res = requests.get(
                "https://api.unsplash.com/search/photos",
                params={"query": query, "per_page": 1},
                headers={"Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"}
            )
            if res.status_code == 200 and res.json()["results"]:
                img_url = res.json()["results"][0]["urls"]["regular"]

        return jsonify({"result": result_text, "image": img_url})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
