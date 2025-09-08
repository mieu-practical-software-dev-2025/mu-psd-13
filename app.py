from flask import Flask, request, jsonify, send_from_directory
import os
import requests

app = Flask(__name__, static_folder="static")

# 疑似冷蔵庫データ
fridge = []

# OpenRouter API Key（環境変数から取得推奨）
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/add_item', methods=['POST'])
def add_item():
    data = request.json
    fridge.append(data)
    return jsonify({"status": "ok", "fridge": fridge})

@app.route('/get_items', methods=['GET'])
def get_items():
    return jsonify(fridge)

@app.route('/get_menu', methods=['GET'])
def get_menu():
    if not fridge:
        return jsonify({"menu": "冷蔵庫が空です。食材を追加してください。"})

    items = ", ".join([item["name"] for item in fridge])
    prompt = f"以下の食材を使って献立を箇条書きのみ5個ほど出力: {items}"

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "openai/gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "あなたは家庭の料理アドバイザーです。"},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 150
    }

    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions",
                                 headers=headers, json=data)
        result = response.json()
        # レスポンス確認用
        print(result)

        # choices が存在するか確認
        if "choices" in result and len(result["choices"]) > 0:
            menu = result["choices"][0]["message"]["content"]
        else:
            menu = f"API 応答に問題があります: {result}"
    except Exception as e:
        menu = f"エラーが発生しました: {e}"

    return jsonify({"menu": menu})

@app.route('/clear', methods=['POST'])
def clear():
    fridge.clear()
    return jsonify({"status": "cleared"})

if __name__ == '__main__':
    app.run(debug=True)
