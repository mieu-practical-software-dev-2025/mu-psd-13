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
    prompt = f"以下の食材を使って献立を提案してください。ただし、箇条書きで五つほどで大丈夫です.箇条書きのみ出力: {items}"

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "openai/gpt-4o-mini",  # お好きなモデルに変更可
        "messages": [
            {"role": "system", "content": "あなたは家庭の料理アドバイザーです。"},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 500
    }

    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions",
                                 headers=headers, json=data)
        result = response.json()
        print(result)

        if "choices" in result and len(result["choices"]) > 0:
            menu = result["choices"][0]["message"]["content"]
        else:
            menu = f"API応答に問題があります: {result}"
    except Exception as e:
        menu = f"エラーが発生しました: {e}"

    return jsonify({"menu": menu})

@app.route('/clear', methods=['POST'])
def clear():
    fridge.clear()
    return jsonify({"status": "cleared"})

@app.route('/remove_item', methods=['POST'])
def remove_item():
    data = request.json
    name = data.get("name")
    exp = data.get("exp")
    # fridge の中で一致するものを削除
    global fridge
    fridge = [item for item in fridge if not (item["name"] == name and item["exp"] == exp)]
    return jsonify({"status": "removed", "fridge": fridge})

if __name__ == '__main__':
    app.run(debug=True)
