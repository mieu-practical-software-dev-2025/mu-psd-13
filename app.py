import os
from flask import Flask, request, jsonify, send_from_directory
import json
import requests # ホテルAPI連携で利用
from openai import OpenAI # Import the OpenAI library
from dotenv import load_dotenv

# .envファイルから環境変数を読み込む
load_dotenv()

# Flaskアプリケーションのインスタンスを作成
# static_folderのデフォルトは 'static' なので、
# このファイルと同じ階層に 'static' フォルダがあれば自動的にそこが使われます。
app = Flask(__name__)

# 開発モード時に静的ファイルのキャッシュを無効にする
if app.debug:
    @app.after_request
    def add_header(response):
        # /static/ 以下のファイルに対するリクエストの場合
        if request.endpoint == 'static':
            response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
            response.headers['Pragma'] = 'no-cache' # HTTP/1.0 backward compatibility
            response.headers['Expires'] = '0' # Proxies
        return response


# OpenRouter APIキーと関連情報を環境変数から取得
# このキーはサーバーサイドで安全に管理してください
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
SITE_URL = os.getenv("YOUR_SITE_URL", "http://localhost:5000") # Default if not set
APP_NAME = os.getenv("YOUR_APP_NAME", "FlaskVueApp") # Default if not set

# URL:/ に対して、static/index.htmlを表示して
    # クライアントサイドのVue.jsアプリケーションをホストする
@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')
    
# URL:/send_api に対するメソッドを定義
@app.route('/send_api', methods=['POST'])
def send_api():
    if not OPENROUTER_API_KEY:
        app.logger.error("OpenRouter API key not configured.")
        return jsonify({"error": "OpenRouter API key is not configured on the server."}), 500

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        default_headers={ # Recommended by OpenRouter
            "HTTP-Referer": SITE_URL,
            "X-Title": APP_NAME,
        }
    )
    
    # POSTリクエストからJSONデータを取得
    data = request.get_json()

    # 'text'フィールドがリクエストのJSONボディに存在するか確認
    if not data or 'text' not in data:
        app.logger.error("Request JSON is missing or does not contain 'text' field.")
        return jsonify({"error": "Missing 'text' in request body"}), 400

    received_text = data['text']
    if not received_text.strip(): # 空文字列や空白のみの文字列でないか確認
        app.logger.error("Received text is empty or whitespace.")
        return jsonify({"error": "Input text cannot be empty"}), 400
    
    location_name = received_text.strip()

    # LLMに観光情報を要求するための詳細なシステムプロンプトを生成します。
    # このプロンプトは、AIに対してJSON形式で構造化されたデータを返すように指示しています。
    # これにより、フロントエンドで情報を扱いやすくなります。
    system_prompt = f"""
あなたは日本の旅行プランナーです。これから指定される日本の地名について、以下の情報をJSON形式で回答してください。

JSONのキーは `tourist_spots`, `local_food`, `model_route` としてください。

- `tourist_spots`: おすすめの観光地を3つ、`name` (名称)と `description` (簡単な説明) を含むオブジェクトのリストで挙げてください。
- `local_food`: 有名な食べ物や名物料理を3つ、`name` (名称)と `description` (簡単な説明) を含むオブジェクトのリストで挙げてください。
- `model_route`: 1日で楽しめるモデル観光ルートを、`morning`, `lunch`, `afternoon`, `dinner` のキーを持つオブジェクトで提案してください。
"""

    try:
        # OpenRouter APIを呼び出し
        # このような複雑な指示には、より高性能なモデル(例: google/gemini-1.5-pro-latest)を推奨します。
        # また、response_format={"type": "json_object"} を指定することで、AIにJSON形式での出力を強制できます。
        # (この機能は比較的新しいバージョンのOpenAIライブラリでサポートされています)
        chat_completion = client.chat.completions.create(
            model="google/gemini-1.5-pro-latest", # より高性能なモデルを推奨
            response_format={"type": "json_object"}, # JSONモードを有効化
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": location_name}
            ],
        )
        
        # APIからのレスポンス(JSON文字列)を取得
        if chat_completion.choices and chat_completion.choices[0].message.content:
            try:
                # LLMからの応答はJSON文字列なので、Pythonの辞書にパースします
                processed_data = json.loads(chat_completion.choices[0].message.content)
                return jsonify({"message": "AIによってデータが処理されました。", "processed_data": processed_data})
            except json.JSONDecodeError:
                app.logger.error(f"Failed to parse JSON response from AI: {chat_completion.choices[0].message.content}")
                return jsonify({"error": "AIからの応答を解析できませんでした。"}), 500
        else:
            app.logger.error("AI response was empty.")
            return jsonify({"error": "AIから有効な応答がありませんでした。"}), 500

    except Exception as e:
        app.logger.error(f"OpenRouter API call failed: {e}")
        # クライアントには具体的なエラー詳細を返しすぎないように注意
        return jsonify({"error": f"AIサービスとの通信中にエラーが発生しました。"}), 500

# ホテル検索用のAPIエンドポイント (概念的な実装例)
@app.route('/api/hotels', methods=['GET'])
def search_hotels():
    location = request.args.get('location')
    if not location:
        return jsonify({"error": "Location parameter is required"}), 400

    # --- ここから下は、実際のホテル予約APIと連携するための実装例です ---
    # 例: 楽天トラベルホテル検索APIなど (別途APIキーの取得が必要です)
    # RAKUTEN_APP_ID = os.getenv("RAKUTEN_APP_ID")
    # if not RAKUTEN_APP_ID:
    #     return jsonify({"error": "Rakuten API key is not configured."}), 500
    #
    # # 実際のAPIリクエスト (これは楽天APIの例であり、仕様に合わせて変更が必要です)
    # api_url = "https://app.rakuten.co.jp/services/api/Travel/SimpleHotelSearch/20170426"
    # params = {
    #     "format": "json",
    #     "applicationId": RAKUTEN_APP_ID,
    #     "keyword": location, # 地名でキーワード検索
    #     "sort": "+hotelMinCharge" # 料金が安い順
    # }
    # try:
    #     response = requests.get(api_url, params=params)
    #     response.raise_for_status() # エラーがあれば例外を発生
    #     hotels_data = response.json()
    #     # ここでフロントエンドで使いやすいようにデータを整形します
    #     return jsonify(hotels_data)
    # except requests.exceptions.RequestException as e:
    #     app.logger.error(f"Hotel API call failed: {e}")
    #     return jsonify({"error": "ホテル情報の取得に失敗しました。"}), 500

    # --- モックデータ(サンプル)を返す例 ---
    # 実際のAPI連携を実装するまでの仮のデータです。
    mock_hotels = [
        {"name": f"{location}のホテルA", "price": 15000, "rating": 4.5, "url": "#"},
        {"name": f"{location}の旅館B", "price": 30000, "rating": 4.8, "url": "#"},
        {"name": f"{location}のビジネスホテルC", "price": 8000, "rating": 3.9, "url": "#"},
    ]
    return jsonify(sorted(mock_hotels, key=lambda h: h['price']))

# スクリプトが直接実行された場合にのみ開発サーバーを起動
if __name__ == '__main__':
    if not OPENROUTER_API_KEY:
        print("警告: 環境変数 OPENROUTER_API_KEY が設定されていません。API呼び出しは失敗します。")
    app.run(debug=True, host='0.0.0.0', port=5000)