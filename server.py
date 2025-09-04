import os, math, json, requests
from flask import Flask, request, jsonify, send_from_directory
from openai import OpenAI
from dotenv import load_dotenv

# ===== .env =====
load_dotenv()
OPENTRIPMAP_KEY = os.getenv("OPENTRIPMAP_API_KEY")     # 必須
GOOGLE_KEY      = os.getenv("GOOGLE_MAPS_API_KEY")     # 任意（ホテルを使うなら必須）
OPENROUTER_KEY  = os.getenv("OPENROUTER_API_KEY")      # 任意（/send_api を使うなら）
SITE_URL = os.getenv("YOUR_SITE_URL", "http://localhost:5000")
APP_NAME = os.getenv("YOUR_APP_NAME", "FlaskVueApp")

# ===== Flask =====
app = Flask(__name__, static_folder=".", static_url_path="")

# ルート: 同一オリジンで index.html を配る
@app.route("/", methods=["GET"])
def index():
    return send_from_directory(".", "index.html")

# 観光地検索：/api/attractions?place=京都
@app.route("/api/attractions", methods=["GET"])
def attractions():
    place = (request.args.get("place") or "").strip()
    if not place:
        return jsonify({"error": "place is required"}), 400
    if not OPENTRIPMAP_KEY:
        return jsonify({"error": "OPENTRIPMAP_API_KEY missing"}), 500

    try:
        # 1) 地名→座標
        g = requests.get(
            "https://api.opentripmap.com/0.1/ja/places/geoname",
            params={"name": place, "apikey": OPENTRIPMAP_KEY},
            timeout=10
        )
        g.raise_for_status()
        gj = g.json()
        lat, lon = gj.get("lat"), gj.get("lon")
        if lat is None or lon is None:
            return jsonify({"place": place, "foods": [], "attractions": []})

        # 2) 周辺POI（半径3km/最大12件）
        r = requests.get(
            "https://api.opentripmap.com/0.1/ja/places/radius",
            params={
                "radius": 3000, "lat": lat, "lon": lon,
                "kinds": "interesting_places,tourist_facilities,architecture,museums,urban_environment",
                "limit": 12, "format": "json", "apikey": OPENTRIPMAP_KEY
            },
            timeout=12
        )
        r.raise_for_status()
        items = r.json()

        # 3) 詳細→整形
        atts = []
        for it in items:
            xid = it.get("xid")
            if not xid:
                continue
            d = requests.get(
                f"https://api.opentripmap.com/0.1/ja/places/xid/{xid}",
                params={"apikey": OPENTRIPMAP_KEY},
                timeout=10
            )
            if not d.ok:
                continue
            dj = d.json()
            point = dj.get("point") or {}
            addr  = dj.get("address") or {}
            preview = (dj.get("preview") or {}).get("source")
            desc = (dj.get("wikipedia_extracts") or {}).get("text") or (dj.get("info") or {}).get("descr") or ""

            atts.append({
                "id": dj.get("xid"),
                "name": dj.get("name") or "スポット",
                "description": desc,
                "imageUrl": preview or "",
                "address": ", ".join([v for v in [addr.get("state"), addr.get("city"), addr.get("road")] if v]),
                "location": {"lat": (point.get("lat") if point else None),
                             "lng": (point.get("lon") if point else None)},
                "nearestTransit": None
            })

        foods_map = {
            "京都": ["湯豆腐", "抹茶スイーツ", "にしんそば"],
            "札幌": ["ジンギスカン", "スープカレー", "札幌ラーメン"],
            "福岡": ["博多ラーメン", "もつ鍋", "明太子"],
        }
        return app.response_class(
            response=json.dumps({"place": place, "foods": foods_map.get(place, []), "attractions": atts}, ensure_ascii=False),
            status=200, mimetype="application/json"
        )
    except requests.exceptions.RequestException as e:
        app.logger.exception("OpenTripMap error: %s", e)
        return jsonify({"error": "OpenTripMap request failed"}), 502

# ホテル検索：/api/hotels?lat=..&lng=..&limit=4
@app.route("/api/hotels", methods=["GET"])
def hotels():
    lat = request.args.get("lat", type=float)
    lng = request.args.get("lng", type=float)
    limit = request.args.get("limit", default=4, type=int)
    if lat is None or lng is None:
        return jsonify({"error": "lat/lng required"}), 400
    if not GOOGLE_KEY:
        return jsonify({"error": "GOOGLE_MAPS_API_KEY missing"}), 500

    try:
        data = requests.get(
            "https://maps.googleapis.com/maps/api/place/nearbysearch/json",
            params={
                "location": f"{lat},{lng}",
                "radius": 1500,
                "type": "lodging",
                "language": "ja",
                "key": GOOGLE_KEY
            },
            timeout=12
        )
        data.raise_for_status()
        dj = data.json()

        def haversine(lat1, lon1, lat2, lon2):
            R = 6371000.0
            phi1, phi2 = math.radians(lat1), math.radians(lat2)
            dphi = math.radians(lat2 - lat1)
            dlmb = math.radians(lon2 - lon1)
            a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlmb/2)**2
            return 2 * R * math.asin(math.sqrt(a))

        hotels = []
        for h in (dj.get("results") or [])[:limit]:
            hl = (h.get("geometry") or {}).get("location") or {}
            dist = round(haversine(lat, lng, hl["lat"], hl["lng"])) if {"lat","lng"} <= hl.keys() else None
            pid = h.get("place_id")
            hotels.append({
                "name": h.get("name"),
                "price": None,
                "currency": "JPY",
                "distanceMeters": dist,
                "address": h.get("vicinity"),
                "url": f"https://www.google.com/maps/place/?q=place_id:{pid}" if pid else None
            })
        return jsonify({"hotels": hotels})
    except requests.exceptions.RequestException as e:
        app.logger.exception("Google Places error: %s", e)
        return jsonify({"error": "Google Places request failed"}), 502

# （使う人だけ）LLM ルート：/send_api
@app.route('/send_api', methods=['POST', 'OPTIONS'])
def send_api():
    if request.method == "OPTIONS":
        return ("", 204)
    if not OPENROUTER_KEY:
        return jsonify({"error": "OpenRouter API key is not configured on the server."}), 500

    data = request.get_json(silent=True) or {}
    received_text = (data.get('text') or "").strip()
    if not received_text:
        return jsonify({"error": "Missing 'text' in request body"}), 400

    system_prompt = (data.get('context') or "140字以内で回答してください。").strip()
    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_KEY,
            default_headers={"HTTP-Referer": SITE_URL, "X-Title": APP_NAME}
        )
        chat = client.chat.completions.create(
            model="google/gemini-1.5-pro-latest",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": received_text}
            ],
        )
        content = chat.choices[0].message.content if chat and chat.choices else "AIから有効な応答がありませんでした。"
        return jsonify({"message": "AIによってデータが処理されました。", "processed_text": content})
    except Exception as e:
        app.logger.exception("OpenRouter API call failed: %s", e)
        return jsonify({"error": "AIサービスとの通信中にエラーが発生しました。"}), 502

# ===== dev helpers =====
if app.debug:
    @app.after_request
    def add_header(response):
        if request.endpoint == 'static':
            response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '0'
        return response

if __name__ == "__main__":
    if not OPENTRIPMAP_KEY:
        print("警告: OPENTRIPMAP_API_KEY が未設定です。/api/attractions は失敗します。")
    if not GOOGLE_KEY:
        print("注意: GOOGLE_MAPS_API_KEY が未設定です。/api/hotels は失敗します。")
    app.run(host="0.0.0.0", port=5000, debug=True)
