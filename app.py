import base64
import os
import cv2
import gdown
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best.pt")

DRIVE_URL = "https://drive.google.com/uc?id=1sPIpagCkQTtqhL_JvZVMFq3re6QOYYID"


if not os.path.exists(MODEL_PATH):
    print("Downloading best.pt model from Google Drive...")
    gdown.download(DRIVE_URL, MODEL_PATH, quiet=False)

model = YOLO(MODEL_PATH)

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "message": "YOLOv8 Mask Detection API is running"}), 200

@app.route("/detect", methods=["POST"])
def detect():
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "No image data provided"}), 400

        image_data = data["image"].split(",")[1] if "," in data["image"] else data["image"]
        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Invalid image data"}), 400

        results = model(frame, conf=0.4)[0]

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            detections.append({
                "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "confidence": round(conf, 2),
                "label": label
            })

        return jsonify(detections), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
