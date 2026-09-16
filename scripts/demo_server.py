#!/usr/bin/env python3
"""Serve the dashboard plus local Keras upload and webcam prediction endpoints."""

from __future__ import annotations

import argparse
import cgi
import json
import mimetypes
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import tensorflow as tf

from traffic_sign_recognition.gradcam import explain
from traffic_sign_recognition.inference import predict_image


class DemoHandler(BaseHTTPRequestHandler):
    model = None
    web_root = Path("dist")
    explanation_dir = Path("artifacts/demo")

    def _json(self, payload, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_POST(self):
        if self.path != "/predict":
            self._json({"error": "Not found"}, 404); return
        form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": self.headers.get("Content-Type", "")})
        upload = form["image"] if "image" in form else None
        if upload is None or not getattr(upload, "file", None):
            self._json({"error": "Send an image field"}, 400); return
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temporary:
            temporary.write(upload.file.read()); image_path = Path(temporary.name)
        try:
            DemoHandler.explanation_dir.mkdir(parents=True, exist_ok=True)
            result = predict_image(self.model, image_path)
            result.update(explain(self.model, image_path, DemoHandler.explanation_dir / "latest_gradcam.png"))
            result["gradcam_url"] = "/artifacts/demo/latest_gradcam.png"
            self._json(result)
        except Exception as error:
            self._json({"error": str(error)}, 422)
        finally:
            image_path.unlink(missing_ok=True)

    def do_GET(self):
        requested = self.path.split("?", 1)[0].lstrip("/") or "index.html"
        if requested.startswith("artifacts/demo/"):
            file_path = Path(requested)
        else:
            file_path = self.web_root / requested
        if not file_path.is_file():
            self.send_error(404); return
        data = file_path.read_bytes(); self.send_response(200); self.send_header("Content-Type", mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=Path("artifacts/models/mobilenet_v2/model.keras"))
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--web-root", type=Path, default=Path("dist"))
    args = parser.parse_args()
    DemoHandler.model = tf.keras.models.load_model(args.model_path)
    DemoHandler.web_root = args.web_root
    print(f"Open http://localhost:{args.port} with model {args.model_path}")
    ThreadingHTTPServer(("127.0.0.1", args.port), DemoHandler).serve_forever()


if __name__ == "__main__": main()
