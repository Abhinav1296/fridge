"""SmartFridge Vision MVP — Flask web layer.

A thin HTTP wrapper around :mod:`vision_service` and :mod:`storage`:

* ``GET  /``            serves the single-page UI.
* ``POST /analyze``     accepts an image (multipart file upload OR base64 JSON from the
  webcam), validates and downscales it, hands the bytes to the vision service, records
  the result to the history database, and returns the parsed inventory as JSON.
* ``GET  /api/current`` returns the current fridge inventory (the most recent scan).
* ``GET  /api/history`` returns recent scans (timestamped inventory history) as JSON.

The web layer owns request/response concerns only. All vision logic lives in
:mod:`vision_service` and all persistence in :mod:`storage`, so a future agent layer can
bypass HTTP entirely.
"""

from __future__ import annotations

import base64
import binascii
import io
import logging
import re

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError

import storage
import vision_service
from vision_service import (
    RateLimitError,
    VisionAPIError,
    VisionConfigError,
    VisionParseError,
)

# Load .env before anything reads configuration.
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("smartfridge.app")

# --- Limits -----------------------------------------------------------------

MAX_IMAGE_BYTES = 5 * 1024 * 1024  # Reject decoded images larger than 5 MB.
MAX_DIMENSION = 1600  # Downscale so the longest side is at most this many pixels.
JPEG_QUALITY = 85

# Hard backstop on the raw request body. A 5 MB image base64-encoded inside a JSON
# body is ~6.7 MB, so allow headroom above MAX_IMAGE_BYTES; oversized bodies are
# turned into a friendly 413 by the error handler below.
MAX_CONTENT_LENGTH = 10 * 1024 * 1024

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# Ensure the history database exists before serving. A DB problem must not take the
# whole app down — analysis still works without history — so failures are logged, not
# raised.
try:
    storage.init_db()
except Exception:  # noqa: BLE001 — degrade gracefully if the DB can't be opened
    logger.exception("Could not initialize the history database; history disabled")

_DATA_URL_PREFIX = re.compile(r"^data:image/[^;]+;base64,", re.IGNORECASE)


class ImageInputError(Exception):
    """A client-side problem with the submitted image (bad/missing/oversized)."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


@app.get("/")
def index() -> str:
    """Serve the single-page UI."""
    return render_template("index.html")


@app.post("/analyze")
def analyze():
    """Analyze an uploaded or captured image and return the structured inventory."""
    source = _request_source()
    try:
        raw_bytes = _extract_image_bytes()
        prepared = _prepare_image(raw_bytes)
        result = vision_service.analyze_image(prepared)
    except ImageInputError as exc:
        return jsonify({"error": exc.message}), exc.status_code
    except VisionConfigError as exc:
        logger.error("Vision service misconfigured: %s", exc)
        return (
            jsonify(
                {
                    "error": "The vision service isn't configured on the server. "
                    "Check the VISION_API_KEY setting."
                }
            ),
            500,
        )
    except RateLimitError as exc:
        return jsonify({"error": str(exc)}), 429
    except VisionParseError as exc:
        return jsonify({"error": str(exc)}), 502
    except VisionAPIError as exc:
        return jsonify({"error": str(exc)}), 502
    except Exception:  # noqa: BLE001 — last-resort guard so we never leak a stack trace
        logger.exception("Unexpected error while analyzing image")
        return (
            jsonify({"error": "Something went wrong while analyzing the image."}),
            500,
        )

    # Record the scan to history. This is additive and must never break the response,
    # so a storage failure is logged and the analysis is still returned to the user.
    try:
        storage.save_scan(result, source)
    except Exception:  # noqa: BLE001 — history is best-effort, analysis is what matters
        logger.exception("Failed to record scan to history (analysis still returned)")

    return jsonify(result)


@app.get("/api/current")
def api_current():
    """Return the current fridge inventory — the most recent scan — for the Fridge tab.

    A scan is a full snapshot of the fridge, so the latest one is treated as "what's in
    the fridge now". Responds ``{"scan": null}`` when there is no history yet.
    """
    try:
        scan = storage.get_latest_scan()
    except Exception:  # noqa: BLE001 — a read failure shouldn't 500 the whole page
        logger.exception("Failed to read current inventory")
        return jsonify({"error": "Couldn't load your fridge right now."}), 500
    return jsonify({"scan": scan})


@app.get("/api/history")
def api_history():
    """Return recent scans (timestamped inventory history) as JSON for the History tab."""
    try:
        scans = storage.list_scans(limit=50)
    except Exception:  # noqa: BLE001 — a read failure shouldn't 500 the whole page
        logger.exception("Failed to read history")
        return jsonify({"error": "Couldn't load history right now."}), 500
    return jsonify({"scans": scans})


@app.errorhandler(413)
def handle_too_large(_error):
    """Turn Flask's raw 413 (body exceeds MAX_CONTENT_LENGTH) into a friendly JSON error."""
    return (
        jsonify({"error": "That image is too large. Please use one under 5 MB."}),
        413,
    )


def _request_source() -> str:
    """Classify how the image arrived, for the history record ('upload' vs 'webcam')."""
    uploaded = request.files.get("image")
    if uploaded is not None and uploaded.filename:
        return storage.SOURCE_UPLOAD
    return storage.SOURCE_WEBCAM


def _extract_image_bytes() -> bytes:
    """Pull raw image bytes from either a multipart upload or a base64 JSON body.

    Raises:
        ImageInputError: If no image is present, it is empty, it is not decodable
            base64, or it exceeds the size limit.
    """
    # Path 1: multipart/form-data file upload.
    uploaded = request.files.get("image")
    if uploaded is not None and uploaded.filename:
        data = uploaded.read()
        _check_size(data)
        if not data:
            raise ImageInputError("The uploaded file was empty.")
        return data

    # Path 2: JSON body with a base64 (optionally data-URL) image, e.g. from the webcam.
    if request.is_json:
        payload = request.get_json(silent=True) or {}
        b64 = payload.get("image_base64")
        if isinstance(b64, str) and b64.strip():
            data = _decode_base64_image(b64)
            _check_size(data)
            return data

    raise ImageInputError("No image was provided. Upload a file or capture one from the webcam.")


def _decode_base64_image(value: str) -> bytes:
    """Decode a base64 string (with or without a ``data:image/...;base64,`` prefix)."""
    stripped = _DATA_URL_PREFIX.sub("", value.strip())
    try:
        data = base64.b64decode(stripped, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ImageInputError("The captured image could not be decoded.") from exc
    if not data:
        raise ImageInputError("The captured image was empty.")
    return data


def _check_size(data: bytes) -> None:
    """Reject images larger than the configured limit with a 413."""
    if len(data) > MAX_IMAGE_BYTES:
        raise ImageInputError(
            "That image is too large. Please use one under 5 MB.", status_code=413
        )


def _prepare_image(data: bytes) -> bytes:
    """Validate the image and downscale/re-encode it to a reasonable JPEG.

    Any image whose longest side exceeds :data:`MAX_DIMENSION` is scaled down to cut
    payload size and latency. All images are re-encoded to JPEG so the vision service
    always receives a clean, known format.

    Raises:
        ImageInputError: If the bytes are not a decodable image.
    """
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageInputError(
            "That file doesn't look like a valid image. Please try a JPEG or PNG."
        ) from exc

    # Flatten transparency / palette modes onto white so JPEG encoding is safe.
    if image.mode not in ("RGB", "L"):
        if image.mode in ("RGBA", "LA", "P"):
            image = image.convert("RGBA")
            background = Image.new("RGBA", image.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, image).convert("RGB")
        else:
            image = image.convert("RGB")

    if max(image.size) > MAX_DIMENSION:
        image.thumbnail((MAX_DIMENSION, MAX_DIMENSION), Image.LANCZOS)

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    return buffer.getvalue()


if __name__ == "__main__":
    # Convenience for `python app.py`. `flask run` is the documented entry point.
    app.run(host="127.0.0.1", port=5000, debug=True)
