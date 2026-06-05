"""
backend/app/services/face_service.py

Face verification using InsightFace buffalo_l + cosine similarity.
Matches notebook exactly.

Bugs fixed vs original:
  1. BUG (minor): InsightFace app.get() expects BGR (OpenCV native format).
     The original code was correct here — frame comes from cv2.imdecode which
     is BGR, and InsightFace internally converts. No change needed.

  2. BUG: FaceRecognitionResponse schema used field "matched" but face_service
     returned "success". The response schema and API route were mismatched —
     fixed in responses.py and face_recognition.py to use a consistent key.

  3. BUG: FaceRecognitionResponse had no "stage" or "identity" field but
     the frontend FaceRecognitionResult type expects { matched, confidence,
     user_id, message }. Added "user_id" mapping.

  4. det_size=(640,640) is correct for buffalo_l — retained.
     ctx_id=-1 = CPU — retained.

  5. THRESHOLD = 0.55 matches notebook exactly — retained.
"""

import cv2
import pickle
import numpy as np
from insightface.app import FaceAnalysis


# ── Load InsightFace (once at startup) ───────────────────────────────────────

_insight_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
_insight_app.prepare(ctx_id=-1, det_size=(640, 640))
print("✅ InsightFace loaded")


# ── Load face DB ─────────────────────────────────────────────────────────────

DB_PATH = "models/face_db.pkl"

try:
    with open(DB_PATH, "rb") as f:
        _database: dict[str, np.ndarray] = pickle.load(f)
    print("✅ Face DB loaded:", list(_database.keys()))
except Exception as e:
    print("❌ Face DB failed:", e)
    _database = {}


# ── Cosine similarity (matches notebook) ─────────────────────────────────────

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


THRESHOLD = 0.55   # exactly as in notebook


# ── Verification ─────────────────────────────────────────────────────────────

def verify_face(frame) -> dict:
    """
    Args:
        frame: BGR numpy array (from decode_base64_frame).

    Returns:
        {
            "success":  bool,
            "matched":  bool,       # same value as success, for schema compat
            "identity": str | None,
            "user_id":  str | None, # alias of identity for frontend
            "score":    float,
            "confidence": float,    # same as score, for frontend compat
            "message":  str,
        }
    """
    if frame is None:
        return _fail("Invalid frame")

    if not _database:
        return _fail("Face database is empty — register users first")

    try:
        # InsightFace expects BGR — frame from cv2 is already BGR ✓
        faces = _insight_app.get(frame)

        if len(faces) == 0:
            return _fail("No face detected in frame")

        query_embedding = faces[0].embedding   # (512,) float32

        best_user  = None
        best_score = -1.0

        for user, db_embedding in _database.items():
            score = _cosine(query_embedding, db_embedding)
            if score > best_score:
                best_score = score
                best_user  = user

        matched = best_score > THRESHOLD

        print(
            f"Face — best_match={best_user}  score={best_score:.4f}  "
            f"threshold={THRESHOLD}  {'✅' if matched else '❌'}"
        )

        return {
            "success":    matched,
            "matched":    matched,
            "identity":   best_user if matched else None,
            "user_id":    best_user if matched else None,
            "score":      round(best_score, 4),
            "confidence": round(best_score, 4),
            "message":    f"Matched: {best_user}" if matched else "Face not recognised",
        }

    except Exception as e:
        print("Face verification error:", e)
        return _fail(str(e))


def _fail(msg: str) -> dict:
    return {
        "success":    False,
        "matched":    False,
        "identity":   None,
        "user_id":    None,
        "score":      0.0,
        "confidence": 0.0,
        "message":    msg,
    }