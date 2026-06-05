"""
backend/app/services/face_service.py

Face verification using InsightFace buffalo_l + cosine similarity.
Matches notebook exactly — no changes needed from original.
"""

import cv2
import pickle
import numpy as np
from insightface.app import FaceAnalysis


# ── Load InsightFace ──────────────────────────────────────────────────────────

app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=-1, det_size=(640, 640))
print("✅ InsightFace loaded")


# ── Load face DB ──────────────────────────────────────────────────────────────

DB_PATH = "models/face_db.pkl"

try:
    with open(DB_PATH, "rb") as f:
        database = pickle.load(f)
    print("✅ Face DB loaded:", list(database.keys()))
except Exception as e:
    print("❌ Face DB failed:", e)
    database = {}


# ── Verification ──────────────────────────────────────────────────────────────

THRESHOLD = 0.55


def verify_face(frame):
    if frame is None:
        return {"success": False, "message": "Invalid frame — could not decode image"}
    try:
        faces = app.get(frame)

        if len(faces) == 0:
            return {"success": False, "message": "No face detected"}

        query_embedding = faces[0].embedding

        best_user  = None
        best_score = -1

        for user, db_embedding in database.items():
            score = np.dot(query_embedding, db_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(db_embedding)
            )
            if score > best_score:
                best_score = score
                best_user  = user

        success = best_score > THRESHOLD

        print(f"Best match: {best_user}  Score: {best_score:.4f}  ({'✅' if success else '❌'})")

        return {
            "success":  success,
            "message":  "Face verified" if success else "Face mismatch",
            "identity": best_user if success else None,
            "score":    float(best_score),
        }

    except Exception as e:
        print("Face verification error:", e)
        return {"success": False, "message": str(e)}