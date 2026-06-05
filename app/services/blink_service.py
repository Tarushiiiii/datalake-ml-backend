import os
import math
import logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("keras").setLevel(logging.ERROR)

import cv2
import mediapipe as mp
import numpy as np

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

# ── Suppress Keras 3 C-level stdout on model load ────────────────────────────

def _load_model_silently(path):
    from keras.models import load_model
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_fd = os.dup(1)
    os.dup2(devnull_fd, 1)
    os.close(devnull_fd)
    try:
        model = load_model(path, compile=False, safe_mode=False)
    finally:
        os.dup2(saved_fd, 1)
        os.close(saved_fd)
    return model


eye_model = _load_model_silently("models/blink_eye.keras")
print("✅ Eye model loaded")

        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=False,
        ) as face_mesh:

# ── MediaPipe singleton (static_image_mode avoids video-tracking state) ──────

_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,   # notebook used refine_landmarks=True
    min_detection_confidence=0.5,
)


# ── Landmark indices (EXACTLY as in notebook) ─────────────────────────────────

LEFT_EYE  = [33,  160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


# ── EAR (Eye Aspect Ratio) ────────────────────────────────────────────────────

def _ear(eye_pts):
    """
    Notebook formula:
        A = dist(eye[1], eye[5])
        B = dist(eye[2], eye[4])
        C = dist(eye[0], eye[3])
        EAR = (A + B) / (2 * C)
    """
    def dist(p, q):
        return math.sqrt((p[0]-q[0])**2 + (p[1]-q[1])**2)

    A = dist(eye_pts[1], eye_pts[5])
    B = dist(eye_pts[2], eye_pts[4])
    C = dist(eye_pts[0], eye_pts[3])
    if C < 1e-6:
        return 0.0
    return (A + B) / (2.0 * C)


EAR_THRESHOLD = 0.22        # exactly as in notebook
MIN_BLINK_FRAMES = 2        # notebook uses this for video; for single frame
                             # we only check the instantaneous gate


# ── Eye crop (±10px padding — matches notebook) ───────────────────────────────

def _crop_eye(frame, eye_pts):
    xs = [p[0] for p in eye_pts]
    ys = [p[1] for p in eye_pts]
    x1 = max(min(xs) - 10, 0)
    y1 = max(min(ys) - 10, 0)
    x2 = min(max(xs) + 10, frame.shape[1])
    y2 = min(max(ys) + 10, frame.shape[0])
    return frame[y1:y2, x1:x2]

        print(
            f"EAR={ear:.3f} "
            f"(L={left_ear:.3f}, R={right_ear:.3f})"
        )

# ── Preprocessing (exactly as notebook) ──────────────────────────────────────

def _preprocess(eye_bgr):
    """Resize to 96×96, float32 /255, add batch dim."""
    eye = cv2.resize(eye_bgr, (96, 96))
    eye = eye.astype(np.float32) / 255.0
    return np.expand_dims(eye, axis=0)   # (1, 96, 96, 3)


# ── Main entry point ─────────────────────────────────────────────────────────

def detect_blink(frame) -> dict:
    """
    Args:
        frame: BGR numpy array from decode_base64_frame.

    Returns dict with:
        success (bool)   — True when dual gate fires (model > 0.5 AND EAR < 0.22)
        blink_count (int)
        confidence (float) — averaged model probability (0-1)
        ear (float)       — average EAR for diagnostics
        message (str)
    """
    if frame is None:
        return {"success": False, "blink_count": 0, "message": "Invalid frame"}

    # MediaPipe expects RGB — convert from BGR
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = _face_mesh.process(rgb)

                print(
                    f"✅ Blink Count = "
                    f"{blink_state['blink_count']} / {REQUIRED_BLINKS}"
                )

    face = results.multi_face_landmarks[0]
    h, w, _ = frame.shape
    points = [(int(lm.x * w), int(lm.y * h)) for lm in face.landmark]

    left_pts  = [points[i] for i in LEFT_EYE]
    right_pts = [points[i] for i in RIGHT_EYE]

    # ── EAR gate ─────────────────────────────────────────────────────────────
    left_ear  = _ear(left_pts)
    right_ear = _ear(right_pts)
    avg_ear   = (left_ear + right_ear) / 2.0
    ear_closed = avg_ear < EAR_THRESHOLD

    # ── Eye crops → model ─────────────────────────────────────────────────────
    left_crop  = _crop_eye(frame, left_pts)
    right_crop = _crop_eye(frame, right_pts)

            print("🎉 Blink Verification Complete")

    left_input  = _preprocess(left_crop)   # (1, 96, 96, 3)
    right_input = _preprocess(right_crop)  # (1, 96, 96, 3)

    # Batch both eyes in a single inference call (matches notebook)
    both = np.concatenate([left_input, right_input], axis=0)  # (2, 96, 96, 3)
    preds = eye_model.predict(both, verbose=0)                 # (2, 1)

    left_pred  = float(preds[0][0])
    right_pred = float(preds[1][0])
    avg_pred   = (left_pred + right_pred) / 2.0

    # ── Dual gate (notebook): model > 0.5 AND EAR < 0.22 ────────────────────
    model_closed = avg_pred > 0.5
    eye_closed   = model_closed and ear_closed

    print(
        f"Blink — L={left_pred:.3f} R={right_pred:.3f} avg={avg_pred:.3f} "
        f"EAR={avg_ear:.3f} model_closed={model_closed} ear_closed={ear_closed} "
        f"BLINK={eye_closed}"
    )

    return {
        "success":     eye_closed,
        "blink_count": 1 if eye_closed else 0,
        "confidence":  round(avg_pred, 4),
        "ear":         round(avg_ear, 4),
        "message":     "Blink detected" if eye_closed else "No blink detected",
    }
