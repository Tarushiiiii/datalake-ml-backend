import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2
import mediapipe as mp
import numpy as np
import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("keras").setLevel(logging.ERROR)

from keras.models import load_model
from keras.layers import Dense


# ── Patch Dense to ignore unsupported kwarg ───────────────────────────────────

_orig_dense_init = Dense.__init__

def _patched_dense_init(self, *args, **kwargs):
    kwargs.pop("quantization_config", None)
    _orig_dense_init(self, *args, **kwargs)

Dense.__init__ = _patched_dense_init


# ── Load model (suppress Keras 3 JSON config dump via OS-level fd redirect) ───

def _load_model_silently(path):
    """
    Keras 3 writes the full model config JSON directly to the C stdout fd,
    bypassing sys.stdout entirely. We must redirect at the OS level.
    """
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_fd   = os.dup(1)          # save a copy of stdout fd 1
    os.dup2(devnull_fd, 1)          # point fd 1 → /dev/null
    os.close(devnull_fd)
    try:
        model = load_model(path, compile=False, safe_mode=False)
    finally:
        os.dup2(saved_fd, 1)        # restore fd 1 → original stdout
        os.close(saved_fd)
    return model

eye_model = _load_model_silently("models/blink_eye.keras")
print("✅ Eye model loaded")


# ── MediaPipe ─────────────────────────────────────────────────────────────────

mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE  = [33, 133]
RIGHT_EYE = [362, 263]


# ── Eye crop ──────────────────────────────────────────────────────────────────

def crop_eye(frame, eye_points):
    x_coords = [p[0] for p in eye_points]
    y_coords = [p[1] for p in eye_points]

    x1 = max(min(x_coords) - 20, 0)
    y1 = max(min(y_coords) - 20, 0)
    x2 = min(max(x_coords) + 20, frame.shape[1])
    y2 = min(max(y_coords) + 20, frame.shape[0])

    return frame[y1:y2, x1:x2]


# ── Blink detection ───────────────────────────────────────────────────────────

def detect_blink(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=False,
    ) as face_mesh:
        results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return {"success": False, "blink_count": 0, "message": "No face detected"}

    face    = results.multi_face_landmarks[0]
    h, w, _ = frame.shape
    points  = [(int(lm.x * w), int(lm.y * h)) for lm in face.landmark]

    left_crop  = crop_eye(frame, [points[i] for i in LEFT_EYE])
    right_crop = crop_eye(frame, [points[i] for i in RIGHT_EYE])

    if left_crop.size == 0 or right_crop.size == 0:
        return {"success": False, "blink_count": 0, "message": "Eye crop failed"}

    def preprocess(eye):
        eye = cv2.resize(eye, (96, 96))
        eye = eye.astype(np.float32) / 255.0
        return np.expand_dims(eye, axis=0)

    left_crop  = preprocess(left_crop)
    right_crop = preprocess(right_crop)

    both_eyes  = np.concatenate([left_crop, right_crop], axis=0)
    preds      = eye_model(both_eyes, training=False).numpy()
    left_pred  = float(preds[0][0])
    right_pred = float(preds[1][0])
    prediction = (left_pred + right_pred) / 2

    print(f"Blink prediction: {prediction:.4f}  (L={left_pred:.3f} R={right_pred:.3f})")

    success = prediction > 0.5

    return {
        "success":     success,
        "blink_count": 1 if success else 0,
        "confidence":  prediction,
        "message":     "Blink detected" if success else "No blink",
    }