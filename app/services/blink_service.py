import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2
import mediapipe as mp
import numpy as np
import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("keras").setLevel(logging.ERROR)

print("✅ EAR Blink Detection Loaded")


# ─────────────────────────────────────────────────────────────
# MediaPipe
# ─────────────────────────────────────────────────────────────

mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


# ─────────────────────────────────────────────────────────────
# Blink Config
# ─────────────────────────────────────────────────────────────

BLINK_THRESHOLD = 0.23
REQUIRED_BLINKS = 2

blink_state = {
    "eyes_closed": False,
    "blink_count": 0,
}


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def eye_aspect_ratio(points):
    """
    EAR = (A + B) / (2 * C)

          p2      p3
            \    /
             \  /
    p1 -------  ------- p4
             /  \
            /    \
          p6      p5
    """

    A = euclidean(points[1], points[5])
    B = euclidean(points[2], points[4])
    C = euclidean(points[0], points[3])

    if C == 0:
        return 0.0

    return (A + B) / (2.0 * C)


# ─────────────────────────────────────────────────────────────
# Blink Detection
# ─────────────────────────────────────────────────────────────

def detect_blink(frame):
    try:

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=False,
        ) as face_mesh:

            results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return {
                "success": False,
                "blink_count": blink_state["blink_count"],
                "message": "No face detected",
            }

        face = results.multi_face_landmarks[0]

        h, w, _ = frame.shape

        points = [
            (int(lm.x * w), int(lm.y * h))
            for lm in face.landmark
        ]

        left_eye_points = [points[i] for i in LEFT_EYE]
        right_eye_points = [points[i] for i in RIGHT_EYE]

        left_ear = eye_aspect_ratio(left_eye_points)
        right_ear = eye_aspect_ratio(right_eye_points)

        ear = (left_ear + right_ear) / 2.0

        print(
            f"EAR={ear:.3f} "
            f"(L={left_ear:.3f}, R={right_ear:.3f})"
        )

        # Eye closed
        if ear < BLINK_THRESHOLD:

            if not blink_state["eyes_closed"]:
                blink_state["eyes_closed"] = True
                print("👁 Eyes Closed")

        # Eye open again
        else:

            if blink_state["eyes_closed"]:

                blink_state["blink_count"] += 1
                blink_state["eyes_closed"] = False

                print(
                    f"✅ Blink Count = "
                    f"{blink_state['blink_count']} / {REQUIRED_BLINKS}"
                )

        success = blink_state["blink_count"] >= REQUIRED_BLINKS

        if success:

            print("🎉 Blink Verification Complete")

            blink_state["blink_count"] = 0
            blink_state["eyes_closed"] = False

            return {
                "success": True,
                "blink_count": REQUIRED_BLINKS,
                "confidence": 1.0,
                "message": "Blink verification completed",
            }

        return {
            "success": False,
            "blink_count": blink_state["blink_count"],
            "confidence": float(1.0 - min(ear, 1.0)),
            "message": (
                f"Blink {blink_state['blink_count']}/{REQUIRED_BLINKS}"
            ),
        }

    except Exception as e:

        print("❌ BLINK ERROR:", str(e))

        return {
            "success": False,
            "blink_count": 0,
            "message": str(e),
        }