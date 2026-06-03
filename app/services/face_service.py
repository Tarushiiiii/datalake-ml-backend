import numpy as np


def verify_face(frame):
    """
    Placeholder face verification.
    Real embedding comparison later.
    """

    similarity = np.random.uniform(0.85, 0.99)

    return {
        "matched": similarity > 0.90,
        "confidence": float(similarity),
        "user_id": "EMP001",
    }