# backend/app/schemas/responses.py

from typing import Literal, Optional
from pydantic import BaseModel
from typing import Optional

class HeadMovementResponse(BaseModel):
    success: bool
    message: Optional[str] = None


class BlinkDetectionResponse(BaseModel):
    success: bool
    blink_count: Optional[int] = None
    message:     Optional[str] = None


class FaceRecognitionResponse(BaseModel):
    matched:    bool
    confidence: float
    user_id:    Optional[str] = None
    message:    Optional[str] = None
