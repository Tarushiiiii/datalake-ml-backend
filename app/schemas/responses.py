from typing import Optional
from pydantic import BaseModel


class HeadMovementResponse(BaseModel):

    success: bool
    stage: Optional[str] = None
    message: Optional[str] = None
    confidence: Optional[float] = None
    session_id: Optional[str] = None


class BlinkDetectionResponse(BaseModel):
    success: bool
    blink_count: Optional[int] = None
    message: Optional[str] = None


class FaceRecognitionResponse(BaseModel):
    matched: bool
    confidence: float
    user_id: Optional[str] = None
    message: Optional[str] = None