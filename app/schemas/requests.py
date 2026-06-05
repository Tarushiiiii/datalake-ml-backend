from typing import Optional
from pydantic import BaseModel

class FrameRequest(BaseModel):

    frame: str
    session_id: Optional[str] = None
    timestamp: Optional[str] = None
    step: Optional[str] = None