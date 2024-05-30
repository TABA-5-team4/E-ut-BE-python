from pydantic import BaseModel
from fastapi import File

class ChatRequest(BaseModel):
    text: str

class STTRequest(BaseModel):
    voiceFile: bytes = File()