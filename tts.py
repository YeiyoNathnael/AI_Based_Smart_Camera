from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from gtts import gTTS
import os

router = APIRouter()

@router.post("/tts")
async def text_to_speech(text: str = Query(...)):
    try:
        tts = gTTS(text=text, lang='en')
        tts.save("output.mp3")
        return FileResponse("output.mp3", media_type='audio/mpeg', filename="output.mp3")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error generating TTS audio.") from e

@router.delete("/tts")
async def delete_tts():
    try:
        if os.path.exists("output.mp3"):
            os.remove("output.mp3")
        return {"message": "TTS file deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error deleting TTS audio.") from e