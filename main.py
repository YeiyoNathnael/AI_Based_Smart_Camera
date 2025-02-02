# main.py
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import base64
import io
from PIL import Image
import cv2
import numpy as np
import openai
from ai_processor import detect_objects, perform_ocr, generate_description
from fastapi.middleware.cors import CORSMiddleware
from tts import router as tts_router

app = FastAPI(
    title="AI-Based Smart Camera API",
    description="API for processing video snapshots on command",
    version="1.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

app.include_router(tts_router)

openai.api_key = "YOUR_OPENAI_API_KEY"

@app.get("/")
async def root():
    return {"message": "Welcome to the AI-Based Smart Camera API"}

# Define a Pydantic model for the JSON payload
class ImagePayload(BaseModel):
    image_data: str  # Expecting a base64-encoded string of the image

@app.post("/process_frame")
async def process_frame(payload: ImagePayload):
    try:
        image_bytes = base64.b64decode(payload.image_data)
        image = Image.open(io.BytesIO(image_bytes))
        image_cv = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image data provided.") from e

    detections = detect_objects(image_cv)
    ocr_text = perform_ocr(image_cv)
    description = generate_description(detections, ocr_text)

    result = {
        "description": description,
        "image_details": {
            "width": image_cv.shape[1],
            "height": image_cv.shape[0],
            "channels": image_cv.shape[2]
        }
    }
    return JSONResponse(content=result)

@app.post("/voice_command")
async def voice_command(command: str = Query(...)):
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"Map the following command to one of these: capture, repeat, ignore: {command}",
            max_tokens=10
        )
        mapped_command = response.choices[0].text.strip().lower()
        if mapped_command not in ["capture", "repeat", "ignore"]:
            mapped_command = "ignore"
        return {"command": mapped_command}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error processing voice command.") from e

def capture_and_process_frame():
    # Open a connection to the default camera (usually index 0)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Error: Could not open camera.")

    ret, frame = cap.read()
    if not ret:
        raise Exception("Error: Could not read frame from camera.")

    # Release the camera
    cap.release()

    # Process the frame
    detections = detect_objects(frame)
    ocr_text = perform_ocr(frame)
    description = generate_description(detections, ocr_text)

    return description

def listen_for_command():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    with microphone as source:
        print("Listening for command...")
        audio = recognizer.listen(source)

    try:
        command = recognizer.recognize_google(audio)
        print(f"Command received: {command}")
        if "what's in front of me" in command.lower():
            description = capture_and_process_frame()
            print(f"Description: {description}")
        else:
            print("Command not recognized.")
    except sr.UnknownValueError:
        print("Could not understand the audio.")
    except sr.RequestError as e:
        print(f"Could not request results; {e}")

if __name__ == "__main__":
    import threading
    import uvicorn

    # Start the FastAPI server in a separate thread
    server_thread = threading.Thread(target=uvicorn.run, args=("main:app",), kwargs={"host": "0.0.0.0", "port": 8000, "reload": True})
    server_thread.start()

    # Listen for voice commands in the main thread
    listen_for_command()

