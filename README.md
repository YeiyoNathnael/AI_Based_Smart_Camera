
# AI-Based Smart Camera System

## Description

The AI-Based Smart Camera System is a cutting-edge application that leverages artificial intelligence to identify and capture real-time photos. The system processes the live camera feed to detect objects and scenes, then provides descriptions and warnings to the user based on the analysis. Whether you're using it for security, home automation, or creative projects, this smart camera system integrates advanced AI capabilities to enhance everyday photography.

## Features

- **Real-Time Image Capture:** Continuously captures photos from a connected camera.
- **AI-Powered Analysis:** Uses AI algorithms to identify objects and analyze scenes in real time.
- **User Alerts:** Displays descriptions and warnings based on the AI analysis.
- **Easy Setup:** Minimal configuration required to get started on your device.

## Requirements

Before running the project, ensure you have the following installed:

- **Node.js:** Required for running the AI services.
- **Python 3:** Required for the backend services.

Additionally, install the necessary Python packages by running:

```bash
pip install fastapi uvicorn gtts pillow opencv-python numpy pydantic openai requests pytest
```

## Installation and Setup

1. **Clone the Repository:**

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install Python Dependencies:**

   Run the following command to install all required Python packages:

   ```bash
   pip install fastapi uvicorn gtts pillow opencv-python numpy pydantic openai requests pytest
   ```

3. **Start the AI Services:**

   Navigate to the `ai_services` folder and run the Node.js service:

   ```bash
   cd ai_services
   npm start
   ```

4. **Run the FastAPI Server:**

   Open a separate terminal window in the root directory and run:

   ```bash
   uvicorn main:app --reload
   ```

   This starts the FastAPI server (by default at `http://127.0.0.1:8000`).

## How to Run the Project

After completing the installation and setup steps above, follow these steps to run the project on your device:

1. **Ensure all dependencies are installed:**  
   Confirm that both Node.js and Python dependencies are properly installed.

2. **Start the AI Services:**  
   In the `ai_services` folder, run:

   ```bash
   npm start
   ```

3. **Launch the Backend Server:**  
   In a separate terminal window at the project's root directory, run:

   ```bash
   uvicorn main:app --reload
   ```

4. **Connect Your Camera:**  
   Once both services are running, connect your camera. The system will automatically begin processing the live feed, providing real-time descriptions and warnings based on what it identifies.


