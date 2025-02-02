import React, { useRef, useState, useEffect, useCallback } from 'react';
import Webcam from 'react-webcam';
import axios from 'axios';
import SpeechRecognition, { useSpeechRecognition } from 'react-speech-recognition';
import './styles.css';

const WebcamCapture = () => {
  console.log('Rendering WebcamCapture component');
  const webcamRef = useRef(null);
  const [description, setDescription] = useState('');
  const [detectedCommand, setDetectedCommand] = useState('');
  const { transcript, listening, resetTranscript } = useSpeechRecognition();

  const deleteTTS = useCallback(async () => {
    try {
      await axios.delete('http://localhost:8000/tts');
    } catch (error) {
      console.error('Error deleting TTS audio:', error);
    }
  }, []);

  const playTTS = useCallback(async (text) => {
    try {
      const response = await axios.post('http://localhost:8000/tts', null, {
        params: { text },
        responseType: 'blob' // Ensure the response is treated as a Blob
      });
      const audioUrl = window.URL.createObjectURL(response.data);
      const audio = new Audio(audioUrl);
      audio.play();
      audio.onended = async () => {
        await deleteTTS(); // Delete the TTS file after it is played
      };
    } catch (error) {
      console.error('Error generating TTS audio:', error);
    }
  }, [deleteTTS]);

  const capture = useCallback(async () => {
    console.log('Capturing image...');
    await deleteTTS(); // Delete the existing TTS file before capturing a new image
    const imageSrc = webcamRef.current.getScreenshot();
    console.log('Image captured:', imageSrc);
    try {
      const response = await axios.post('http://localhost:8000/process_frame', {
        image_data: imageSrc.split(',')[1] // Remove the base64 header
      });
      console.log('Response:', response.data);
      setDescription(response.data.description);
      playTTS(response.data.description);
    } catch (error) {
      console.error('Error processing image:', error);
    }
  }, [playTTS, deleteTTS]);

  const handleVoiceCommand = useCallback(async (command) => {
    try {
      const response = await axios.post('http://localhost:8000/voice_command', null, {
        params: { command }
      });
      const mappedCommand = response.data.command;
      if (mappedCommand === 'capture') {
        capture();
      } else if (mappedCommand === 'repeat') {
        playTTS(description);
      }
    } catch (error) {
      console.error('Error processing voice command:', error);
    }
  }, [capture, playTTS, description]);

  useEffect(() => {
    console.log('Component mounted');
    console.log('Transcript:', transcript);
    setDetectedCommand(transcript);
    if (transcript.toLowerCase().includes('eye')) {
      console.log('Eye command detected');
      capture();
      resetTranscript();
    } else {
      handleVoiceCommand(transcript);
      resetTranscript();
    }
  }, [transcript, resetTranscript, capture, handleVoiceCommand]);

  return (
    <div className="webcam-container">
      <h1>AI-Based Smart Camera</h1>
      <div className="webcam-wrapper">
        <Webcam
          audio={false}
          ref={webcamRef}
          screenshotFormat="image/jpeg"
          className="webcam"
        />
      </div>
      <div className="controls">
        <button onClick={capture}>Capture</button>
        <button onClick={SpeechRecognition.startListening}>Start Listening</button>
      </div>
      {listening && <p className="status">Listening...</p>}
      {detectedCommand && <p className="status">Detected Command: {detectedCommand}</p>}
      {description && <p className="description">Description: {description}</p>}
    </div>
  );
};

export default WebcamCapture;
