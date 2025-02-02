import React from 'react';
import './App.css';
import WebcamCapture from './WebcamCapture';
import ErrorBoundary from './ErrorBoundary';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>AI-Based Smart Camera</h1>
        <ErrorBoundary>
          <WebcamCapture />
        </ErrorBoundary>
      </header>
    </div>
  );
}

export default App;
