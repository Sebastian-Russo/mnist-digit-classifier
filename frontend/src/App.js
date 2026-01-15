import React, { useRef, useState } from 'react';
import { ReactSketchCanvas } from 'react-sketch-canvas';
import './App.css';

function App() {
  const canvasRef = useRef(null);
  const [prediction, setPrediction] = useState(null);
  const [confidence, setConfidence] = useState(null);
  const [loading, setLoading] = useState(false);

  const handlePredict = async () => {
    setLoading(true);

    const imageData = await canvasRef.current.exportImage('png');
    const base64Image = imageData.split(',')[1];

    try {
      const response = await fetch('https://nwc8c4muag.execute-api.us-east-1.amazonaws.com/prod/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image: base64Image }),
      });

      const data = await response.json();

      // Parse nested response
      const result = JSON.parse(data.body);
      setPrediction(result.prediction);
      setConfidence(result.confidence);
    } catch (error) {
      console.error('Error:', error);
      alert('Failed to get prediction');
    }

    setLoading(false);
  };
  const handleClear = () => {
    canvasRef.current.clearCanvas();
    setPrediction(null);
    setConfidence(null);
  };

  return (
    <div className="App">
      <h1>MNIST Digit Classifier</h1>
      <p>Draw a digit (0-9) below:</p>

      <div className="canvas-container">
        <ReactSketchCanvas
          ref={canvasRef}
          strokeWidth={20}
          strokeColor="black"
          canvasColor="white"
          width="280px"
          height="280px"
        />
      </div>

      <div className="buttons">
        <button onClick={handlePredict} disabled={loading}>
          {loading ? 'Predicting...' : 'Predict'}
        </button>
        <button onClick={handleClear}>Clear</button>
      </div>

      {prediction !== null && (
        <div className="result">
          <h2>Prediction: {prediction}</h2>
          <p>Confidence: {confidence}%</p>
        </div>
      )}
    </div>
  );
}

export default App;