import React, { useState } from "react";
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  LineElement,
  CategoryScale,
  LinearScale,
  PointElement,
  Legend,
} from "chart.js";

ChartJS.register(LineElement, CategoryScale, LinearScale, PointElement, Legend);

function App() {
  const [pred, setPred] = useState([]);
  const [actual, setActual] = useState([]);
  const [mse, setMse] = useState(null);

  const fetchData = async () => {
    try {
      const res = await fetch("http://127.0.0.1:8000/predict");
      const json = await res.json();

      setPred(json.predicted);
      setActual(json.actual);
      setMse(json.mse);
    } catch (err) {
      console.error("Error fetching:", err);
    }
  };

  const chartData = {
    labels: pred.map((_, i) => i),
    datasets: [
      {
        label: "Predicted",
        data: pred,
        borderColor: "cyan",
        tension: 0.3,
      },
      {
        label: "Actual",
        data: actual,
        borderColor: "orange",
        tension: 0.3,
      },
    ],
  };

  return (
    <div style={{
      textAlign: "center",
      padding: "40px",
      background: "#0f172a",
      color: "white",
      minHeight: "100vh"
    }}>
      <h1>📊 Stock Prediction App</h1>

      <button
        onClick={fetchData}
        style={{
          padding: "10px 20px",
          margin: "20px",
          fontSize: "16px",
          borderRadius: "8px",
          cursor: "pointer"
        }}
      >
        Run Prediction
      </button>

      {mse && <h3>MSE: {mse.toFixed(4)}</h3>}

      {pred.length > 0 && (
        <div style={{ width: "80%", margin: "auto" }}>
          <Line data={chartData} />
        </div>
      )}
    </div>
  );
}

export default App;