import React, { useState } from "react";
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  LineElement,
  CategoryScale,
  LinearScale,
  PointElement,
} from "chart.js";

ChartJS.register(LineElement, CategoryScale, LinearScale, PointElement);

function App() {
  const [data, setData] = useState([]);

  const fetchData = async () => {
    const res = await fetch("http://127.0.0.1:8000/predict");
    const json = await res.json();
    setData(json.prediction_sample);
  };

  const chartData = {
    labels: data.map((_, i) => i),
    datasets: [
      {
        label: "Predicted Values",
        data: data,
        borderColor: "cyan",
        fill: false,
      },
    ],
  };

  return (
    <div style={{ textAlign: "center", padding: "40px", background: "#0f172a", color: "white" }}>
      <h1>📊 Stock Prediction App</h1>

      <button onClick={fetchData} style={{ padding: "10px", margin: "20px" }}>
        Run Prediction
      </button>

      {data.length > 0 && (
        <div style={{ width: "80%", margin: "auto" }}>
          <Line data={chartData} />
        </div>
      )}
    </div>
  );
}

export default App;