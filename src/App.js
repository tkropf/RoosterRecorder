import React, { useEffect, useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

function App() {
  const [dailyCounts, setDailyCounts] = useState(null);
  const [weeklyCounts, setWeeklyCounts] = useState(null);
  const [minuteCounts, setMinuteCounts] = useState(null);
  const [heatmapData, setHeatmapData] = useState(null);
  const [error, setError] = useState(null);

  // Use dynamic base URL for API calls based on the current hostname.
  // When accessed from the Internet, window.location.hostname will be "kropf.selfhost.eu".
  const baseUrl = `http://${window.location.hostname}:5000`;

  // --- LAST 60 MINUTES BAR CHART ---
  useEffect(() => {
    const fetchMinuteCounts = async () => {
      try {
        const response = await fetch(
          `${baseUrl}/api/last_60_minutes_counts?t=${Date.now()}`
        );
        const data = await response.json();

        // Sort the entries by key.
        // If the keys are numeric (as strings), compare numerically.
        // Otherwise, assume they are date strings and compare as Dates.
        const sortedEntries = Object.entries(data).sort((a, b) => {
          if (!isNaN(a[0]) && !isNaN(b[0])) {
            return parseInt(a[0], 10) - parseInt(b[0], 10);
          }
          return new Date(a[0]) - new Date(b[0]);
        });

        const now = new Date();

        // Ensure we have exactly 60 entries.
        // If the API returns fewer than 60, pad with 0.
        const counts = [];
        for (let i = 0; i < 60; i++) {
          if (i < sortedEntries.length) {
            counts.push(sortedEntries[i][1]);
          } else {
            counts.push(0);
          }
        }

        // Map the counts to an array of objects with a time label computed relative to "now."
        // The first entry corresponds to 59 minutes ago and the last to the current minute.
        const formattedData = counts.map((count, index) => {
          const minuteTime = new Date(now.getTime() - (59 - index) * 60000);
          const label = minuteTime.toLocaleTimeString([], {
            hour: "2-digit",
            minute: "2-digit",
          });
          return { minute: label, count };
        });

        setMinuteCounts(formattedData);
      } catch (error) {
        setError(error.message);
      }
    };

    fetchMinuteCounts();
    const interval = setInterval(fetchMinuteCounts, 60000); // Refresh every minute
    return () => clearInterval(interval);
  }, [baseUrl]);

  // --- DAILY (HOURLY) BAR CHART ---
  useEffect(() => {
    const fetchDailyCounts = async () => {
      try {
        const response = await fetch(
          `${baseUrl}/api/daily_counts?t=${Date.now()}`
        );
        const data = await response.json();

        // Create an array for all 24 hours with a default count of 0.
        const hourlyCounts = Array.from({ length: 24 }, (_, hour) => ({
          hour: `${hour}:00`,
          count: 0,
        }));
        // Update with actual counts from the API.
        Object.entries(data).forEach(([hour, count]) => {
          const h = parseInt(hour, 10);
          if (!isNaN(h) && h >= 0 && h < 24) {
            hourlyCounts[h] = { hour: `${h}:00`, count };
          }
        });
        setDailyCounts(hourlyCounts);
      } catch (error) {
        setError(error.message);
      }
    };

    fetchDailyCounts();
    const interval = setInterval(fetchDailyCounts, 60000); // Refresh every minute
    return () => clearInterval(interval);
  }, [baseUrl]);

  // --- WEEKLY BAR CHART ---
  useEffect(() => {
    const fetchWeeklyCounts = async () => {
      try {
        const response = await fetch(
          `${baseUrl}/api/weekly_counts?t=${Date.now()}`
        );
        const data = await response.json();

        const formattedData = Object.entries(data).map(([day, count]) => ({
          day,
          count,
        }));

        setWeeklyCounts(formattedData);
      } catch (error) {
        setError(error.message);
      }
    };

    fetchWeeklyCounts();
    const interval = setInterval(fetchWeeklyCounts, 60000); // Refresh every minute
    return () => clearInterval(interval);
  }, [baseUrl]);

  // --- HEATMAP CHART ---
  useEffect(() => {
    const fetchHeatmapData = async () => {
      try {
        const response = await fetch(
          `${baseUrl}/api/heatmap_data?t=${Date.now()}`
        );
        const data = await response.json();

        // Sort the days chronologically and format each day as DD.MM.
        const formattedData = Object.entries(data)
          .sort((a, b) => new Date(a[0]) - new Date(b[0]))
          .map(([day, hours]) => {
            const dateObj = new Date(day);
            const dayFormatted = dateObj.getDate().toString().padStart(2, "0");
            const monthFormatted = (dateObj.getMonth() + 1)
              .toString()
              .padStart(2, "0");
            return {
              day: `${dayFormatted}.${monthFormatted}.`,
              // Convert hourly data to bins sorted by hour.
              bins: Object.entries(hours)
                .map(([hour, count]) => ({
                  bin: parseInt(hour),
                  count,
                }))
                .sort((a, b) => a.bin - b.bin),
            };
          });

        setHeatmapData(formattedData);
      } catch (error) {
        setError(error.message);
      }
    };

    fetchHeatmapData();
    const interval = setInterval(fetchHeatmapData, 60000); // Refresh every minute
    return () => clearInterval(interval);
  }, [baseUrl]);

  // --- Heatmap scaling adjustments ---
  // Increase width by 30%: container and SVG width, plus scale the tile width and offset.
  const heatmapScale = 1.3;
  const tileWidth = 25 * heatmapScale; // originally 25px, now scaled (e.g., 32.5px)
  const xOffset = 50 * heatmapScale;

  return (
    <div style={{ padding: "20px" }}>
      <h1>🐓 Rooster Crow Dashboard</h1>

      {error && <p style={{ color: "red" }}>❌ Error: {error}</p>}

      {/* LAST 60 MINUTES BAR CHART */}
      <h2>⏳ Rooster Crows in Last 60 Minutes</h2>
      {minuteCounts ? (
        <ResponsiveContainer width="100%" height={300}>
          <BarChart
            data={minuteCounts}
            margin={{ top: 20, right: 30, left: 20, bottom: 60 }}
          >
            <XAxis
              dataKey="minute"
              label={{ value: "Time", position: "insideBottom", dy: 40 }}
            />
            <YAxis />
            <Tooltip />
            <Legend verticalAlign="top" />
            <Bar dataKey="count" fill="#ffc658" name="Crows per Minute" />
          </BarChart>
        </ResponsiveContainer>
      ) : (
        <p>Loading...</p>
      )}

      {/* DAILY CROW BAR CHART */}
      <h2>📊 Daily Rooster Crows (Hour-wise)</h2>
      {dailyCounts ? (
        <ResponsiveContainer width="100%" height={300}>
          <BarChart
            data={dailyCounts}
            margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
          >
            <XAxis dataKey="hour" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey="count" fill="#8884d8" name="Crows per Hour" />
          </BarChart>
        </ResponsiveContainer>
      ) : (
        <p>Loading...</p>
      )}

      {/* WEEKLY CROW BAR CHART */}
      <h2>📆 Weekly Rooster Crows (Day-wise)</h2>
      {weeklyCounts ? (
        <ResponsiveContainer width="100%" height={300}>
          <BarChart
            data={weeklyCounts}
            margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
          >
            <XAxis dataKey="day" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey="count" fill="#82ca9d" name="Crows per Day" />
          </BarChart>
        </ResponsiveContainer>
      ) : (
        <p>Loading...</p>
      )}

      {/* HEATMAP CHART */}
      <h2>🔥 Rooster Crow Heatmap (Last 7 Days x 24 Hours)</h2>
      {heatmapData ? (
        <div style={{ position: "relative", width: "910px", height: "300px" }}>
          <svg width="910" height="260">
            {/* Render heatmap tiles */}
            {heatmapData.map((row, rowIndex) =>
              row.bins.map((col) => {
                let fillColor;
                if (col.count === 0) {
                  fillColor = "#cccccc";
                } else if (col.count >= 1 && col.count <= 15) {
                  fillColor = "#ffcccc";
                } else if (col.count >= 16 && col.count <= 30) {
                  fillColor = "#ff9999";
                } else if (col.count >= 31 && col.count <= 45) {
                  fillColor = "#ff6666";
                } else if (col.count >= 46 && col.count <= 60) {
                  fillColor = "#ff3333";
                } else { // col.count > 60
                  fillColor = "#ff0000";
                }
                return (
                  <rect
                    key={`${row.day}-${col.bin}`}
                    x={col.bin * tileWidth + xOffset}
                    y={rowIndex * 30 + 30}
                    width={tileWidth}
                    height={30}
                    fill={fillColor}
                    stroke="#fff"
                  />
                );
              })
            )}

            {/* X-axis labels: Hours (0 to 23) */}
            {Array.from({ length: 24 }).map((_, hour) => (
              <text
                key={hour}
                x={hour * tileWidth + xOffset + tileWidth / 2}
                y={250} // Placed just below the heatmap tiles
                fontSize={10}
                textAnchor="middle"
                fill="#000"
              >
                {hour}
              </text>
            ))}

            {/* Y-axis labels: Dates in DD.MM. format */}
            {heatmapData.map((row, rowIndex) => (
              <text
                key={row.day}
                x={xOffset - 15}
                y={rowIndex * 30 + 45}
                fontSize={10}
                textAnchor="end"
                fill="#000"
              >
                {row.day}
              </text>
            ))}
          </svg>

          {/* Legend positioned below the SVG */}
          <div
            style={{
              position: "absolute",
              top: "270px",
              left: `${xOffset}px`,
              display: "flex",
              alignItems: "center",
            }}
          >
            <span style={{ fontSize: "12px", marginRight: "10px" }}>
              Legend:
            </span>
            <div
              style={{
                background:
                  "linear-gradient(to right, #cccccc, #ff0000)",
                width: "200px",
                height: "10px",
              }}
            ></div>
            <span style={{ fontSize: "12px", marginLeft: "10px" }}>
              Max Crows
            </span>
          </div>
        </div>
      ) : (
        <p>Loading...</p>
      )}
    </div>
  );
}

export default App;
