from flask import Flask, jsonify, send_from_directory, make_response
from flask_cors import CORS
import sqlite3
from datetime import datetime, timedelta
import os

app = Flask(__name__)
CORS(app)

DB_FILE = "classification_events.db"

def query_db(query, args=(), one=False):
    """Helper function to query SQLite database."""
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute(query, args)
    result = cursor.fetchall()
    conn.close()
    return (result[0] if result else None) if one else result

@app.route("/")
def serve_dashboard():
    """Serves the dashboard page."""
    return send_from_directory(os.path.dirname(__file__), "dashboard.html")

@app.route("/api/daily_counts")
def daily_counts():
    """Returns rooster crows per hour for the current day."""
    query = """
        SELECT strftime('%H', timestamp) AS hour, COUNT(*)
        FROM events
        WHERE label = 'rooster'
          AND timestamp >= datetime('now', 'start of day')
        GROUP BY hour
        ORDER BY hour;
    """
    results = query_db(query)

    response = make_response(jsonify({hour: count for hour, count in results}))
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return response

@app.route("/api/weekly_counts")
def weekly_counts():
    """Returns rooster crows per day for the past 7 days."""
    query = """
        SELECT strftime('%Y-%m-%d', timestamp) AS day, COUNT(*)
        FROM events
        WHERE label = 'rooster'
          AND timestamp >= date('now', '-6 days')
        GROUP BY day
        ORDER BY day;
    """
    results = query_db(query)

    response = make_response(jsonify({day: count for day, count in results}))
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return response

@app.route("/api/last_60_minutes_counts")
def last_60_minutes_counts():
    """Returns rooster crows for the last 60 minutes, grouped by minute."""
    query = """
        SELECT strftime('%Y-%m-%dT%H:%M', timestamp) AS minute, COUNT(*)
        FROM events
        WHERE label = 'rooster'
          AND timestamp >= datetime('now', '-60 minutes')
        GROUP BY minute
        ORDER BY minute;
    """
    results = query_db(query)

    # Build a dictionary for the last 60 minutes based on current time.
    now = datetime.now()
    last_60 = {}
    # Create keys for each minute from (now - 59 minutes) to now.
    for i in range(60):
        minute_time = now - timedelta(minutes=59 - i)
        key = minute_time.strftime('%Y-%m-%dT%H:%M')
        last_60[key] = 0

    # Fill in counts from the query results.
    for minute, count in results:
        if minute in last_60:
            last_60[minute] = count

    response = make_response(jsonify(last_60))
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return response


@app.route("/api/heatmap_data")
def heatmap_data():
    """Returns rooster crows for each hour of the last 7 days (7x24 grid)."""
    query = """
        SELECT strftime('%Y-%m-%d', timestamp) AS day, strftime('%H', timestamp) AS hour, COUNT(*)
        FROM events
        WHERE label = 'rooster'
          AND timestamp >= datetime('now', '-6 days', 'start of day')
        GROUP BY day, hour
        ORDER BY day, hour;
    """
    results = query_db(query)

    # Initialize a full 7-day x 24-hour grid with 0 counts
    last_7_days = [(datetime.utcnow() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(6, -1, -1)]
    heatmap = {day: {str(h).zfill(2): 0 for h in range(24)} for day in last_7_days}

    # Fill in available data
    for day, hour, count in results:
        heatmap[day][hour] = count

    response = make_response(jsonify(heatmap))
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return response

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
