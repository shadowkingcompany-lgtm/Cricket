import os
import json
import numpy as np
from flask import Flask, render_template, jsonify, request

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from cricket_engine import CricketEngine
from ml_predictor import CricketPredictor

app = Flask(__name__)

# ── paths ──────────────────────────────────────────────────────────────────
SESSIONS_FILE = os.path.join(os.path.dirname(__file__), 'sessions.json')
PROFILE_FILE  = os.path.join(os.path.dirname(__file__), 'profile.json')

# ── initialise cricket subsystem ───────────────────────────────────────────
cricket_engine    = CricketEngine(ball_interval=2.0)   # 2 s between balls
cricket_predictor = CricketPredictor()

def _on_match_complete(record):
    cricket_predictor.record_completed_match(record)

cricket_engine.set_match_complete_callback(_on_match_complete)


def init_db():
    """Initialise JSON data files if they do not exist."""
    if not os.path.exists(SESSIONS_FILE):
        mock_sessions = [
            {"id": 1, "date": "2026-07-01", "type": "batting",  "shot": "Defensive Block",  "metric1": 65, "metric2": 80, "metric3": 75, "score": 73},
            {"id": 2, "date": "2026-07-02", "type": "batting",  "shot": "Cover Drive",       "metric1": 70, "metric2": 72, "metric3": 68, "score": 70},
            {"id": 3, "date": "2026-07-03", "type": "bowling",  "shot": "Fast Bowling",       "metric1": 115,"metric2": 82, "metric3": 70, "score": 76},
            {"id": 4, "date": "2026-07-05", "type": "batting",  "shot": "Cover Drive",        "metric1": 78, "metric2": 85, "metric3": 80, "score": 81},
            {"id": 5, "date": "2026-07-06", "type": "batting",  "shot": "Pull Shot",          "metric1": 82, "metric2": 75, "metric3": 85, "score": 80},
            {"id": 6, "date": "2026-07-08", "type": "bowling",  "shot": "Spin Bowling",       "metric1": 78, "metric2": 85, "metric3": 85, "score": 83},
            {"id": 7, "date": "2026-07-10", "type": "batting",  "shot": "Sweep Shot",         "metric1": 85, "metric2": 88, "metric3": 90, "score": 88},
            {"id": 8, "date": "2026-07-11", "type": "bowling",  "shot": "Fast Bowling",       "metric1": 122,"metric2": 89, "metric3": 78, "score": 87},
        ]
        with open(SESSIONS_FILE, 'w') as f:
            json.dump(mock_sessions, f, indent=4)

    if not os.path.exists(PROFILE_FILE):
        default_profile = {
            "name": "Nawin",
            "role": "All-Rounder",
            "experience": "Intermediate",
            "battingHand": "Right-Handed",
            "bowlingStyle": "Right-Arm Fast Medium",
        }
        with open(PROFILE_FILE, 'w') as f:
            json.dump(default_profile, f, indent=4)


init_db()

# ══════════════════════════════════════════════════════════════════════════════
# Existing routes (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/sessions', methods=['GET', 'POST'])
def handle_sessions():
    if request.method == 'GET':
        try:
            with open(SESSIONS_FILE, 'r') as f:
                sessions = json.load(f)
            return jsonify(sessions)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        try:
            data = request.json
            with open(SESSIONS_FILE, 'r') as f:
                sessions = json.load(f)
            new_id = max((s['id'] for s in sessions), default=0) + 1
            data['id'] = new_id
            sessions.append(data)
            with open(SESSIONS_FILE, 'w') as f:
                json.dump(sessions, f, indent=4)
            return jsonify({"status": "success", "session": data})
        except Exception as e:
            return jsonify({"error": str(e)}), 400


@app.route('/api/profile', methods=['GET', 'POST'])
def handle_profile():
    if request.method == 'GET':
        try:
            with open(PROFILE_FILE, 'r') as f:
                profile = json.load(f)
            return jsonify(profile)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        try:
            data = request.json
            with open(PROFILE_FILE, 'w') as f:
                json.dump(data, f, indent=4)
            return jsonify({"status": "success", "profile": data})
        except Exception as e:
            return jsonify({"error": str(e)}), 400


@app.route('/api/predict', methods=['POST'])
def predict_performance():
    """
    ML prediction using scikit-learn.
    Trains on historical/synthetic performance log and inputs current
    profile parameters to predict next match performance values.
    """
    try:
        input_data = request.json
        avg_score       = float(input_data.get('avgScore',       75))
        recent_runs     = float(input_data.get('recentRuns',     45))
        recent_sr       = float(input_data.get('recentSR',      120))
        recent_wickets  = float(input_data.get('recentWickets',  1.2))
        training_hours  = float(input_data.get('trainingHours',   5))

        X = np.array([
            [60, 20, 80,  0.2, 2],
            [65, 25, 90,  0.5, 3],
            [70, 32, 105, 0.8, 4],
            [73, 40, 115, 1.0, 4],
            [78, 55, 130, 1.5, 6],
            [82, 60, 135, 1.8, 7],
            [85, 75, 150, 2.0, 8],
            [90, 88, 165, 2.5, 10],
            [95, 95, 180, 3.0, 12],
            [75, 45, 118, 1.2, 5],
            [72, 38, 110, 1.0, 5],
            [80, 52, 125, 1.4, 6],
        ])
        y_runs    = np.array([15, 22, 28, 38, 48, 55, 68, 80, 92, 42, 36, 49])
        y_sr      = np.array([75, 88, 102, 112, 128, 132, 145, 158, 172, 115, 108, 122])
        y_wickets = np.array([0.1, 0.3, 0.5, 0.8, 1.2, 1.5, 1.9, 2.3, 2.8, 1.0, 0.9, 1.3])
        y_economy = np.array([9.5, 9.0, 8.2, 7.8, 7.2, 6.8, 6.4, 6.0, 5.5, 7.9, 8.1, 7.4])
        y_impact  = np.array([4.2, 4.8, 5.5, 6.1, 7.2, 7.8, 8.5, 9.2, 9.8, 6.4, 5.9, 7.0])

        X_query = np.array([[avg_score, recent_runs, recent_sr, recent_wickets, training_hours]])

        def _fit_predict(y):
            return float(LinearRegression().fit(X, y).predict(X_query)[0])

        pred_runs    = max(0.0,  min(150.0, _fit_predict(y_runs)))
        pred_sr      = max(30.0, min(250.0, _fit_predict(y_sr)))
        pred_wickets = max(0.0,  min(10.0,  _fit_predict(y_wickets)))
        pred_economy = max(2.5,  min(15.0,  _fit_predict(y_economy)))
        pred_impact  = max(1.0,  min(10.0,  _fit_predict(y_impact)))

        scaler       = StandardScaler()
        coef_model   = LinearRegression().fit(scaler.fit_transform(X), y_impact)
        importances  = list(coef_model.coef_)
        feature_names = ["Practice Score", "Recent Runs", "Recent SR",
                         "Recent Wickets", "Weekly Training"]
        importance_mapping = {n: round(float(v), 3)
                              for n, v in zip(feature_names, importances)}

        return jsonify({
            "status": "success",
            "predictions": {
                "runs":       round(pred_runs,    1),
                "strikeRate": round(pred_sr,      1),
                "wickets":    round(pred_wickets, 1),
                "economy":    round(pred_economy, 1),
                "impact":     round(pred_impact,  1),
            },
            "featureImportance": importance_mapping,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ══════════════════════════════════════════════════════════════════════════════
# Cricket live-match routes
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/cricket')
def cricket_dashboard():
    return render_template('cricket.html')


@app.route('/api/cricket/live')
def cricket_live():
    """
    Main polling endpoint for the cricket dashboard.
    Returns live match state, pitch delivery data, and ML prediction.
    Triggers adaptive retraining whenever a new over completes.
    """
    try:
        state      = cricket_engine.get_live_state()
        pitch_data = cricket_engine.get_pitch_data()
        prediction = cricket_predictor.predict(state)

        # Online learning: add snapshot when over completes
        cricket_predictor.maybe_train_on_over(state)

        return jsonify({
            "match_state": state,
            "pitch_data":  pitch_data[-150:],   # cap payload
            "prediction":  prediction,
            "model_stats": cricket_predictor.stats(),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/cricket/history')
def cricket_history():
    """Return model training history (sample count, completed matches)."""
    try:
        return jsonify({
            "stats": cricket_predictor.stats(),
            "completed_matches": cricket_predictor.history.get("completed_matches", [])[-10:],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # Port 5001 avoids conflict with macOS AirPlay Receiver on 5000
    app.run(debug=True, host='0.0.0.0', port=5001)
