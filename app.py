from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow requests from frontend

# Load the trained models
models = joblib.load("maintenance_models.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No data provided"}), 400

        tire = data.get("tire_condition")
        ac = data.get("ac_condition")
        battery = data.get("hybrid_battery_condition")
        engine = data.get("engine_condition")

        if tire is None or ac is None or battery is None or engine is None:
            return jsonify({"error": "Missing data"}), 400

        features = [[tire, ac, battery, engine]]

        # Predict each maintenance type
        predictions = {
            "tire": models["tire_maintenance"].predict(features)[0],
            "ac": models["ac_maintenance"].predict(features)[0],
            "battery": models["battery_maintenance"].predict(features)[0],
            "engine": models["engine_maintenance"].predict(features)[0],
        }

        # Convert to human-readable messages
        messages = {
            "tire": "Tire needs to be replaced" if predictions["tire"] else "Tire is in good condition",
            "ac": "A/C needs maintenance" if predictions["ac"] else "A/C is in good condition",
            "battery": "Battery needs to be replaced" if predictions["battery"] else "Battery is in good condition",
            "engine": "Engine needs maintenance" if predictions["engine"] else "Engine is in good condition",
        }

        # Calculate overall health score 
        good_parts = sum(1 for key in predictions if predictions[key] == 0)
        overall_health = good_parts * 25

        messages["overall"] = overall_health

        return jsonify(messages)

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5001, debug=True)
