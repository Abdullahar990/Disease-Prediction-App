from flask import Flask, render_template, request, jsonify
import joblib
import json
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load("model_filtered.pkl")

# Load symptoms from JSON
with open("features.json") as f:
    symptom_list = json.load(f)

@app.route("/")
def index():
    print(symptom_list)  # Debugging: Check if symptoms are loaded
    return render_template("index.html", symptom_list=symptom_list)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    selected_symptoms = data["symptoms"]

    # Convert symptoms to binary input vector
    input_vector = [1 if symptom in selected_symptoms else 0 for symptom in symptom_list]
    prediction = model.predict([input_vector])[0]

    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(debug=True)
