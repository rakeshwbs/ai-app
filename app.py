from flask import Flask, request, jsonify
import json, os

app = Flask(__name__)
BASE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE, "mpg_regression_params.json")) as f:
    P = json.load(f)
COEF = float(P["coef"][0])      # one feature: weight
INTERCEPT = float(P["intercept"])

@app.route("/")
def health():
    return jsonify({"status": "ok", "model": "linear_regression_json"})

@app.route("/predict", methods=["GET", "POST"])
def predict():
    w = request.args.get("weight") if request.method == "GET" else (request.get_json(silent=True) or {}).get("weight")
    if w is None:
        return jsonify({"error": "Provide weight"}), 400
    try:
        weight = float(w)
    except ValueError:
        return jsonify({"error": "weight must be numeric"}), 400
    mpg = COEF * weight + INTERCEPT
    return jsonify({"weight": weight, "predicted_mpg": round(float(mpg), 2)})
