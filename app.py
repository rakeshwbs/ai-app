from flask import Flask, request, jsonify
import joblib, os, json

app = Flask(__name__)

# If you exported to JSON (lighter, no sklearn):
with open("mpg_regression_params.json") as f:
    P = json.load(f)
COEF = float(P["coef"][0])
INTERCEPT = float(P["intercept"])

@app.route("/")
def health():
    return jsonify({"status": "ok", "model": "mpg linear regression"})

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        w = request.args.get("weight")
    else:
        data = request.get_json(silent=True) or {}
        w = data.get("weight")

    if w is None:
        return jsonify({"error": "Provide weight"}), 400
    try:
        weight = float(w)
    except ValueError:
        return jsonify({"error": "weight must be numeric"}), 400

    mpg = COEF * weight + INTERCEPT
    return jsonify({"weight": weight, "predicted_mpg": round(mpg, 2)})
