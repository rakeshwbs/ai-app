from flask import Flask, request, jsonify
import json, os

app = Flask(__name__)

# If you are using the JSON-coefficients approach:
BASE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE, "mpg_regression_params.json")) as f:
    P = json.load(f)
COEF = float(P["coef"][0])
INTERCEPT = float(P["intercept"])

def extract_weight(req: request):
    # Priority: JSON body > form > query param
    data = req.get_json(silent=True) or {}
    if "weight" in data:
        return data["weight"]
    if "weight" in req.form:
        return req.form["weight"]
    return req.args.get("weight")

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "linear_regression_json"})

@app.route("/predict", methods=["GET", "POST"])
def predict():
    w = extract_weight(request)
    if w is None:
        return jsonify({"error": "Provide 'weight' via JSON body, form data, or query parameter"}), 400
    try:
        weight = float(w)
    except (TypeError, ValueError):
        return jsonify({"error": "'weight' must be numeric"}), 400
    mpg = COEF * weight + INTERCEPT
    return jsonify({"weight": weight, "predicted_mpg": round(float(mpg), 2)})

@app.route("/openapi.json", methods=["GET"])
def openapi():
    # Minimal spec for quick Postman import
    spec = {
        "openapi": "3.0.0",
        "info": {"title": "MPG API", "version": "1.0.0"},
        "paths": {
            "/predict": {
                "get": {
                    "parameters": [{
                        "in": "query", "name": "weight", "schema": {"type": "number"}, "required": True
                    }],
                    "responses": {"200": {"description": "OK"}}
                },
                "post": {
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"type": "object", "properties": {"weight": {"type": "number"}}, "required": ["weight"]}
                            }
                        }
                    },
                    "responses": {"200": {"description": "OK"}}
                }
            }
        }
    }
    return jsonify(spec)
