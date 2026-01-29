from __future__ import annotations

from flask import Blueprint, jsonify, request

from src.inference.predict import Predictor

bp = Blueprint("api", __name__)
predictor: Predictor | None = None


def init_routes(p: Predictor) -> Blueprint:
    global predictor
    predictor = p
    return bp


@bp.get("/health")
def health():
    return jsonify({"status": "ok"})


@bp.get("/models")
def models():
    assert predictor is not None
    return jsonify(predictor.info())


@bp.post("/predict/article")
def predict_article():
    assert predictor is not None

    payload = request.get_json(silent=True) or {}
    text = payload.get("article", "")
    print("Received article for prediction: ", text)
    try:
        result = predictor.predict_article(text=text, max_length=int(payload.get("max_length", 512)))
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@bp.post("/predict/claim")
def predict_claim():
    assert predictor is not None

    payload = request.get_json(silent=True) or {}
    statement = payload.get("claim", "")

    try:
        result = predictor.predict_claim(statement=statement, max_length=int(payload.get("max_length", 128)))
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
