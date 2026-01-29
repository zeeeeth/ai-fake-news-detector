from __future__ import annotations

import os
from flask import Flask, jsonify

from src.api.routes import init_routes
from src.inference.predict import Predictor


def create_app(
    article_model_dir: str = "models/article_roberta",
    claim_model_dir: str = "models/claim_roberta",
) -> Flask:
    app = Flask(__name__)

    predictor = Predictor(
        article_model_dir=article_model_dir,
        claim_model_dir=claim_model_dir,
    )

    bp = init_routes(predictor)
    app.register_blueprint(bp, url_prefix="/api")

    @app.errorhandler(404)
    def not_found(_):
        return jsonify({"error": "not found"}), 404

    return app


if __name__ == "__main__":
    app = create_app()
    print(app.url_map)
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
