import os
import sys
from flask import Flask, render_template, request, redirect, url_for, session

# Add the project root directory to PYTHONPATH
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_DIR)

from scripts.backend import load_backend
from web.config import Config
from web.routes.recommendation import recommendation_bp
from web.routes.sentiment import sentiment_bp
from web.routes.utils import get_lang


# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)

# Load backend data
ratings_df, movies_df, algo, similarity_df = load_backend()

# Register blueprints
app.register_blueprint(recommendation_bp)
app.register_blueprint(sentiment_bp)


@app.route("/", methods=["GET"])
def home():
    lang = get_lang(request, session)
    return render_template("index.html", lang=lang)


@app.route("/to_hybrid")
def to_hybrid():
    lang = session.get("lang", "pt-br")
    return redirect(url_for("hybrid", lang=lang))


@app.route("/to_sentiment")
def to_sentiment():
    lang = session.get("lang", "pt-br")
    return redirect(url_for("sentiment", lang=lang))


@app.route("/hybrid", methods=["GET"])
def hybrid():
    lang = request.args.get("lang") or session.get("lang", "pt-br")
    session["lang"] = lang
    return render_template("hybrid.html", lang=lang)


@app.errorhandler(Exception)
def handle_exception(e):
    lang = session.get("lang", "pt-br")
    error_msg = str(e) if lang == "en" else f"Erro: {str(e)}"
    return render_template("error.html", error=error_msg, lang=lang), 500


@app.before_request
def ensure_lang():
    get_lang(request, session)


if __name__ == "__main__":
    app.run(debug=True)
