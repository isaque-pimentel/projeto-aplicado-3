import os
import sys
from flask import Flask, render_template, request, redirect, url_for, session

# Add the project root directory to PYTHONPATH
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_DIR)

from scripts.backend import load_backend
from scripts.sentiment_recommendation import (
    detect_emotions_multi_label,
    explain_emotion_recommendation,
    recommend_movies_multi_emotion,
)
from scripts.hybrid_recommendation_system import (
    calculate_content_similarity,
)
from scripts.helpers import (
    get_dynamic_alpha,
    print_table,
    load_similarity_matrix,
    save_similarity_matrix,
    evaluate_recommendations,
)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'histflix-secret')

# Load backend data
ratings_df, movies_df, algo, similarity_df = load_backend()

@app.route("/", methods=["GET"])
def home():
    lang = request.args.get("lang") or session.get("lang", "pt-br")
    session["lang"] = lang
    return render_template("index.html", lang=lang)

@app.route("/recommend", methods=["POST"])
def recommend():
    lang = request.form.get("lang") or request.args.get("lang") or session.get("lang", "pt-br")
    session["lang"] = lang
    from_sentiment = request.form.get("from_sentiment", "false") == "true"
    try:
        user_id = int(request.form["user_id"])
        n = int(request.form.get("n", 10))
        alpha = float(request.form.get("alpha", 0.7))
        # Cold-start: user_id==0 ou não existe no dataset
        if user_id == 0 or user_id not in ratings_df["UserID"].values:
            # Usa a mesma lógica de show_cold_start_recommendations
            movie_counts = ratings_df["MovieID"].value_counts().head(n)
            recommendations = movies_df[movies_df["MovieID"].isin(movie_counts.index)].copy()
            recommendations["NumRatings"] = recommendations["MovieID"].map(movie_counts)
            recommendations = recommendations.sort_values(by="NumRatings", ascending=False)
            # Adiciona coluna HybridScore=None para compatibilidade com o template
            recommendations["HybridScore"] = None
            recommendations = recommendations[["MovieID", "Title", "Genres", "NumRatings", "HybridScore"]]
            # Converte para lista de dicionários para o template
            recommendations = recommendations.to_dict(orient="records")
        else:
            from scripts.hybrid_recommendation_system import calculate_hybrid_scores
            user_ratings = ratings_df[ratings_df["UserID"] == user_id]
            top_n = calculate_hybrid_scores(algo, user_ratings, similarity_df, alpha_func=get_dynamic_alpha)
            top_n.sort(key=lambda x: x[2], reverse=True)
            top_n = top_n[:n]
            recommendations = []
            for _, movie_id, score in top_n:
                movie_details = movies_df[movies_df["MovieID"] == movie_id].iloc[0].to_dict()
                recommendations.append({
                    "MovieID": movie_id,
                    "Title": movie_details["Title"],
                    "Genres": movie_details["Genres"],
                    "HybridScore": score,
                })
        return render_template("recommendations.html", recommendations=recommendations, lang=lang, from_sentiment=from_sentiment)
    except Exception as e:
        error_msg = ("Invalid User ID. Please enter a valid number." if lang == "en" else "ID de usuário inválido. Por favor, insira um número válido.")
        return render_template("error.html", error=error_msg, lang=lang)

@app.route("/sentiment", methods=["GET", "POST"])
def sentiment():
    lang = request.form.get("lang") or request.args.get("lang") or session.get("lang", "pt-br")
    session["lang"] = lang
    explanation = None
    recommendations = None
    error = None
    user_input = ""
    if request.method == "POST":
        user_input = request.form.get("user_input", "")
        n = int(request.form.get("n", 10))
        try:
            emotion_weights, clarification, translation_error = detect_emotions_multi_label(user_input)
            explanation = explain_emotion_recommendation(emotion_weights)
            recommendations = recommend_movies_multi_emotion(movies_df, emotion_weights, n=n)
        except Exception as e:
            error = str(e) if lang == "en" else f"Erro: {str(e)}"
    return render_template(
        "sentiment.html",
        user_input=user_input,
        explanation=explanation,
        recommendations=recommendations,
        error=error,
        lang=lang,
        from_hybrid=False
    )

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

if __name__ == "__main__":
    app.run(debug=True)