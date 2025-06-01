import os
import sys
from flask import Flask, render_template, request

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

# Load backend data
ratings_df, movies_df, algo, similarity_df = load_backend()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        user_id = int(request.form["user_id"])
        n = int(request.form.get("n", 10))
        alpha = float(request.form.get("alpha", 0.7))

        # Get user ratings
        user_ratings = ratings_df[ratings_df["UserID"] == user_id]
        if user_ratings.empty:
            # Cold-start: show top-n popular movies
            movie_counts = ratings_df["MovieID"].value_counts().head(n)
            recommendations = movies_df[movies_df["MovieID"].isin(movie_counts.index)].copy()
            recommendations["NumRatings"] = recommendations["MovieID"].map(movie_counts)
            recommendations = recommendations.sort_values(by="NumRatings", ascending=False)
            recommendations = recommendations[["MovieID", "Title", "Genres", "NumRatings"]]
        else:
            # Hybrid recommendation (dynamic alpha)
            # Use the same logic as in test_hybrid_recommendation.py
            from scripts.hybrid_recommendation_system import calculate_hybrid_scores
            top_n = calculate_hybrid_scores(algo, user_ratings, similarity_df, alpha_func=get_dynamic_alpha)
            top_n.sort(key=lambda x: x[2], reverse=True)
            top_n = top_n[:n]
            recommendations = []
            for _, movie_id, score in top_n:
                movie_details = movies_df[movies_df["MovieID"] == movie_id].iloc[0].to_dict()
                recommendations.append(
                    {
                        "MovieID": movie_id,
                        "Title": movie_details["Title"],
                        "Genres": movie_details["Genres"],
                        "HybridScore": score,
                    }
                )
        return render_template("recommendations.html", recommendations=recommendations)

    except Exception as e:
        return render_template("error.html", error=str(e))

@app.route("/sentiment", methods=["GET", "POST"])
def sentiment():
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
            # For web: skip manual adjustment, but could add a UI for it
            recommendations = recommend_movies_multi_emotion(movies_df, emotion_weights, n=n)
        except Exception as e:
            error = str(e)
    return render_template(
        "sentiment.html",
        user_input=user_input,
        explanation=explanation,
        recommendations=recommendations,
        error=error,
    )

if __name__ == "__main__":
    app.run(debug=True)