from flask import Flask, request, render_template
import os
import sqlite3
import pandas as pd
from helpers import load_model
from hybrid_recommendation_system import (
    calculate_content_similarity,
    calculate_hybrid_scores,
)

# Initialize Flask app
app = Flask(__name__)

# Load data and model
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_DIR, "dataset", "sqlite", "movielens_1m.db")
MODEL_PATH = os.path.join(PROJECT_DIR, "models", "svd_model_movielens_1m.pkl")
print("Loading model from:", MODEL_PATH)
print("Loading database from:", DB_PATH)


# Load ratings and movies data
def load_data():
    conn = sqlite3.connect(DB_PATH)
    ratings_df = pd.read_sql("SELECT UserID, MovieID, Rating FROM ratings", conn)
    movies_df = pd.read_sql("SELECT MovieID, Title, Genres FROM movies", conn)
    conn.close()
    return ratings_df, movies_df


ratings_df, movies_df = load_data()
algo = load_model(MODEL_PATH)

# Precompute content-based similarity
similarity_method = "tfidf"
similarity_df = calculate_content_similarity(movies_df, method=similarity_method)


# Route for the home page
@app.route("/")
def home():
    lang = request.args.get("lang", "en")
    return render_template("index.html", lang=lang)


# Route for generating recommendations
@app.route("/recommend", methods=["POST"])
def recommend():
    lang = request.args.get("lang", "en")
    try:
        user_id = int(request.form["user_id"])
        n = int(request.form.get("n", 10))  # Default to top 10 recommendations
        alpha = float(request.form.get("alpha", 0.7))  # Default alpha value

        # Get user ratings
        user_ratings = ratings_df[ratings_df["UserID"] == user_id]
        if user_ratings.empty:
            # Fallback to content-based recommendations for new users
            avg_similarity = similarity_df.mean(axis=1).sort_values(ascending=False)
            top_n_movies = avg_similarity.head(n).index
            recommendations = movies_df[movies_df["MovieID"].isin(top_n_movies)].copy()
            recommendations["AvgSimilarity"] = recommendations["MovieID"].map(
                avg_similarity
            )
            recommendations = recommendations.sort_values(
                by="AvgSimilarity", ascending=False
            )
        else:
            # Generate hybrid recommendations
            hybrid_scores = calculate_hybrid_scores(
                algo, user_ratings, similarity_df, alpha
            )
            hybrid_scores.sort(key=lambda x: x[2], reverse=True)
            top_n_scores = hybrid_scores[:n]

            # Get movie details
            recommendations = []
            for _, movie_id, hybrid_score in top_n_scores:
                movie_details = (
                    movies_df[movies_df["MovieID"] == movie_id].iloc[0].to_dict()
                )
                recommendations.append(
                    {
                        "MovieID": movie_id,
                        "Title": movie_details["Title"],
                        "Genres": movie_details["Genres"],
                        "HybridScore": hybrid_score,
                    }
                )
            recommendations = pd.DataFrame(recommendations)

        # Render recommendations
        return render_template(
            "recommendations.html",
            recommendations=recommendations.to_dict(orient="records"),
            lang=lang,
        )

    except Exception as e:
        return render_template("error.html", error=str(e))


# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
