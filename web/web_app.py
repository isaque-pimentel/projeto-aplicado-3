import os
import sys
from flask import Flask, render_template, request

# Add the project root directory to PYTHONPATH
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_DIR)

from scripts import calculate_hybrid_scores, load_backend

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
            # Fallback to content-based recommendations
            avg_similarity = similarity_df.mean(axis=1).sort_values(ascending=False)
            top_n_movies = avg_similarity.head(n).index
            recommendations = movies_df[movies_df["MovieID"].isin(top_n_movies)].copy()
            recommendations["AvgSimilarity"] = recommendations["MovieID"].map(avg_similarity)
            recommendations = recommendations.sort_values(by="AvgSimilarity", ascending=False)
        else:
            # Generate hybrid recommendations
            hybrid_scores = calculate_hybrid_scores(algo, user_ratings, similarity_df, alpha)
            hybrid_scores.sort(key=lambda x: x[2], reverse=True)
            top_n_scores = hybrid_scores[:n]

            # Get movie details
            recommendations = []
            for _, movie_id, hybrid_score in top_n_scores:
                movie_details = movies_df[movies_df["MovieID"] == movie_id].iloc[0].to_dict()
                recommendations.append(
                    {
                        "MovieID": movie_id,
                        "Title": movie_details["Title"],
                        "Genres": movie_details["Genres"],
                        "HybridScore": hybrid_score,
                    }
                )
        return render_template("recommendations.html", recommendations=recommendations)

    except Exception as e:
        return render_template("error.html", error=str(e))

if __name__ == "__main__":
    app.run(debug=True)