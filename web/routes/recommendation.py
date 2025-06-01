from flask import Blueprint, render_template, request, session
from scripts.helpers import get_dynamic_alpha
from scripts.backend import load_backend
from web.routes.utils import to_recommendation_dicts, get_lang

recommendation_bp = Blueprint("recommendation", __name__)
ratings_df, movies_df, algo, similarity_df = load_backend()


@recommendation_bp.route("/recommend", methods=["POST"])
def recommend():
    lang = get_lang(request, session)
    from_sentiment = request.form.get("from_sentiment", "false") == "true"
    try:
        user_id = int(request.form.get("user_id", 0))
        n = int(request.form.get("n", 10))
        alpha = float(request.form.get("alpha", 0.7))
        if user_id == 0 or user_id not in ratings_df["UserID"].values:
            movie_counts = ratings_df["MovieID"].value_counts().head(n)
            recommendations = movies_df[
                movies_df["MovieID"].isin(movie_counts.index)
            ].copy()
            recommendations["NumRatings"] = recommendations["MovieID"].map(movie_counts)
            recommendations = recommendations.sort_values(
                by="NumRatings", ascending=False
            )
            recommendations["HybridScore"] = None
            recommendations = recommendations[
                ["MovieID", "Title", "Genres", "NumRatings", "HybridScore"]
            ]
        else:
            from scripts.hybrid_recommendation_system import calculate_hybrid_scores

            user_ratings = ratings_df[ratings_df["UserID"] == user_id]
            top_n = calculate_hybrid_scores(
                algo, user_ratings, similarity_df, alpha_func=get_dynamic_alpha
            )
            top_n.sort(key=lambda x: x[2], reverse=True)
            top_n = top_n[:n]
            recs = []
            for _, movie_id, score in top_n:
                movie_details = (
                    movies_df[movies_df["MovieID"] == movie_id].iloc[0].to_dict()
                )
                recs.append(
                    {
                        "MovieID": movie_id,
                        "Title": movie_details["Title"],
                        "Genres": movie_details["Genres"],
                        "HybridScore": score,
                    }
                )
            recommendations = recs
        recommendations = to_recommendation_dicts(recommendations)
        return render_template(
            "recommendations.html",
            recommendations=recommendations,
            lang=lang,
            from_sentiment=from_sentiment,
        )
    except Exception as e:
        error_msg = str(e) if lang == "en" else f"Erro: {str(e)}"
        return render_template("error.html", error=error_msg, lang=lang)
