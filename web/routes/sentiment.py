from flask import Blueprint, render_template, request, session
from scripts.helpers import get_dynamic_alpha
from scripts.backend import load_backend
from scripts.sentiment_recommendation import (
    detect_emotions_multi_label,
    explain_emotion_recommendation,
    recommend_movies_multi_emotion,
    get_genre_weights_for_emotions,
)
from web.routes.utils import to_recommendation_dicts, combine_errors, get_lang

sentiment_bp = Blueprint("sentiment", __name__)
ratings_df, movies_df, algo, similarity_df = load_backend()


@sentiment_bp.route("/sentiment", methods=["GET", "POST"])
def sentiment():
    lang = get_lang(request, session)
    explanation = None
    recommendations = None
    error = None
    user_input = ""
    interpreted_sentiment = None
    user_id = None
    n = 4  # default
    if request.method == "POST":
        user_input = request.form.get("user_input", "")
        n = int(request.form.get("n", 4))
        user_id_raw = request.form.get("user_id", "")
        try:
            emotion_weights, clarification, translation_error = (
                detect_emotions_multi_label(user_input)
            )
            interpreted_sentiment = emotion_weights
            explanation = explain_emotion_recommendation(emotion_weights)
            error = combine_errors(
                (
                    translation_error
                    if lang == "en"
                    else (
                        f"Aviso de tradução: {translation_error}"
                        if translation_error
                        else None
                    )
                ),
                clarification if lang == "en" else clarification,
            )
            if user_id_raw.strip():
                try:
                    user_id = int(user_id_raw)
                    if user_id in ratings_df["UserID"].values:
                        from scripts.hybrid_recommendation_system import (
                            calculate_hybrid_scores,
                        )

                        user_ratings = ratings_df[ratings_df["UserID"] == user_id]
                        top_n = calculate_hybrid_scores(
                            algo,
                            user_ratings,
                            similarity_df,
                            alpha_func=get_dynamic_alpha,
                        )
                        top_n.sort(key=lambda x: x[2], reverse=True)
                        sorted_emotions = sorted(
                            emotion_weights.items(), key=lambda x: x[1], reverse=True
                        )
                        top_emotions = [e for e, w in sorted_emotions if w > 0][:2]
                        genre_weights = get_genre_weights_for_emotions(top_emotions)
                        genre_set = set(genre_weights.keys())
                        filtered = []
                        for _, movie_id, score in top_n:
                            movie = movies_df[movies_df["MovieID"] == movie_id].iloc[0]
                            movie_genres = set(str(movie["Genres"]).split("|"))
                            if genre_set & movie_genres:
                                filtered.append(
                                    {"Title": movie["Title"], "Genres": movie["Genres"]}
                                )
                            if len(filtered) >= n:
                                break
                        if len(filtered) < n:
                            extra = recommend_movies_multi_emotion(
                                movies_df, emotion_weights, n=n - len(filtered)
                            )
                            for _, row in extra.iterrows():
                                filtered.append(
                                    {"Title": row["Title"], "Genres": row["Genres"]}
                                )
                        recommendations = filtered[:n]
                    else:
                        error = combine_errors(
                            error,
                            (
                                "User ID not found. Showing recommendations based only on your mood."
                                if lang == "en"
                                else "ID de usuário não encontrado. Mostrando recomendações apenas pelo seu humor."
                            ),
                        )
                        recs = recommend_movies_multi_emotion(
                            movies_df, emotion_weights, n=n
                        )
                        recommendations = [
                            {"Title": row["Title"], "Genres": row["Genres"]}
                            for _, row in recs.iterrows()
                        ]
                except Exception:
                    error = combine_errors(
                        error,
                        (
                            "Invalid User ID. Showing recommendations based only on your mood."
                            if lang == "en"
                            else "ID de usuário inválido. Mostrando recomendações apenas pelo seu humor."
                        ),
                    )
                    recs = recommend_movies_multi_emotion(
                        movies_df, emotion_weights, n=n
                    )
                    recommendations = [
                        {"Title": row["Title"], "Genres": row["Genres"]}
                        for _, row in recs.iterrows()
                    ]
            else:
                recs = recommend_movies_multi_emotion(movies_df, emotion_weights, n=n)
                recommendations = [
                    {"Title": row["Title"], "Genres": row["Genres"]}
                    for _, row in recs.iterrows()
                ]
        except Exception as e:
            error = combine_errors(
                error, (str(e) if lang == "en" else f"Erro: {str(e)}")
            )
    recommendations = to_recommendation_dicts(recommendations)
    return render_template(
        "sentiment.html",
        user_input=user_input,
        explanation=explanation,
        recommendations=recommendations,
        interpreted_sentiment=interpreted_sentiment,
        error=error,
        lang=lang,
        n=n,
        user_id=user_id,
        from_hybrid=False,
    )
