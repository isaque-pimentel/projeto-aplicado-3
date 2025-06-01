def to_recommendation_dicts(recommendations):
    """
    Helper to convert a DataFrame or list of dicts to a list of dicts for template rendering.
    """
    if recommendations is None:
        return []
    if isinstance(recommendations, list):
        return recommendations
    try:
        return recommendations.to_dict(orient="records")
    except Exception:
        return []


def combine_errors(*msgs):
    """
    Helper to combine multiple error/clarification messages into a single string, skipping None/empty.
    """
    return " ".join([str(m) for m in msgs if m])


def get_lang(request, session):
    lang = (
        request.form.get("lang")
        or request.args.get("lang")
        or session.get("lang", "pt-br")
    )
    session["lang"] = lang
    return lang
