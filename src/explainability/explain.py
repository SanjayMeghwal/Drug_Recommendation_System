"""Explanation interface (Module E). Stub until Day 7, when this wraps the
real trained model with SHAP.
"""


def explain(drug: str, score: float) -> str:
    """Return a human-readable explanation for a prediction/recommendation.

    STUB: returns a canned sentence. Real implementation runs SHAP against
    the trained model from Module D.
    """
    return (
        f"[stub explanation] {drug} scored {score:.2f} — SHAP-based reasoning not yet implemented."
    )
