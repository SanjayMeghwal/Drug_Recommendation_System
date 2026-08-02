"""Personalized recommendation interface (Module G) — our academic
improvement. Stub until Day 8, when this ranks real condition-relevant
candidates using Module D's predictions.
"""


def recommend(condition: str, current_medications: list[str]) -> list[dict]:
    """Return a ranked list of recommended drugs for the given condition.

    STUB: returns a fixed candidate list regardless of input. Real
    implementation looks up condition-relevant drugs (Module B) and scores
    them against current_medications using Module D.
    """
    return [
        {"drug": "Metformin", "score": 0.82},
        {"drug": "Losartan", "score": 0.75},
    ]
