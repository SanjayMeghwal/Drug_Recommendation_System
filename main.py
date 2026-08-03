"""Application entry point.

Run this directly for a quick smoke test of the wired (currently stubbed)
pipeline:
    python main.py

For the real API server:      python -m uvicorn src.api.main:app --reload
For the real demo interface:  python -m streamlit run src/interface/app.py
"""

from src.api.main import orchestrate_recommendation


def main() -> None:
    result = orchestrate_recommendation(
        condition="hypertension",
        current_medications=["Warfarin"],
    )
    print("Recommended:")
    for item in result["recommended"]:
        print(f"  - {item['drug']} (score={item['score']:.2f}): {item['explanation']}")
    print("Warnings:", result["warnings"] or "none")


if __name__ == "__main__":
    main()
