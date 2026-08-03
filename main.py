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
        current_medications=["Warfarin", "Ibuprofen"],
    )

    print(f"Condition: {result['condition']}")
    print(f"Candidates considered: {result['candidates_considered']}\n")

    print(f"Recommended ({len(result['recommended'])}):")
    for item in result["recommended"]:
        print(f"  - {item['drug']} (score {item['score']:.2f}, {item['risk_level']} risk)")
        print(f"      {item['explanation']}")

    print(f"\nExcluded for interaction risk ({len(result['warnings'])}):")
    for item in result["warnings"][:5]:
        print(
            f"  - {item['drug']} (risk {item['max_interaction_risk']:.2f} "
            f"vs {item['riskiest_medication']})"
        )


if __name__ == "__main__":
    main()
