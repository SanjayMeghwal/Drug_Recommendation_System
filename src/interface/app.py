"""Demo interface (Module I).

Calls the orchestration layer directly rather than over HTTP, so the demo
runs from a single command. The API is a separate deliverable, exercised by
its own tests and browsable at /docs when uvicorn is running.

Inputs are dropdowns rather than free text. Only nine conditions are
supported and drug names must match exactly, so typing invites a failure the
user cannot diagnose — picking from the known set makes an invalid request
impossible to express.
"""

import json

import streamlit as st

from src.api.main import orchestrate_recommendation
from src.config import load_config, resolve_path
from src.evaluation.run import REPORT_NAME
from src.explainability.explain import explain_interaction
from src.models.predict import UnknownDrugError, get_service
from src.recommendation.recommend import get_recommendation_service

RISK_STYLES = {
    "low": ("🟢", "Low interaction risk"),
    "moderate": ("🟡", "Moderate interaction risk — use with caution"),
    "high": ("🔴", "High interaction risk"),
}


@st.cache_resource
def load_services():
    """Load the model and data once, not on every interaction.

    Streamlit re-runs this script top to bottom on every widget change, so
    without caching each click would rebuild the graph and model.
    """
    predictor = get_service()
    recommender = get_recommendation_service()
    drug_names = sorted(predictor.name_for(drug_id) for drug_id in predictor.graph.drug_ids)
    named_drugs = [name for name in drug_names if not name.startswith("Unknown (")]
    conditions = [entry["condition"] for entry in recommender.available_conditions()]
    return predictor, recommender, named_drugs, conditions


@st.cache_data
def load_metrics() -> dict | None:
    config = load_config()
    report_path = resolve_path(config["paths"]["results_dir"]) / REPORT_NAME
    if not report_path.exists():
        return None
    with open(report_path, encoding="utf-8") as handle:
        return json.load(handle)


st.set_page_config(page_title="Explainable Drug Recommendation", page_icon="💊", layout="wide")

st.title("💊 Explainable Drug Recommendation & DDI Prediction")
st.caption(
    "Graph neural network predicting drug-drug interactions, with personalized "
    "recommendations ranked by safety."
)

st.warning(
    "**Academic prototype — not for clinical use.** Predictions come from a model "
    "trained on public data and are not verified medical facts. Nothing here should "
    "inform real prescribing decisions.",
    icon="⚠️",
)

try:
    predictor, recommender, drug_names, conditions = load_services()
except FileNotFoundError as error:
    st.error(f"The system is not ready: {error}")
    st.stop()

recommend_tab, interaction_tab, performance_tab = st.tabs(
    ["Recommend a drug", "Check an interaction", "Model performance"]
)


# --- Recommend -------------------------------------------------------------

with recommend_tab:
    st.subheader("Find a drug that is safe alongside current medications")

    left, right = st.columns(2)
    with left:
        condition = st.selectbox("Condition", conditions, index=conditions.index("hypertension"))
    with right:
        current_medications = st.multiselect(
            "Current medications",
            drug_names,
            default=["Warfarin"],
            help="Type to search. Leave empty to rank by overall interaction burden.",
        )

    if st.button("Get recommendations", type="primary"):
        result = orchestrate_recommendation(condition, current_medications)

        if result["unrecognised_medications"]:
            # Surfaced loudly: an unchecked interaction must never be mistaken
            # for one that was checked and found safe.
            st.error(
                "Could not identify: "
                + ", ".join(result["unrecognised_medications"])
                + ". These were **not** checked for interactions."
            )

        a, b, c = st.columns(3)
        a.metric("Candidates considered", result["candidates_considered"])
        b.metric("Recommended", len(result["recommended"]))
        c.metric("Excluded for risk", len(result["warnings"]))

        if not result["recommended"]:
            st.error(
                f"**No safe option found.** All {result['candidates_considered']} candidate "
                f"drugs for {result['condition']} carry a high predicted interaction risk "
                "with the selected medications. See the excluded list below."
            )
        else:
            st.success(f"{len(result['recommended'])} option(s) ranked safest first.")

        for item in result["recommended"]:
            icon, risk_label = RISK_STYLES[item["risk_level"]]
            with st.container(border=True):
                headline, score_column = st.columns([3, 1])
                headline.markdown(f"### {icon} {item['drug']}")
                headline.caption(f"{risk_label} · acts on {item['target_gene']}")
                score_column.metric("Safety score", f"{item['score']:.2f}")

                if item["riskiest_medication"]:
                    st.markdown(
                        f"Worst interaction: **{item['max_interaction_risk']:.0%}** "
                        f"probability with **{item['riskiest_medication']}**."
                    )
                with st.expander("Why this result?"):
                    st.write(item["explanation"])

        if result["warnings"]:
            with st.expander(f"Excluded for high interaction risk ({len(result['warnings'])})"):
                st.caption(
                    "These treat the condition but are predicted to interact strongly "
                    "with the current medications."
                )
                for item in result["warnings"]:
                    st.markdown(
                        f"🔴 **{item['drug']}** — {item['max_interaction_risk']:.0%} "
                        f"probability with {item['riskiest_medication']}"
                    )
                    st.caption(item["explanation"])


# --- Check an interaction --------------------------------------------------

with interaction_tab:
    st.subheader("Check whether two drugs interact")

    first, second = st.columns(2)
    with first:
        drug_a = st.selectbox("First drug", drug_names, index=drug_names.index("Warfarin"))
    with second:
        drug_b = st.selectbox("Second drug", drug_names, index=drug_names.index("Ibuprofen"))

    if st.button("Check interaction", type="primary"):
        if drug_a == drug_b:
            st.warning("Select two different drugs.")
        else:
            try:
                explanation = explain_interaction(drug_a, drug_b)
            except UnknownDrugError as error:
                st.error(str(error))
            else:
                probability = explanation.probability
                if probability >= 0.8:
                    st.error(f"High interaction risk — {probability:.0%} probability")
                elif probability >= 0.5:
                    st.warning(f"Possible interaction — {probability:.0%} probability")
                else:
                    st.success(f"Interaction unlikely — {probability:.0%} probability")

                st.write(explanation.summary)

                a, b, c = st.columns(3)
                a.metric("Shared interaction partners", explanation.shared_partners)
                b.metric(f"{explanation.drug_a} interactions", explanation.drug_a_interactions)
                c.metric(f"{explanation.drug_b} interactions", explanation.drug_b_interactions)

                if explanation.shared_partner_names:
                    st.caption(
                        "Most-connected shared partners: "
                        + ", ".join(explanation.shared_partner_names)
                    )


# --- Model performance -----------------------------------------------------

with performance_tab:
    st.subheader("Measured performance on held-out test data")

    metrics = load_metrics()
    if metrics is None:
        st.info("No evaluation report yet. Run `python -m src.evaluation.run`.")
    else:
        test = metrics["results"]["test"]

        row = st.columns(4)
        row[0].metric("AUC", f"{test['auc']:.4f}")
        row[1].metric("AUPR", f"{test['aupr']:.4f}")
        row[2].metric("Accuracy", f"{test['accuracy']:.4f}")
        row[3].metric("F1", f"{test['f1']:.4f}")

        st.markdown("#### Against the baseline paper")
        st.dataframe(
            [
                {
                    "Metric": name.capitalize(),
                    "Ours": f"{values['ours']:.4f}",
                    "Baseline paper": f"{values['baseline_paper']:.4f}",
                    "Difference": f"{values['difference']:+.4f}",
                }
                for name, values in metrics["baseline_comparison"].items()
            ],
            hide_index=True,
            width="stretch",
        )

        st.info(metrics["comparison_caveat"], icon="ℹ️")

        st.caption(
            f"Evaluated on {test['num_examples']:,} held-out drug pairs at a decision "
            f"threshold of {test['threshold']:.2f}, tuned on the validation split."
        )
