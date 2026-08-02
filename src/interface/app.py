"""Demo interface (Module I). Calls the orchestration logic directly — no
HTTP round-trip needed for the local demo.
"""

import streamlit as st

from src.api.main import orchestrate_recommendation

st.title("Explainable Drug Recommendation & DDI Prediction")

condition = st.text_input("Current condition")
current_meds_raw = st.text_input("Current medications (comma-separated)")

if st.button("Get Recommendation"):
    current_medications = [m.strip() for m in current_meds_raw.split(",") if m.strip()]
    result = orchestrate_recommendation(condition, current_medications)

    st.subheader("Recommended Drugs")
    for item in result["recommended"]:
        st.write(f"**{item['drug']}** — score {item['score']:.2f}")
        st.caption(item["explanation"])

    if result["warnings"]:
        st.subheader("Warnings")
        for warning in result["warnings"]:
            st.warning(f"{warning['drug']}: {warning['reason']}")
