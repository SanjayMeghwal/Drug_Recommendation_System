# Module G — Recommendation (our academic improvement)

**Purpose:** Given a patient's condition and current medications, find
condition-relevant candidate drugs (via Module B), score each against the
patient's current medications using Module D, and return a ranked,
safety-adjusted list.

**Depends on:** Module B (drug/condition data), Module D (interaction
predictions).
**Feeds into:** Module H (Orchestration).
**Type:** Software Engineering that orchestrates ML output — ranking logic
is a deterministic scoring formula, not a second trained model.
