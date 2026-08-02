# Module H — API / Orchestration

**Purpose:** The integration point. Receives a patient query, calls Module G
for recommendations, calls Module E for explanations, assembles the combined
response. Implements the designed endpoints: `/recommend`, `/drugs/search`,
`/conditions`, `/ddi/check`, `/explain`, `/model/metrics`, `/health`.

**Depends on:** Modules D, E, G.
**Feeds into:** Module I (Presentation).
**Type:** Software Engineering — pure coordination, no learning.
