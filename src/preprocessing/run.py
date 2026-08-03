"""Entry point for Module B (Preprocessing).

Cleans the raw CSVs in data/raw/ into two schema-fixed tables in
data/processed/: drugs.csv and ddi_pairs.csv. See docs/data_acquisition.md
for the raw file schemas.
"""

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def build_drugs_table(smiles_df: pd.DataFrame, attributes_df: pd.DataFrame) -> pd.DataFrame:
    """Combine drug_smiles.csv (authoritative drug list + SMILES) with
    file_drugs.csv (name, drug group) into one row per drug.

    file_drugs.csv has multiple rows per drug (one per target/interaction),
    so it's deduplicated by ID first. Not every drug in the SMILES list has
    a matching entry in file_drugs.csv — those get a placeholder name
    rather than being dropped, since they're still valid graph nodes.
    """
    smiles_df = smiles_df[["drug_id", "smiles"]].drop_duplicates(subset="drug_id")

    attrs = attributes_df.rename(
        columns={"DrugBank ID": "drug_id", "Name": "name", "Drug Groups": "drug_groups"}
    )
    attrs = attrs[["drug_id", "name", "drug_groups"]].drop_duplicates(subset="drug_id")

    drugs = smiles_df.merge(attrs, on="drug_id", how="left")
    drugs["name"] = drugs["name"].fillna("Unknown (" + drugs["drug_id"] + ")")
    drugs["drug_groups"] = drugs["drug_groups"].fillna("unknown")

    return drugs.reset_index(drop=True)


def build_ddi_pairs_table(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    known_drug_ids: set,
) -> pd.DataFrame:
    """Combine the three raw DDI files and use the `split` COLUMN as the
    authoritative train/val/test assignment. The raw filenames do NOT
    correspond to a clean partition — each file contains a mix of rows
    tagged training/validation/testing in the split column, so the
    filename itself must be ignored.
    """
    columns = ["d1", "d2", "type", "split"]
    combined = pd.concat(
        [train_df[columns], val_df[columns], test_df[columns]],
        ignore_index=True,
    )

    combined = combined.drop_duplicates(subset=["d1", "d2", "type"])
    combined = combined[combined["d1"].isin(known_drug_ids) & combined["d2"].isin(known_drug_ids)]

    return combined.reset_index(drop=True)


def run() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading raw files...")
    smiles_df = pd.read_csv(RAW_DIR / "drug_smiles.csv")
    attributes_df = pd.read_csv(
        RAW_DIR / "file_drugs.csv", usecols=["DrugBank ID", "Name", "Drug Groups"]
    )
    train_df = pd.read_csv(RAW_DIR / "ddi_training.csv", usecols=["d1", "d2", "type", "split"])
    val_df = pd.read_csv(RAW_DIR / "ddi_validation.csv", usecols=["d1", "d2", "type", "split"])
    test_df = pd.read_csv(RAW_DIR / "ddi_test.csv", usecols=["d1", "d2", "type", "split"])

    print("Building drugs table...")
    drugs = build_drugs_table(smiles_df, attributes_df)
    missing_names = (drugs["drug_groups"] == "unknown").sum()
    print(f"  {len(drugs)} drugs ({missing_names} without a matched name/group)")

    print("Building ddi_pairs table...")
    ddi_pairs = build_ddi_pairs_table(
        train_df, val_df, test_df, known_drug_ids=set(drugs["drug_id"])
    )
    print(f"  {len(ddi_pairs)} pairs")
    print(f"  split counts:\n{ddi_pairs['split'].value_counts().to_string()}")

    drugs.to_csv(PROCESSED_DIR / "drugs.csv", index=False)
    ddi_pairs.to_csv(PROCESSED_DIR / "ddi_pairs.csv", index=False)
    print(f"Saved to {PROCESSED_DIR}")


if __name__ == "__main__":
    run()
