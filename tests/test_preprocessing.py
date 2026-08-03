"""Unit tests for Module B (Preprocessing), using small synthetic
DataFrames rather than the full raw files — the transformation logic is
tested independently of any real data or network/disk I/O.
"""

import pandas as pd

from src.preprocessing.run import build_ddi_pairs_table, build_drugs_table


def test_build_drugs_table_merges_name_and_smiles():
    smiles_df = pd.DataFrame({"drug_id": ["DB001", "DB002"], "smiles": ["CCO", "CCC"]})
    attributes_df = pd.DataFrame(
        {
            "DrugBank ID": ["DB001", "DB001", "DB002"],  # DB001 repeated, as in the real file
            "Name": ["Ethanol", "Ethanol", "Propane"],
            "Drug Groups": ["approved", "approved", "experimental"],
        }
    )

    drugs = build_drugs_table(smiles_df, attributes_df)

    assert len(drugs) == 2
    assert set(drugs.columns) == {"drug_id", "smiles", "name", "drug_groups"}
    assert drugs.loc[drugs["drug_id"] == "DB001", "name"].iloc[0] == "Ethanol"


def test_build_drugs_table_fills_placeholder_for_unmatched_drug():
    smiles_df = pd.DataFrame({"drug_id": ["DB999"], "smiles": ["CCO"]})
    attributes_df = pd.DataFrame(
        {"DrugBank ID": ["DB001"], "Name": ["Ethanol"], "Drug Groups": ["approved"]}
    )

    drugs = build_drugs_table(smiles_df, attributes_df)

    assert drugs.loc[0, "name"] == "Unknown (DB999)"
    assert drugs.loc[0, "drug_groups"] == "unknown"


def test_build_ddi_pairs_table_uses_split_column_not_filename():
    # Simulate the real data shape: each "file" contains a mix of split values.
    train_df = pd.DataFrame({"d1": ["DB001"], "d2": ["DB002"], "type": [5], "split": ["testing"]})
    val_df = pd.DataFrame({"d1": ["DB002"], "d2": ["DB003"], "type": [6], "split": ["training"]})
    test_df = pd.DataFrame({"d1": ["DB001"], "d2": ["DB003"], "type": [7], "split": ["validation"]})
    known_ids = {"DB001", "DB002", "DB003"}

    pairs = build_ddi_pairs_table(train_df, val_df, test_df, known_ids)

    assert len(pairs) == 3
    assert set(pairs["split"]) == {"testing", "training", "validation"}


def test_build_ddi_pairs_table_drops_orphan_pairs():
    train_df = pd.DataFrame({"d1": ["DB001"], "d2": ["DB999"], "type": [5], "split": ["training"]})
    val_df = pd.DataFrame(columns=["d1", "d2", "type", "split"])
    test_df = pd.DataFrame(columns=["d1", "d2", "type", "split"])
    known_ids = {"DB001"}  # DB999 is not a known drug

    pairs = build_ddi_pairs_table(train_df, val_df, test_df, known_ids)

    assert len(pairs) == 0


def test_build_ddi_pairs_table_drops_exact_duplicates():
    train_df = pd.DataFrame(
        {
            "d1": ["DB001", "DB001"],
            "d2": ["DB002", "DB002"],
            "type": [5, 5],
            "split": ["training", "training"],
        }
    )
    val_df = pd.DataFrame(columns=["d1", "d2", "type", "split"])
    test_df = pd.DataFrame(columns=["d1", "d2", "type", "split"])
    known_ids = {"DB001", "DB002"}

    pairs = build_ddi_pairs_table(train_df, val_df, test_df, known_ids)

    assert len(pairs) == 1
