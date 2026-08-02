"""Unit tests for Module A (Data Ingestion).

verify_file() is tested in isolation, without network access. The actual
download is network-dependent and is exercised by running
`python src/ingestion/run.py` directly, not as part of this suite.
"""

from src.ingestion.run import FILES, verify_file


def test_verify_file_rejects_missing_file(tmp_path):
    missing = tmp_path / "does_not_exist.csv"
    assert verify_file(missing, min_size_bytes=100) is False


def test_verify_file_rejects_undersized_file(tmp_path):
    small_file = tmp_path / "small.csv"
    small_file.write_text("too small")
    assert verify_file(small_file, min_size_bytes=1000) is False


def test_verify_file_accepts_correctly_sized_file(tmp_path):
    ok_file = tmp_path / "ok.csv"
    ok_file.write_text("x" * 2000)
    assert verify_file(ok_file, min_size_bytes=1000) is True


def test_expected_files_registry_matches_documentation():
    expected = {
        "drug_smiles.csv",
        "file_drugs.csv",
        "ddi_training.csv",
        "ddi_validation.csv",
        "ddi_test.csv",
    }
    assert set(FILES.keys()) == expected
