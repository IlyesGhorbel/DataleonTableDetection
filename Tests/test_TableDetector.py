import sys
import os
import pytest

# Ajouter src/ au chemin pour que Python trouve TableDetector.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from TableDetector import TableDetector

detector = TableDetector()

def test_invoice():
    results = detector.predict(os.path.join("Tests", "Data", "Invoice.png"))
    assert any(r["label"] == "table" for r in results)

def test_bank_statement():
    results = detector.predict(os.path.join("Tests", "Data", "BankStatement.png"))
    assert any(r["label"] == "table" for r in results)

def test_no_table():
    results = detector.predict(os.path.join("Tests", "Data", "NoTable.png"))
    assert len(results) == 0

def test_invalid_file():
    with pytest.raises(ValueError):
        detector.predict(os.path.join("Tests", "Data", "missing.png"))
