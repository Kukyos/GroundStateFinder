import sys
import os
import pytest

# Ensure the repository root is on sys.path so pytest can import the module when run
# from different working directories (CI or conda-run). Tests live in ./tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from qiskit.quantum_info import SparsePauliOp

from h2_qubit_hamiltonian import build_molecule_qubit_hamiltonian


def test_fallback_returns_sparsepauliop():
    op = build_molecule_qubit_hamiltonian("NH3", force_precomputed=True)
    assert isinstance(op, SparsePauliOp)
    assert len(op.coeffs) > 0


def test_abinitio_or_fallback_type():
    # Try to run the ab-initio route; if the environment lacks PySCF/Qiskit N, we still
    # expect a SparsePauliOp from the fallback path. This test is intentionally tolerant.
    op = build_molecule_qubit_hamiltonian("NH3", force_precomputed=False)
    assert isinstance(op, SparsePauliOp)
