"""Save the SparsePauliOp produced by h2_qubit_hamiltonian.build_molecule_qubit_hamiltonian

Usage:
    python save_operator.py --out nh3_op.json

This writes a small JSON with `paulis` and `coeffs` so the operator can be reloaded later.
"""
import json
import argparse
from qiskit.quantum_info import SparsePauliOp

from h2_qubit_hamiltonian import build_molecule_qubit_hamiltonian


def save(op: SparsePauliOp, path: str):
    paulis = [p.to_label() for p in op.paulis]
    coeffs = [complex(c) for c in op.coeffs]
    # JSON-friendly
    data = {"paulis": paulis, "coeffs": [[c.real, c.imag] for c in coeffs]}
    with open(path, "w", encoding="utf8") as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="Output JSON file path")
    parser.add_argument("--precomputed", action="store_true", help="Force precomputed fallback")
    args = parser.parse_args()

    op = build_molecule_qubit_hamiltonian("NH3", force_precomputed=args.precomputed)
    save(op, args.out)
    print(f"Saved operator with {len(op.coeffs)} terms to {args.out}")


if __name__ == "__main__":
    main()
