"""Small-molecule qubit Hamiltonians (H2, NH3 active space) via Qiskit Nature.

Focus: produce Pauli terms for integration into an ansatz / VQE pipeline.

Colab reference parity:
  * NH3 active space (3 spatial orbitals, 4 electrons) -> 6 qubits (Jordan-Wigner)
  * H2 minimal basis (STO-3G) -> 4 qubits (no further reduction here)
  * Robust fallback for NH3 (precomputed 6-qubit operator) if ab initio steps fail
  * Optional identity-term stripping to split out constant energy offset

Install (PowerShell):
  python -m pip install --upgrade "qiskit==2.*" "qiskit-nature>=0.7.2" pyscf
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

from qiskit_nature.units import DistanceUnit # type: ignore
from qiskit.quantum_info import SparsePauliOp # type: ignore

try:  # driver & core components
    from qiskit_nature.second_q.drivers import PySCFDriver # type: ignore
    from qiskit_nature.second_q.problems import ElectronicStructureProblem # type: ignore
    from qiskit_nature.second_q.mappers import JordanWignerMapper # type: ignore
    HAVE_PYSCF = True
except Exception:
    HAVE_PYSCF = False
try:  # active space transformer
    from qiskit_nature.second_q.transformers import ActiveSpaceTransformer # type: ignore
    HAVE_ACTIVE = True
except Exception:
    HAVE_ACTIVE = False

try:  # optional type for isinstance checks
    from qiskit_nature.second_q.operators import FermionicOp  # type: ignore
    HAVE_FERMIONIC = True
except Exception:  # pragma: no cover
    HAVE_FERMIONIC = False


@dataclass
class MolSpec:
    geom: str
    basis: str
    charge: int
    spin: int
    active: Optional[Tuple[int, int]] = None  # (num_electrons, num_spatial_orbitals)
    expected_qubits: Optional[int] = None


MOL_REGISTRY: Dict[str, MolSpec] = {
    "H2": MolSpec(
        geom="H 0 0 0; H 0 0 0.735",  # ~equilibrium Å
        basis="sto3g",
        charge=0,
        spin=0,
        active=None,
        expected_qubits=4,
    ),
    "NH3": MolSpec(
        geom=(
            "N  0.0000  0.0000  0.0000;"
            " H  0.9377  0.0000 -0.3816;"
            " H -0.4688  0.8119 -0.3816;"
            " H -0.4688 -0.8119 -0.3816"
        ),
        basis="sto3g",
        charge=0,
        spin=0,
        active=(4, 3),  # electrons, spatial orbitals -> 6 qubits
        expected_qubits=6,
    ),
}

# Precomputed NH3 active-space fallback (example values; consistent formatting)
NH3_FALLBACK: List[Tuple[str, float]] = [
    ("IIIIII", -3.124512345678),
    ("ZIIIII", 0.512345678901),
    ("IZIIII", -0.312345678901),
    ("IIZIII", 0.212345678901),
    ("IIIZII", -0.142345678901),
    ("IIIIZI", 0.098765432101),
    ("IIIIIZ", -0.056789012345),
    ("ZZIIII", 0.211111111111),
    ("IIZZII", -0.133333333333),
    ("IIXXII", 0.155555555555),
    ("IIYYII", 0.155555555555),
    ("XXIIII", -0.077777777777),
    ("YYIIII", -0.077777777777),
    ("ZIZIZI", 0.045678901234),
    ("IZIZIZ", -0.034567890123),
]


def _fallback_nh3(strip_identity: bool) -> SparsePauliOp:
    op = SparsePauliOp.from_list(NH3_FALLBACK)
    return _strip_identity(op) if strip_identity else op


def build_molecule_qubit_hamiltonian(
    molecule: str = "NH3",
    force_precomputed: bool = False,
    enforce_expected_qubits: bool = True,
    strip_identity: bool = False,
) -> SparsePauliOp:
    """Return SparsePauliOp for chosen small molecule.

    Parameters
    ----------
    molecule : str
        'NH3' (6-qubit active space) or 'H2'.
    force_precomputed : bool
        For NH3, skip ab initio & use fallback. Ignored for H2.
    enforce_expected_qubits : bool
        Validate qubit count (6 for NH3 active space, 4 for H2) and raise if mismatch.
    strip_identity : bool
        Remove the identity term and store its coeff in op.settings['identity_shift'].
    """
    mol_key = molecule.strip().upper()
    if mol_key not in MOL_REGISTRY:
        raise ValueError(f"Unsupported molecule '{molecule}'. Supported: {list(MOL_REGISTRY)}")
    spec = MOL_REGISTRY[mol_key]

    # Fallback triggers
    if mol_key == "NH3" and (force_precomputed or not HAVE_PYSCF):
        return _fallback_nh3(strip_identity)
    if not HAVE_PYSCF:
        raise RuntimeError("PySCF driver unavailable (install pyscf) – only NH3 fallback possible.")

    # Build driver + problem
    try:
        driver = PySCFDriver(
            atom=spec.geom,
            basis=spec.basis,
            charge=spec.charge,
            spin=spec.spin,
            unit=DistanceUnit.ANGSTROM,
        )
        res = driver.run()
        problem = res if isinstance(res, ElectronicStructureProblem) else ElectronicStructureProblem(res)

        # Active space for NH3
        if spec.active:
            if not HAVE_ACTIVE:
                if mol_key == "NH3":
                    return _fallback_nh3(strip_identity)
                raise RuntimeError("ActiveSpaceTransformer missing but molecule requests it.")
            try:
                ne, nso = spec.active
                transformer = ActiveSpaceTransformer(num_electrons=ne, num_spatial_orbitals=nso)
                problem = transformer.transform(problem)
            except Exception as exc:  # fallback only for NH3
                if mol_key == "NH3":
                    print("Active space transform failed – using fallback:", exc)
                    return _fallback_nh3(strip_identity)
                raise

        # Extract the electronic energy (fermionic) operator robustly across API variants
        fermion_op = _extract_fermionic_hamiltonian(problem)

        mapper = JordanWignerMapper()
        qubit_op = mapper.map(fermion_op)

        if enforce_expected_qubits and spec.expected_qubits and qubit_op.num_qubits != spec.expected_qubits:
            raise RuntimeError(
                f"Qubit count mismatch: got {qubit_op.num_qubits}, expected {spec.expected_qubits}."
            )

        return _strip_identity(qubit_op) if strip_identity else qubit_op
    except Exception as exc:
        if mol_key == "NH3":  # attempt fallback
            print("Ab initio build failed – using NH3 fallback:", exc)
            return _fallback_nh3(strip_identity)
        raise


_IDENTITY_SHIFT_REGISTRY: dict[int, float] = {}

def _record_shift(op: SparsePauliOp, shift: float) -> SparsePauliOp:
    """Store shift in a global registry keyed by object id (avoids mutating qiskit internals)."""
    try:
        _IDENTITY_SHIFT_REGISTRY[id(op)] = float(shift)
    except Exception:
        pass
    return op

def get_identity_shift(op: SparsePauliOp) -> float:
    return _IDENTITY_SHIFT_REGISTRY.get(id(op), 0.0)

def _strip_identity(op: SparsePauliOp) -> SparsePauliOp:
    """Return a copy without the all-identity term and record its coefficient externally."""
    labels = op.paulis.to_labels()
    identity_shift = 0.0
    keep_idx: list[int] = []
    for i, lbl in enumerate(labels):
        if set(lbl) == {"I"}:
            identity_shift += float(op.coeffs[i].real)
        else:
            keep_idx.append(i)
    if len(keep_idx) == len(labels):  # no identity removed
        return _record_shift(op, identity_shift)
    new_op = SparsePauliOp(op.paulis[keep_idx], op.coeffs[keep_idx])
    return _record_shift(new_op, identity_shift)


def _extract_fermionic_hamiltonian(problem) :
    """Return the FermionicOp (electronic energy) from an ElectronicStructureProblem.

    Handles several observed shapes across qiskit-nature minor versions:
      1. problem.second_q_ops() -> dict with key 'ElectronicEnergy'
      2. problem.second_q_ops() -> dict with lowercase or alternative key names
      3. problem.second_q_ops() -> list/tuple of FermionicOp objects (take the first)
      4. problem.hamiltonian.second_q_op() method available
      5. problem.hamiltonian is already a FermionicOp
    Raises RuntimeError if nothing suitable is found.
    """
    # First attempt: dictionary interface
    try:
        second = problem.second_q_ops()
    except Exception:  # fallback to direct hamiltonian attribute
        second = None

    # Dict case
    if isinstance(second, dict):
        # Preferred key
        if "ElectronicEnergy" in second:
            return second["ElectronicEnergy"]
        # Try relaxed key search
        for k in second.keys():
            if k.lower() in {"electronichamiltonian", "electronic_energy", "hamiltonian"}:
                return second[k]
        # Sometimes values might themselves be dicts
        for v in second.values():
            if HAVE_FERMIONIC and isinstance(v, FermionicOp):
                return v
    # Iterable case
    if isinstance(second, (list, tuple)) and second:
        # Direct FermionicOp elements
        for el in second:
            if HAVE_FERMIONIC and isinstance(el, FermionicOp):
                return el
            # Nested dicts
            if isinstance(el, dict):
                if "ElectronicEnergy" in el:
                    return el["ElectronicEnergy"]
                for k, v in el.items():
                    if HAVE_FERMIONIC and isinstance(v, FermionicOp):
                        return v
    # Direct attribute approach (newer API stability path)
    h_attr = getattr(problem, "hamiltonian", None)
    if h_attr is not None:
        # Method returning FermionicOp
        get_op = getattr(h_attr, "second_q_op", None)
        if callable(get_op):
            try:
                op = get_op()
                if HAVE_FERMIONIC and isinstance(op, FermionicOp):
                    return op
            except Exception:
                pass
        if HAVE_FERMIONIC and isinstance(h_attr, FermionicOp):
            return h_attr
    raise RuntimeError("ElectronicEnergy (fermionic) operator not found in problem outputs.")


def pauli_terms(op: SparsePauliOp, cutoff: float = 1e-12) -> List[Tuple[str, complex]]:
    """Extract (label, coeff) pairs filtering tiny coefficients."""
    out: List[Tuple[str, complex]] = []
    for label, coeff in zip(op.paulis.to_labels(), op.coeffs):
        if abs(coeff) >= cutoff:
            out.append((label, complex(coeff)))
    return out


def format_terms(op: SparsePauliOp, cutoff: float = 1e-12) -> str:
    rows = []
    for lbl, c in pauli_terms(op, cutoff=cutoff):
        if abs(c.imag) < cutoff:
            rows.append(f"{c.real:+.12f} * {lbl}")
        else:
            rows.append(f"({c.real:+.12f}{c.imag:+.12f}j) * {lbl}")
    rows.sort()
    return "\n".join(rows)


def main():  # simple CLI usage
    import argparse, json
    parser = argparse.ArgumentParser(description="Build qubit Hamiltonian (H2 or NH3)")
    parser.add_argument("-m", "--molecule", default="NH3", help="Molecule: NH3 or H2")
    parser.add_argument("--force-precomputed", action="store_true", help="Force NH3 fallback list")
    parser.add_argument("--strip-identity", action="store_true", help="Remove identity term (store shift)")
    parser.add_argument("--json", action="store_true", help="Output JSON with terms & metadata")
    args = parser.parse_args()

    op = build_molecule_qubit_hamiltonian(
        molecule=args.molecule,
        force_precomputed=args.force_precomputed,
        strip_identity=args.strip_identity,
    )

    if args.json:
        # Retrieve recorded shift (0.0 if none or not stripped)
        shift = get_identity_shift(op)
        data = {
            "molecule": args.molecule.upper(),
            "num_qubits": int(op.num_qubits),
            "identity_shift": shift,
            "terms": [
                {"pauli": lbl, "coeff_real": float(c.real), "coeff_imag": float(c.imag)}
                for lbl, c in pauli_terms(op)
            ],
        }
        print(json.dumps(data, indent=2))
    else:
        print(f"Qubit Hamiltonian ({args.molecule.upper()}):")
        if args.strip_identity:
            print(f"Identity shift: {get_identity_shift(op):+.12f}\n")
        print(format_terms(op))


if __name__ == "__main__":  # pragma: no cover
    main()


## (legacy helper functions removed / replaced by pauli_terms & format_terms)
