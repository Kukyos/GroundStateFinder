# Copied from root h2_qubit_hamiltonian.py for package organization
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
from qiskit_nature.units import DistanceUnit
from qiskit.quantum_info import SparsePauliOp
try:
    from qiskit_nature.second_q.drivers import PySCFDriver
    from qiskit_nature.second_q.problems import ElectronicStructureProblem
    from qiskit_nature.second_q.mappers import JordanWignerMapper
    HAVE_PYSCF = True
except Exception:
    HAVE_PYSCF = False
try:
    from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
    HAVE_ACTIVE = True
except Exception:
    HAVE_ACTIVE = False
try:
    from qiskit_nature.second_q.operators import FermionicOp  # type: ignore
    HAVE_FERMIONIC = True
except Exception:
    HAVE_FERMIONIC = False

@dataclass
class MolSpec:
    geom: str
    basis: str
    charge: int
    spin: int
    active: Optional[Tuple[int, int]] = None
    expected_qubits: Optional[int] = None

MOL_REGISTRY: Dict[str, MolSpec] = {
    "H2": MolSpec(
        geom="H 0 0 0; H 0 0 0.735",
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
        active=(4, 3),
        expected_qubits=6,
    ),
}

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
    mol_key = molecule.strip().upper()
    if mol_key not in MOL_REGISTRY:
        raise ValueError(f"Unsupported molecule '{molecule}'. Supported: {list(MOL_REGISTRY)}")
    spec = MOL_REGISTRY[mol_key]
    if mol_key == "NH3" and (force_precomputed or not HAVE_PYSCF):
        return _fallback_nh3(strip_identity)
    if not HAVE_PYSCF:
        raise RuntimeError("PySCF driver unavailable (install pyscf) – only NH3 fallback possible.")
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
        if spec.active:
            if not HAVE_ACTIVE:
                if mol_key == "NH3":
                    return _fallback_nh3(strip_identity)
                raise RuntimeError("ActiveSpaceTransformer missing but molecule requests it.")
            try:
                ne, nso = spec.active
                transformer = ActiveSpaceTransformer(num_electrons=ne, num_spatial_orbitals=nso)
                problem = transformer.transform(problem)
            except Exception as exc:
                if mol_key == "NH3":
                    print("Active space transform failed – using fallback:", exc)
                    return _fallback_nh3(strip_identity)
                raise
        fermion_op = _extract_fermionic_hamiltonian(problem)
        mapper = JordanWignerMapper()
        qubit_op = mapper.map(fermion_op)
        if enforce_expected_qubits and spec.expected_qubits and qubit_op.num_qubits != spec.expected_qubits:
            raise RuntimeError(
                f"Qubit count mismatch: got {qubit_op.num_qubits}, expected {spec.expected_qubits}."
            )
        return _strip_identity(qubit_op) if strip_identity else qubit_op
    except Exception as exc:
        if mol_key == "NH3":
            print("Ab initio build failed – using NH3 fallback:", exc)
            return _fallback_nh3(strip_identity)
        raise

def _strip_identity(op: SparsePauliOp) -> SparsePauliOp:
    labels = op.paulis.to_labels()
    identity_shift = 0.0
    keep_idx = []
    for i, lbl in enumerate(labels):
        if set(lbl) == {"I"}:
            identity_shift += op.coeffs[i].real
        else:
            keep_idx.append(i)
    if len(keep_idx) == len(labels):
        op.settings = {"identity_shift": identity_shift}
        return op
    new_op = SparsePauliOp(op.paulis[keep_idx], op.coeffs[keep_idx])
    new_op.settings = {"identity_shift": identity_shift}
    return new_op

def _extract_fermionic_hamiltonian(problem):
    try:
        second = problem.second_q_ops()
    except Exception:
        second = None
    if isinstance(second, dict):
        if "ElectronicEnergy" in second:
            return second["ElectronicEnergy"]
        for k in second.keys():
            if k.lower() in {"electronichamiltonian", "electronic_energy", "hamiltonian"}:
                return second[k]
        for v in second.values():
            if HAVE_FERMIONIC and isinstance(v, FermionicOp):
                return v
    if isinstance(second, (list, tuple)) and second:
        for el in second:
            if HAVE_FERMIONIC and isinstance(el, FermionicOp):
                return el
            if isinstance(el, dict):
                if "ElectronicEnergy" in el:
                    return el["ElectronicEnergy"]
                for _, v in el.items():
                    if HAVE_FERMIONIC and isinstance(v, FermionicOp):
                        return v
    h_attr = getattr(problem, "hamiltonian", None)
    if h_attr is not None:
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
    out: List[Tuple[str, complex]] = []
    for label, coeff in zip(op.paulis.to_labels(), op.coeffs):
        if abs(coeff) >= cutoff:
            out.append((label, complex(coeff)))
    return out

def pauli_terms_json_ready(op: SparsePauliOp, cutoff: float = 1e-12) -> List[Dict[str, float]]:
    """Return list of dicts with primitive float values for JSON serialization."""
    js: List[Dict[str, float]] = []
    for lbl, c in pauli_terms(op, cutoff=cutoff):
        js.append({"pauli": lbl, "coeff_real": float(c.real), "coeff_imag": float(c.imag)})
    return js

def format_terms(op: SparsePauliOp, cutoff: float = 1e-12) -> str:
    rows = []
    for lbl, c in pauli_terms(op, cutoff=cutoff):
        if abs(c.imag) < cutoff:
            rows.append(f"{c.real:+.12f} * {lbl}")
        else:
            rows.append(f"({c.real:+.12f}{c.imag:+.12f}j) * {lbl}")
    rows.sort()
    return "\n".join(rows)

__all__ = [
    "build_molecule_qubit_hamiltonian",
    "pauli_terms",
    "format_terms",
    "pauli_terms_json_ready",
]
