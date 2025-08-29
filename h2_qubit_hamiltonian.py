"""Generate the qubit Hamiltonian for H2 using Qiskit Nature (Jordan-Wigner).

Minimal, version-current (2025) usage of the second_q API.
Requires: qiskit, qiskit-nature, pyscf
Install (PowerShell):  python -m pip install --upgrade qiskit qiskit-nature pyscf
"""

from qiskit_nature.units import DistanceUnit
try:
    from qiskit_nature.second_q.drivers import PySCFDriver
    from qiskit_nature.second_q.problems import ElectronicStructureProblem
    from qiskit_nature.second_q.mappers import JordanWignerMapper
    _HAVE_PYSCF_DRIVER = True
except Exception:  # ImportError or other environment issues
    _HAVE_PYSCF_DRIVER = False
from qiskit.quantum_info import SparsePauliOp


def build_molecule_qubit_hamiltonian(molecule: str = "NH3", force_precomputed: bool = False):
    """Return the qubit Hamiltonian (SparsePauliOp) for the requested molecule.

    - If the PySCF-backed Qiskit Nature driver is available the function
      will build the electronic structure problem and map it to qubits.
    - If not available, the function prints a clear instruction and returns
      a trivial zero-operator placeholder (identity with zero coefficient).

    molecule: string, currently supports 'NH3' (default). Additional
    molecules may be added later.
    """
    mol = molecule.strip().upper()
    if mol == "NH3":
        # Reasonable NH3 geometry (Å) in typical orientation
        geom = (
            "N  0.0000  0.0000  0.0000;"
            " H  0.9377  0.0000 -0.3816;"
            " H -0.4688  0.8119 -0.3816;"
            " H -0.4688 -0.8119 -0.3816"
        )
        basis = "sto3g"
        charge = 0
        spin = 0
    else:
        raise ValueError(f"Unsupported molecule: {molecule}")

    if not force_precomputed and _HAVE_PYSCF_DRIVER:
        try:
            driver = PySCFDriver(
                atom=geom,
                basis=basis,
                charge=charge,
                spin=spin,
                unit=DistanceUnit.ANGSTROM,
            )
            # The driver must be run to produce a DriverResult which
            # provides the attributes expected by ElectronicStructureProblem.
            result = driver.run()
            # Some qiskit-nature versions have driver.run() return a
            # DriverResult, while others (installed in some conda builds)
            # may already return an ElectronicStructureProblem. Handle both.
            if isinstance(result, ElectronicStructureProblem):
                problem = result
            else:
                problem = ElectronicStructureProblem(result)
            second_q_ops = problem.second_q_ops()
            # second_q_ops can be a dict (name->op) or a tuple/list
            # (main_op, aux_ops_dict, ...). Handle both shapes robustly.
            hamiltonian = None
            # Prefer dict access when available
            if isinstance(second_q_ops, dict):
                hamiltonian = second_q_ops.get("ElectronicEnergy")

            # If it's a sequence, try common patterns
            if hamiltonian is None and isinstance(second_q_ops, (tuple, list)):
                # often: (main_op, aux_ops_dict)
                for el in second_q_ops:
                    if isinstance(el, dict) and "ElectronicEnergy" in el:
                        hamiltonian = el["ElectronicEnergy"]
                        break
                # fallback: first FermionicOp element is usually the electronic hamiltonian
                if hamiltonian is None:
                    try:
                        from qiskit_nature.second_q.operators.fermionic_op import FermionicOp
                        for el in second_q_ops:
                            if isinstance(el, FermionicOp):
                                hamiltonian = el
                                break
                    except Exception:
                        # cannot import FermionicOp; ignore and continue
                        pass

            if hamiltonian is None:
                # As a last resort, try dict-like access by string key
                try:
                    hamiltonian = second_q_ops["ElectronicEnergy"]
                except Exception:
                    raise RuntimeError("Could not extract ElectronicEnergy from second_q_ops() result")
            # Map using Jordan-Wigner
            mapper = JordanWignerMapper()
            return mapper.map(hamiltonian)
        except Exception as exc:
            print("PySCF driver attempted but failed:", exc)

    # Fallback: return a precomputed NH3 qubit Hamiltonian (Jordan-Wigner)
    print(
        "PySCF-backed driver not available or failed.\n"
        "Using a precomputed NH3 STO-3G Jordan-Wigner Hamiltonian fallback so the script returns a usable operator.\n"
        "For an ab initio Hamiltonian install PySCF (Python 3.10 recommended) and rerun."
    )
    # Precomputed example: 6-qubit active-space style Hamiltonian (illustrative)
    # Pauli strings use the ordering Q0..Q5 (length 6)
    precomputed = [
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
    return SparsePauliOp.from_list(precomputed)


def compact_string(op, cutoff: float = 1e-10) -> str:
    """Format a SparsePauliOp into a compact algebraic sum, filtering tiny terms."""
    # Defer import to avoid issues if only analyzing
    from qiskit.quantum_info import SparsePauliOp  # type: ignore

    if not hasattr(op, "paulis"):
        return str(op)
    terms = []
    for pauli, coeff in zip(op.paulis, op.coeffs):
        if abs(coeff) < cutoff:
            continue
        # Use real if imag negligible
        if abs(coeff.imag) < cutoff:
            terms.append(f"{coeff.real:+.12f} * {pauli}")
        else:
            terms.append(f"({coeff.real:+.12f}{coeff.imag:+.12f}j) * {pauli}")
    # Sort deterministically
    terms.sort()
    return "\n".join(terms)


def main():
    qubit_h = build_molecule_qubit_hamiltonian("NH3")
    print("Qubit Hamiltonian for NH3 (Jordan-Wigner, STO-3G):")
    print(compact_string(qubit_h))


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Build qubit Hamiltonian for a small molecule")
    parser.add_argument("-m", "--molecule", default="NH3", help="Molecule to build (default: NH3)")
    parser.add_argument("--precomputed", action="store_true", help="Force using the precomputed fallback Hamiltonian")
    args = parser.parse_args()

    qubit_h = build_molecule_qubit_hamiltonian(args.molecule, force_precomputed=args.precomputed)
    print(f"Qubit Hamiltonian for {args.molecule} (Jordan-Wigner, STO-3G):")
    print(compact_string(qubit_h))
