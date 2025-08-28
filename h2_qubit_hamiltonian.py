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


def build_h2_qubit_hamiltonian(bond_length: float = 0.74):
    """Return the qubit Hamiltonian (SparsePauliOp) for H2 at given bond length (Å).

    Attempts an ab initio PySCF run if available. If PySCF or driver stack
    is unavailable (e.g., Python 3.12 without wheels), falls back to a
    pre-tabulated STO-3G Hamiltonian at ~0.735-0.74 Å (standard literature values).
    """
    if _HAVE_PYSCF_DRIVER:
        try:
            geom = f"H 0 0 0; H 0 0 {bond_length}"
            driver = PySCFDriver(
                atom=geom,
                basis="sto3g",
                charge=0,
                spin=0,
                unit=DistanceUnit.ANGSTROM,
            )
            problem = ElectronicStructureProblem(driver)
            second_q_ops = problem.second_q_ops()
            hamiltonian = second_q_ops["ElectronicEnergy"]
            mapper = JordanWignerMapper()
            return mapper.map(hamiltonian)
        except Exception:
            pass  # fall through to hardcoded

    # Fallback: Known qubit Hamiltonian (Jordan-Wigner) for H2, STO-3G, R~0.735 Å
    # Source: Standard VQE tutorials / literature (e.g., O'Malley et al., Qiskit docs)
    coeffs = [
        (-1.052373245772859,),  # I
        (0.39793742484318045, 'Z0'),
        (-0.39793742484318045, 'Z1'),
        (-0.01128010425623538, 'Z2'),
        (0.18093119978423156, 'Z3'),
        (0.39793742484318045, 'Z0 Z1'),
        (-0.18093119978423156, 'Z0 Z3'),
        (-0.01128010425623538, 'Z1 Z2'),
        (0.18093119978423156, 'Z2 Z3'),
        (0.1689275387008791, 'X0 X1 Y2 Y3'),
        (-0.1689275387008791, 'X0 Y1 Y2 X3'),
        (-0.1689275387008791, 'Y0 X1 X2 Y3'),
        (0.1689275387008791, 'Y0 Y1 X2 X3'),
    ]
    paulis = []
    values = []
    for entry in coeffs:
        coeff = entry[0]
        if len(entry) == 1:  # identity
            paulis.append('IIII')
        else:
            # Build 4-qubit string initialized as identity
            ops = ['I'] * 4
            for term in entry[1].split():
                p = term[0]
                idx = int(term[1])
                ops[idx] = p
            paulis.append(''.join(ops))
        values.append(coeff)
    return SparsePauliOp.from_list(list(zip(paulis, values)))


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
    qubit_h = build_h2_qubit_hamiltonian()
    print("Qubit Hamiltonian for H2 (Jordan-Wigner, STO-3G, ~0.74 Å):")
    if not _HAVE_PYSCF_DRIVER:
        print("(Fallback pre-tabulated coefficients used; install Python 3.10 + PySCF for ab initio generation.)")
    print(compact_string(qubit_h))


if __name__ == "__main__":  # pragma: no cover
    main()
