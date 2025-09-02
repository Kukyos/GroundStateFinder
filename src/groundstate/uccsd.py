"""UCCSD ansatz builders aligned with groundstate.hamiltonian outputs.

Provides helpers to construct a UCCSD circuit sized to match an existing
qubit Hamiltonian (e.g. from build_molecule_qubit_hamiltonian) or to
derive an active-space ansatz from raw geometry.

These are distilled from the prototype notebook `UCCSD_final_version.ipynb`.
Minor adjustments:
  * Removed print noise except explicit compatibility check helpers.
  * Deterministic optional seed for parameter generation.
  * Validates qubit parity with provided Hamiltonian.
"""

from __future__ import annotations
from typing import Iterable, List, Sequence, Tuple, Optional
import numpy as np

from qiskit_nature.units import DistanceUnit # type: ignore
from qiskit_nature.second_q.drivers import PySCFDriver # type: ignore
from qiskit_nature.second_q.problems import ElectronicStructureProblem # type: ignore
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer # type: ignore
from qiskit_nature.second_q.mappers import JordanWignerMapper # type: ignore
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock # type: ignore


def _geom_to_atom_string(geometry) -> str:
    if isinstance(geometry, str):
        return geometry
    # Expect list like [["N", [x,y,z]], ...]
    parts = []
    for item in geometry:
        el, coords = item[0], item[1]
        parts.append(f"{el} {coords[0]:.4f} {coords[1]:.4f} {coords[2]:.4f}")
    return "; ".join(parts)


def _active_space_from_target_qubits(problem: ElectronicStructureProblem, target_qubits: int) -> Tuple[int, int]:
    """Return (electrons, spatial_orbitals) approximating target qubit count.

    target_qubits should be even. spatial_orbitals = target_qubits/2.
    Adjust electrons downwards if necessary to remain even and <= 2*orbitals.
    """
    if target_qubits % 2:
        raise ValueError("Target qubit count must be even for spin-orbital pairing.")
    spatial = target_qubits // 2
    full_electrons = sum(problem.num_particles)
    # electrons limited by 2*spatial; choose min even number <= full_electrons
    max_allowed = 2 * spatial
    e = min(full_electrons, max_allowed)
    if e % 2:
        e -= 1
    # UCCSD requires at least one virtual orbital per spin: n_spin < spatial.
    # If we end up with a *fully* occupied active space (e/2 == spatial) there
    # are zero virtual orbitals and UCCSD raises a configuration error.
    # Reduce electrons (in pairs) until each spin count < spatial.
    while e // 2 >= spatial and e > 0:
        e -= 2
    if e <= 0:
        raise ValueError("Computed non-positive active electron count")
    return e, spatial


def uccsd_active(
    geometry,
    basis: str = "sto3g",
    target_qubits: Optional[int] = None,
    max_qubits: int = 12,
    param_scale: float = 0.02,
    seed: Optional[int] = None,
):
    """Build a UCCSD ansatz with optional target qubit count.

    Returns (ansatz, initial_parameters, num_qubits)
    """
    atom_string = _geom_to_atom_string(geometry)
    driver = PySCFDriver(atom=atom_string, basis=basis, charge=0, spin=0, unit=DistanceUnit.ANGSTROM)
    problem_raw = driver.run()
    problem = problem_raw if isinstance(problem_raw, ElectronicStructureProblem) else ElectronicStructureProblem(problem_raw)

    if target_qubits is None:
        target_qubits = min(max_qubits, 8 if max_qubits >= 8 else max_qubits)
    electrons, spatial = _active_space_from_target_qubits(problem, target_qubits)

    # Build active space; if still fully occupied (edge case) decrement electrons and retry.
    # (Defensive: the earlier guard should prevent this, but race against API nuances.)
    attempt_e = electrons
    while True:
        transformer = ActiveSpaceTransformer(num_electrons=attempt_e, num_spatial_orbitals=spatial)
        problem_active = transformer.transform(problem)
        spin_orbs = problem_active.num_spin_orbitals // 2
        # problem_active.num_particles maybe tuple (alpha,beta)
        parts = problem_active.num_particles if isinstance(problem_active.num_particles, (list, tuple)) else (problem_active.num_particles, problem_active.num_particles)
        if all(p < spin_orbs for p in parts):
            break
        attempt_e -= 2
        if attempt_e <= 0:
            raise ValueError("Failed to find active space with at least one virtual orbital per spin.")

    mapper = JordanWignerMapper()
    ansatz = UCCSD(
        num_spatial_orbitals=problem_active.num_spin_orbitals // 2,
        num_particles=problem_active.num_particles,
        qubit_mapper=mapper,
    )

    rng = np.random.default_rng(seed)
    params = rng.normal(0, param_scale, ansatz.num_parameters) if ansatz.num_parameters else np.array([])
    return ansatz, params, ansatz.num_qubits


def uccsd_for_hamiltonian(geometry, hamiltonian, basis: str = "sto3g", **kwargs):
    """Convenience wrapper sizing ansatz to an existing qubit Hamiltonian."""
    target_qubits = hamiltonian.num_qubits
    ansatz, params, _ = uccsd_active(geometry, basis=basis, target_qubits=target_qubits, **kwargs)
    if ansatz.num_qubits != target_qubits:
        raise RuntimeError(
            f"Ansatz qubits {ansatz.num_qubits} != Hamiltonian qubits {target_qubits} (unexpected mismatch)."
        )
    return ansatz, params


def random_params(num_params: int, scale: float = 0.02, seed: Optional[int] = None):
    rng = np.random.default_rng(seed)
    return rng.normal(0, scale, num_params) if num_params else np.array([])


def circuit_summary(ansatz, max_gates: int = 25) -> str:
    """Return a textual summary of the ansatz (limited gate listing)."""
    lines = [f"UCCSD: qubits={ansatz.num_qubits} params={ansatz.num_parameters} depth={ansatz.depth()}"]
    data = ansatz.data
    if len(data) <= max_gates:
        lines.append(str(ansatz))
    else:
        lines.append(f"First {max_gates} / {len(data)} gates:")
        for inst in data[:max_gates]:
            qubits = [ansatz.find_bit(q).index for q in inst.qubits]
            lines.append(f"  {inst.operation.name} {qubits}")
        lines.append(f"  ... +{len(data) - max_gates} more gates")
    return "\n".join(lines)


__all__ = [
    "uccsd_active",
    "uccsd_for_hamiltonian",
    "random_params",
    "circuit_summary",
]
