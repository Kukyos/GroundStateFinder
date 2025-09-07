"""Print the exact ground-state energy for the current NH3 active-space Hamiltonian.

This uses the same HamiltonianPlugin as the VQE flow, then performs exact
diagonalization of the 6-qubit operator to obtain the true ground energy (Hartree).
"""
import sys
import traceback

try:
    from vqeskeletal import HamiltonianPlugin, exact_ground_state_energy
except Exception:
    print("[Error] Unable to import vqeskeletal. Run from repo root or fix PYTHONPATH.")
    sys.exit(2)

try:
    h_plugin = HamiltonianPlugin()
    info = h_plugin.get_hamiltonian()
    H = info['hamiltonian_active']
    e0 = exact_ground_state_energy(H)
    print("NH3 active-space exact ground-state energy (Hartree):", f"{e0:.10f}")
    if info.get('fallback', False):
        print("[Note] Fallback/simplified Hamiltonian in use; this value is for the simplified model.")
except Exception:
    traceback.print_exc()
    sys.exit(2)
