"""Local runner for VQE to mirror Colab setup.

Installs/validates required modules and executes a minimal VQE run,
reporting whether the ab initio Hamiltonian or fallback synthetic
Hamiltonian was used.
"""
import importlib
import sys
import traceback

REQUIRED = [
    ("qiskit", None),
    ("qiskit_nature", None),
    ("pyscf", None),
]

missing = []
for mod, pkg in REQUIRED:
    try:
        importlib.import_module(mod)
    except ImportError:
        missing.append(pkg or mod)

if missing:
    print("Missing packages detected:", ", ".join(missing))
    print("Install them (PowerShell):")
    print("    pip install " + " ".join(missing))
    sys.exit(1)

from vqeskeletal import HamiltonianPlugin, AnsatzPlugin, HybridSPSAThenCOBYLA, ZNEDenoiserPlugin, VQE

try:
    h_plugin = HamiltonianPlugin()
    a_plugin = AnsatzPlugin(verbose=True)
    opt_plugin = HybridSPSAThenCOBYLA(spsa_iters=10, switch_tol=5e-3, min_spsa=5, force_cobyla=True, verbose=True)
    zne_plugin = ZNEDenoiserPlugin()
    vqe = VQE(a_plugin, h_plugin, opt_plugin, zne_plugin, verbose=True)
    if getattr(h_plugin, 'was_fallback', lambda: False)():
        print("[DIAG] Fallback Hamiltonian in use -> energies will be around -5 Hartree.")
    else:
        print("[DIAG] Ab initio Hamiltonian active.")
    params0 = a_plugin.get_initial_parameters("random_small")
    best_params, best_energy = vqe.run(params0)
    print("Final energy:", best_energy)
except Exception:
    traceback.print_exc()
    sys.exit(2)
