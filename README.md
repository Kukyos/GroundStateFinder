## NH3 qubit Hamiltonian (Qiskit Nature)

This repository provides a small example that produces a Jordan–Wigner qubit Hamiltonian for ammonia (NH3) in the STO-3G basis.

The script `h2_qubit_hamiltonian.py` (misnamed historically) attempts to compute the molecular integrals and build the electronic Hamiltonian using Qiskit Nature with a PySCF driver. If PySCF isn't available or can't be built on your system, the script falls back to a precomputed NH3 Hamiltonian so the repository can be used for development and algorithm tests without heavy native dependencies.

Files

- `h2_qubit_hamiltonian.py` – Main script. Call with `--precomputed` to force the included fallback operator.
- `requirements.txt` – Suggested dependencies (use Python 3.10 if you intend to install PySCF).

Additional helper files

- `save_operator.py` – small helper to save the produced SparsePauliOp to a JSON file.
- `tests/test_hamiltonian.py` – pytest tests (happy path + fallback) that run quickly and verify the function returns a SparsePauliOp.

Quick start (use precomputed fallback)

```powershell
# prints the precomputed NH3 Jordan-Wigner Hamiltonian
python h2_qubit_hamiltonian.py --precomputed
```

Attempt full ab initio generation (PySCF)

On Windows the easiest reliable path is WSL (Ubuntu) or a conda env with appropriate native toolchain installed. Briefly:

1. Create a conda env with Python 3.10 (Anaconda/Miniconda):
```powershell
conda create -n qn-env python=3.10 -y
conda activate qn-env
```

2. On Linux/WSL use conda-forge:
```bash
conda install -c conda-forge pyscf qiskit qiskit-nature -y
```

3. On Windows native builds require compilers and BLAS/LAPACK (see README notes in code). For many users, WSL is simpler.

Output format

Each printed line is `<coefficient> * <PauliString>` (sorted and compact). The operator is a `qiskit.quantum_info.SparsePauliOp` and is directly usable in VQE or other quantum algorithms.

Provenance and best practices

- The precomputed NH3 Hamiltonian is included as a convenience for development. Add provenance for any scientific use: geometry, basis (STO-3G), method (HF/CCSD/etc), mapping (Jordan–Wigner), and qubit ordering.
- Using precomputed operators is common for testing, demos, and CI. For science-grade results, compute the Hamiltonian ab initio and document the method.

Examples

Force precomputed operator and run (PowerShell):
```powershell
python h2_qubit_hamiltonian.py --precomputed
```

Run the real generator (PySCF present):
```powershell
python h2_qubit_hamiltonian.py
```

If PySCF is available the script will attempt to compute integrals and map them; otherwise it will print the precomputed operator with a note.

Setup, test, and save (WSL / conda recommended on Windows)

1) Create and activate a conda env (Python 3.10 recommended):

```powershell
# in PowerShell
wsl -d Ubuntu -- bash -lc "conda create -n qn-env python=3.10 -y"
wsl -d Ubuntu -- bash -lc "conda run -n qn-env conda install -c conda-forge pyscf qiskit qiskit-nature pytest -y"
```

2) Run the script (ab‑initio when PySCF available):

```powershell
wsl -d Ubuntu -- bash -lc "conda run -n qn-env python /mnt/c/Users/Cleo/Desktop/groundstate/h2_qubit_hamiltonian.py"
```

3) Run tests (quick, uses precomputed fallback to avoid heavy native steps):

```powershell
wsl -d Ubuntu -- bash -lc "conda run -n qn-env pytest -q"
```

4) Save the operator to a file (JSON) using the helper:

```powershell
wsl -d Ubuntu -- bash -lc "conda run -n qn-env python /mnt/c/Users/Cleo/Desktop/groundstate/save_operator.py --out nh3_op.json"
```

Version pins

The environment used during development was pinned for stability. If you want reproducible behavior pin these versions in `requirements.txt` or your conda specs. The recommended pins (known-good in this repo) are included in `requirements.txt`.

Contributing / PR

1. Create a branch named `feature/<short-desc>`.
2. Run tests locally: `pytest -q`.
3. Commit and push, open a PR to `main`. Include a short description and the output of `pytest -q` in the PR body.

Notes

- The precomputed operator is provided for development and CI. For scientific use, regenerate the Hamiltonian with PySCF and include provenance (geometry, basis, method, mapping, qubit ordering).
- If you hit import errors for PySCF on Windows prefer WSL+conda-forge or a Linux CI runner.

License

Apache-2.0. See `LICENSE`.
