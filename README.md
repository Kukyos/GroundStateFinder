## GroundStateFinder – Small Molecule Qubit Hamiltonians (H2, NH3 Active Space)

Generates Jordan–Wigner qubit Hamiltonians for:
* NH3 (active space: 3 spatial orbitals / 4 electrons → 6 qubits)
* H2 (minimal STO‑3G → 4 qubits)

Robust behavior:
* Ab‑initio path via Qiskit Nature + PySCF when available.
* Automatic NH3 6‑qubit fallback (precomputed) if integrals / transformer fail.
* Optional identity stripping to separate constant shift.
* Pauli term utilities (`pauli_terms`, `format_terms`).

Package layout now lives under `src/groundstate/` while a legacy CLI wrapper `h2_qubit_hamiltonian.py` remains for convenience.

Core files

| Path | Purpose |
|------|---------|
| `h2_qubit_hamiltonian.py` | Backward‑compatible CLI (will delegate to package soon). |
| `src/groundstate/hamiltonian.py` | Main implementation (builder, formatting, fallback). |
| `requirements.txt` | Reproducible dependency pins (Qiskit 2.x, PySCF). |
| `tests/test_hamiltonian.py` | Basic qubit count & fallback tests. |
| `save_operator.py` | Serialize `SparsePauliOp` to JSON. |
| `colab/nh3_abinitio_colab.ipynb` | Colab helper (now minimal: install + clone + run). |

Quick start (precomputed fallback)

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

Provenance & best practices

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

Google Colab (minimal workflow)
-------------------------------

New notebook flow is simplified to: (1) install dependencies, (2) clone repo, (3) run script. Active‑space + fallback logic lives only in the repo code (do not duplicate in the notebook).

Single cell quick start (pip wheels):
```python
!pip install -q qiskit==2.1.2 qiskit-nature==0.7.2 pyscf==2.6.1
!git clone https://github.com/Kukyos/GroundStateFinder.git repo
!python repo/h2_qubit_hamiltonian.py -m NH3 --strip-identity
```

If PySCF wheel is unavailable for the Colab Python version, fallback automatically triggers for NH3. For guaranteed PySCF success use a local WSL / conda environment.

License

Apache-2.0. See `LICENSE`.
