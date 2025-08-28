# H2 Qubit Hamiltonian (Qiskit Nature)

Minimal example producing the Jordan–Wigner qubit Hamiltonian for the hydrogen molecule (H₂) in the STO-3G basis near equilibrium bond length (~0.74 Å).

The script attempts an ab initio build via PySCF. If PySCF (and a compatible Python version, e.g. 3.10) is not available, it falls back to a literature-sourced Hamiltonian so you always get a usable `SparsePauliOp`.

## Files

- `h2_qubit_hamiltonian.py` – Main script. Run to print the Hamiltonian.
- `requirements.txt` – Pinned dependencies (choose Python 3.10 for PySCF wheels).

## Quick Start (Fallback Works Without PySCF)

```powershell
python h2_qubit_hamiltonian.py
```

If PySCF is missing / unsupported, you'll see a note that a fallback Hamiltonian was used.

## Recommended (Full Ab Initio Path)

1. Install Python 3.10 (add to PATH).
2. Create and activate a virtual environment:
   ```powershell
   py -3.10 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
3. Install dependencies:
   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```
4. Run:
   ```powershell
   python h2_qubit_hamiltonian.py
   ```

If PySCF installs successfully, the script will compute integrals and map them; otherwise the pre-tabulated Hamiltonian prints.

## Output Format

Each line: `<coefficient> * <PauliString>` (sorted, compact). Suitable for direct use in VQE or other algorithms.

## Example Use in VQE (Sketch)

```python
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import SLSQP
from qiskit.circuit.library import EfficientSU2
from h2_qubit_hamiltonian import build_h2_qubit_hamiltonian

ham = build_h2_qubit_hamiltonian()
ansatz = EfficientSU2(ham.num_qubits, reps=2)
vqe = VQE(ansatz=ansatz, optimizer=SLSQP())
result = vqe.compute_minimum_eigenvalue(operator=ham)
print(result.eigenvalue.real)
```

## Publish to GitHub

After creating a repo on GitHub (e.g. `yourname/h2-qubit-hamiltonian`):

```powershell
git init
git add .
git commit -m "Initial commit: H2 qubit Hamiltonian example"
git branch -M main
git remote add origin https://github.com/yourname/h2-qubit-hamiltonian.git
git push -u origin main
```

## License

Apache-2.0. See `LICENSE`.

## Notes

- PySCF wheels lag newer Python releases; use Python 3.10 for easiest installation.
- Fallback coefficients sourced from widely cited H₂ minimal basis benchmark Hamiltonian (Jordan–Wigner mapping).
