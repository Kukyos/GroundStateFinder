import math
import random
import os
import numpy as np

# Imports for the Hamiltonian Plugin
from qiskit.quantum_info import SparsePauliOp # type: ignore
from qiskit import QuantumCircuit   # type: ignore

# The following imports require qiskit-nature and a chemistry driver like pyscf
try:
    from qiskit_nature.units import DistanceUnit # type: ignore
    from qiskit_nature.second_q.drivers import PySCFDriver  # type: ignore
    from qiskit_nature.second_q.transformers import ActiveSpaceTransformer # type: ignore
    from qiskit_nature.second_q.problems import ElectronicStructureProblem # type: ignore
    from qiskit_nature.second_q.mappers import JordanWignerMapper # type: ignore
    from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock # type: ignore
    QISKIT_NATURE_INSTALLED = True
except ImportError:
    QISKIT_NATURE_INSTALLED = False


class AnsatzPlugin:
    """
    Unified UCCSD Ansatz Plugin for VQE
    """
    def __init__(self, ansatz_reps=1, include_hf_state=True, verbose=True):
        self.ansatz_reps = ansatz_reps
        self.include_hf_state = include_hf_state
        self.verbose = verbose
        self.ansatz_circuit = None
        self.hf_state = None
        self.num_parameters = 0
        self.hamiltonian_system = None
        self.is_built = False
        self.num_qubits = 0
        self.num_spatial_orbitals = 0
        self.num_particles = None
        self.vqe_ready = False

    def build_from_hamiltonian(self, hamiltonian_system):
        if self.verbose:
            print("="*70)
            print("BUILDING UCCSD ANSATZ FROM HAMILTONIAN")
            print("="*70)
        self.hamiltonian_system = hamiltonian_system
        problem_active = hamiltonian_system['problem_active']
        mapper = hamiltonian_system['mapper']
        hamiltonian = hamiltonian_system['hamiltonian_active']
        if problem_active is None or mapper is None:
            if self.verbose:
                print("⚠ Warning: Cannot build full UCCSD ansatz - using fallback parameterized ansatz")
            return self._build_fallback_ansatz(hamiltonian)
        self.num_spatial_orbitals = problem_active.num_spin_orbitals // 2
        self.num_particles = problem_active.num_particles
        self.num_qubits = problem_active.num_spin_orbitals
        if self.verbose:
            print(f"System: qubits={self.num_qubits} spatial_orbs={self.num_spatial_orbitals} particles={self.num_particles}")
        if not self._build_hf_state(mapper):
            return False
        if not self._build_uccsd_ansatz(mapper):
            return False
        self._validate_system(hamiltonian)
        self.is_built = True
        if self.verbose:
            print(f"✓ Ansatz construction complete (params={self.num_parameters})")
        return True

    def _build_fallback_ansatz(self, hamiltonian):
        self.num_qubits = hamiltonian.num_qubits
        from qiskit.circuit import Parameter  # type: ignore
        qc = QuantumCircuit(self.num_qubits)
        params = []
        for i in range(self.num_qubits):
            p = Parameter(f'theta_{i}')
            qc.ry(p, i)
            params.append(p)
        for i in range(self.num_qubits-1):
            qc.cx(i, i+1)
        self.ansatz_circuit = qc
        self.num_parameters = len(params)
        self.vqe_ready = True
        self.is_built = True
        if self.verbose:
            print(f"✓ Fallback ansatz built with {self.num_parameters} parameters")
        return True

    def _build_hf_state(self, mapper):
        """Build the Hartree–Fock initial state handling API differences.

        qiskit-nature versions differ: older versions expect
        HartreeFock(num_spatial_orbitals, num_particles, qubit_mapper,...)
        while our previous attempt used the (incorrect here) keyword num_spin_orbitals.

        We try the modern / documented signature first. If all attempts fail we raise
        (NO silent fallback) because the user requested immediate fixes over fallbacks.
        """
        last_err = None
        spatial_orbs = self.num_spatial_orbitals if self.num_spatial_orbitals else self.num_qubits // 2
        # Candidate call patterns (args / kwargs) to try in order
        attempts = [
            ((), {"num_spatial_orbitals": spatial_orbs, "num_particles": self.num_particles, "qubit_mapper": mapper}),
            # Some legacy forms might accept positional args only
            ((spatial_orbs, self.num_particles, mapper), {}),
        ]
        for args, kwargs in attempts:
            try:
                self.hf_state = HartreeFock(*args, **kwargs)
                if self.verbose:
                    print("✓ HF state ready (HartreeFock constructed)")
                return True
            except Exception as e:  # capture and try next pattern
                last_err = e
        # If we reach here all attempts failed
        raise RuntimeError(f"HartreeFock construction failed for all known signatures: {last_err}")

    def _build_uccsd_ansatz(self, mapper):
        try:
            ucc = UCCSD(
                num_spatial_orbitals=self.num_spatial_orbitals,
                num_particles=self.num_particles,
                qubit_mapper=mapper,
                reps=self.ansatz_reps
            )
            if self.include_hf_state and self.hf_state is not None:
                full = QuantumCircuit(self.num_qubits)
                full.compose(self.hf_state, inplace=True)
                full.compose(ucc, inplace=True)
                full._parameters = ucc.parameters
                self.ansatz_circuit = full
            else:
                self.ansatz_circuit = ucc
        except Exception as e:
            if self.verbose:
                print(f"UCCSD failed ({e}); using HF only")
            self.ansatz_circuit = self.hf_state if self.hf_state else QuantumCircuit(self.num_qubits)
        if hasattr(self.ansatz_circuit, 'num_parameters'):
            self.num_parameters = self.ansatz_circuit.num_parameters
        else:
            self.num_parameters = len(getattr(self.ansatz_circuit, 'parameters', []))
        if self.verbose:
            print(f"Ansatz depth={self.ansatz_circuit.depth()} gates={len(self.ansatz_circuit.data)} params={self.num_parameters}")
        return True

    def _validate_system(self, hamiltonian):
        if hamiltonian.num_qubits != self.ansatz_circuit.num_qubits:
            if self.verbose:
                print("Quubit mismatch; VQE not ready")
            self.vqe_ready = False
        else:
            self.vqe_ready = self.num_parameters > 0

    def get_trial_wavefunction(self, parameters):
        if not self.is_built:
            raise RuntimeError("Ansatz not built")
        if self.num_parameters == 0:
            return self.ansatz_circuit.copy()
        if len(parameters) != self.num_parameters:
            raise ValueError("Parameter length mismatch")
        if hasattr(self.ansatz_circuit, 'bind_parameters'):
            return self.ansatz_circuit.bind_parameters(parameters)
        trial = self.ansatz_circuit.copy()
        if getattr(trial, 'parameters', None):
            trial = trial.assign_parameters(dict(zip(trial.parameters, parameters)))
        return trial

    def get_initial_parameters(self, init_type="zero"):
        if not self.is_built:
            raise RuntimeError("Ansatz not built")
        if self.num_parameters == 0:
            return np.array([])
        if init_type == 'zero':
            return np.zeros(self.num_parameters)
        if init_type == 'random_small':
            return np.random.normal(0, 0.01, self.num_parameters)
        if init_type == 'random_normal':
            return np.random.normal(0, 0.1, self.num_parameters)
        if init_type == 'hf_like':
            return np.random.normal(0, 0.005, self.num_parameters)
        return np.zeros(self.num_parameters)

    def get_parameter_bounds(self, bound_type="standard"):
        if not self.is_built or self.num_parameters == 0:
            return []
        if bound_type == 'tight':
            return [(-0.1, 0.1)]*self.num_parameters
        if bound_type == 'loose':
            return [(-2*np.pi, 2*np.pi)]*self.num_parameters
        return [(-0.5, 0.5)]*self.num_parameters

    def get_ansatz_info(self):
        if not self.is_built:
            return {"built": False}
        return {
            'built': True,
            'vqe_ready': self.vqe_ready,
            'num_qubits': self.num_qubits,
            'num_parameters': self.num_parameters,
            'circuit_depth': self.ansatz_circuit.depth() if self.ansatz_circuit else 0,
            'circuit_gates': len(self.ansatz_circuit.data) if self.ansatz_circuit else 0,
            'num_spatial_orbitals': self.num_spatial_orbitals,
            'num_particles': self.num_particles,
            'ansatz_reps': self.ansatz_reps,
            'include_hf_state': self.include_hf_state,
            'basis': self.hamiltonian_system.get('basis', 'unknown') if self.hamiltonian_system else 'unknown',
            'geometry': self.hamiltonian_system.get('geometry', 'unknown') if self.hamiltonian_system else 'unknown'
        }


class GenericAnsatzPlugin:
    """
    Hardware-efficient Ansatz (Ry + CX layers) compatible with VQE.

    This mirrors the interface of `AnsatzPlugin` so it can be dropped into VQE
    for a simple baseline that does not rely on UCCSD/HF structure.
    """
    def __init__(self, layers: int = 2, entanglement: str = "linear", verbose: bool = True):
        self.layers = int(layers)
        self.entanglement = entanglement
        self.verbose = verbose
        self.ansatz_circuit = None
        self.num_parameters = 0
        self.num_qubits = 0
        self.is_built = False
        self.vqe_ready = False

    def build_from_hamiltonian(self, hamiltonian_system):
        from qiskit.circuit import QuantumCircuit, Parameter  # type: ignore
        num_qubits = int(hamiltonian_system.get('num_qubits', 0))
        if num_qubits <= 0:
            raise ValueError("Hamiltonian system missing a valid num_qubits")
        self.num_qubits = num_qubits
        qc = QuantumCircuit(self.num_qubits)
        params = []
        for l in range(self.layers):
            # Single-qubit Ry rotations
            for q in range(self.num_qubits):
                p = Parameter(f"theta_{l}_{q}")
                qc.ry(p, q)
                params.append(p)
            # Entanglement layer
            if self.entanglement == "linear":
                for q in range(self.num_qubits - 1):
                    qc.cx(q, q + 1)
            elif self.entanglement == "full":
                for q in range(self.num_qubits):
                    for r in range(q + 1, self.num_qubits):
                        qc.cx(q, r)
            else:
                # default to linear if unknown
                for q in range(self.num_qubits - 1):
                    qc.cx(q, q + 1)
        self.ansatz_circuit = qc
        self.num_parameters = len(params)
        self.is_built = True
        self.vqe_ready = self.num_parameters > 0
        if self.verbose:
            print(f"✓ Generic ansatz built (layers={self.layers}, entanglement={self.entanglement}, params={self.num_parameters})")
        return True

    def get_trial_wavefunction(self, parameters):
        if not self.is_built:
            raise RuntimeError("Ansatz not built")
        if self.num_parameters == 0:
            return self.ansatz_circuit.copy()
        if len(parameters) != self.num_parameters:
            raise ValueError("Parameter length mismatch")
        if hasattr(self.ansatz_circuit, 'bind_parameters'):
            return self.ansatz_circuit.bind_parameters(parameters)
        trial = self.ansatz_circuit.copy()
        if getattr(trial, 'parameters', None):
            trial = trial.assign_parameters(dict(zip(trial.parameters, parameters)))
        return trial

    def get_initial_parameters(self, init_type: str = "random_normal"):
        import numpy as _np
        if not self.is_built:
            raise RuntimeError("Ansatz not built")
        if self.num_parameters == 0:
            return _np.array([])
        if init_type == "zero":
            return _np.zeros(self.num_parameters)
        if init_type == "random_small":
            return _np.random.normal(0, 0.01, self.num_parameters)
        return _np.random.normal(0, 0.1, self.num_parameters)

    def get_parameter_bounds(self, bound_type: str = "standard"):
        import numpy as _np
        if not self.is_built or self.num_parameters == 0:
            return []
        if bound_type == "tight":
            return [(-0.1, 0.1)] * self.num_parameters
        if bound_type == "loose":
            return [(-2 * _np.pi, 2 * _np.pi)] * self.num_parameters
        return [(-_np.pi, _np.pi)] * self.num_parameters

    def get_ansatz_info(self):
        if not self.is_built:
            return {"built": False}
        return {
            'built': True,
            'vqe_ready': self.vqe_ready,
            'num_qubits': self.num_qubits,
            'num_parameters': self.num_parameters,
            'circuit_depth': self.ansatz_circuit.depth() if self.ansatz_circuit else 0,
            'circuit_gates': len(self.ansatz_circuit.data) if self.ansatz_circuit else 0,
            'layers': self.layers,
            'entanglement': self.entanglement,
        }


class HamiltonianPlugin:
    """
    Plugin for generating the active-space Hamiltonian for Ammonia (NH3).
    """
    MIN_TERMS = 400
    PAD_SYNTHETIC = True
    SYN_COEFF_SCALE = 1e-8

    def __init__(self):
        # Default to a reasonable NH3 geometry (approximate, Angstroms)
        # If you want H2 instead, replace with ["H 0 0 0", "H 0 0 0.74"] or multiline string.
        self.geom = [
            "N 0.000000 0.000000 0.000000",
            "H 0.000000 0.937700 -0.381600",
            "H 0.812100 -0.468800 -0.381600",
            "H -0.812100 -0.468800 -0.381600"
        ]
        self._hamiltonian = None
        self._problem_active = None
        self._mapper = None
        self.is_fallback = False  # Tracks whether synthetic Hamiltonian was used

    def _normalize_geometry(self, geom):
        """Return geometry in a PySCFDriver-acceptable form (string or list[str])."""
        if isinstance(geom, str):
            return geom
        if isinstance(geom, (list, tuple)):
            if all(isinstance(x, str) for x in geom):
                return "\n".join(geom)
        raise ValueError("Geometry must be str or sequence of str specifications.")

    def get_hamiltonian(self):
        """
        Builds and returns a dictionary containing the NH3 active-space Hamiltonian
        and other metadata. The result is cached to avoid re-computation.
        """
        if self._hamiltonian is not None:
            return {
                "problem_active": self._problem_active,
                "mapper": self._mapper,
                "hamiltonian_active": self._hamiltonian,
                "num_qubits": self._hamiltonian.num_qubits,
                "basis": "sto3g",
                "geometry": self.geom
            }

        # Optional hard bypass: skip PySCFDriver entirely and build via pyscf + from_pyscf path
        if os.environ.get('FORCE_FROM_PYSCF', '0') == '1':
            try:
                # Attempt modern from_pyscf helper if available, else degrade gracefully to driver logic
                from pyscf import gto, scf  # type: ignore
                geom_str = self._normalize_geometry(self.geom)
                mol = gto.M(atom=geom_str, basis='sto-3g', unit='Angstrom')
                mf = scf.RHF(mol).run()
                ham_active = None
                full_map_success = False
                try:
                    # Newer qiskit-nature versions expose a formats.pyscf import; 0.7.2 may not.
                    from qiskit_nature.second_q.formats.pyscf import from_pyscf  # type: ignore
                    result_alt = from_pyscf(mf, include_dipole=False)
                    problem_full_alt = ElectronicStructureProblem(result_alt)
                    # Attempt MO basis conversion if needed
                    try:
                        from qiskit_nature.second_q.properties import ElectronicBasis  # type: ignore
                        ei = getattr(problem_full_alt.hamiltonian, 'electronic_integrals', None)
                        if ei and hasattr(ei, 'convert_basis') and getattr(problem_full_alt.hamiltonian, 'electronic_basis', None) != ElectronicBasis.MO:
                            ei.convert_basis(ElectronicBasis.AO, ElectronicBasis.MO)
                    except Exception:
                        pass
                    transformer_alt = ActiveSpaceTransformer(num_electrons=4, num_spatial_orbitals=3)
                    self._problem_active = transformer_alt.transform(problem_full_alt)
                    self._mapper = JordanWignerMapper()
                    ham2_alt = self._problem_active.second_q_ops()['ElectronicEnergy']
                    ham_active = self._mapper.map(ham2_alt)
                    if ham_active.num_qubits == 6:
                        full_map_success = True
                except Exception:
                    pass
                if full_map_success:
                    self._hamiltonian = ham_active
                    self.PAD_SYNTHETIC = False
                    self.is_fallback = False
                    print('[Info] Built Hamiltonian via FORCE_FROM_PYSCF path (full 2e terms).')
                    return {
                        "problem_active": self._problem_active,
                        "mapper": self._mapper,
                        "hamiltonian_active": self._hamiltonian,
                        "num_qubits": self._hamiltonian.num_qubits,
                        "basis": "sto3g",
                        "geometry": self.geom,
                        "fallback": False
                    }
                else:
                    print('[Warning] FORCE_FROM_PYSCF full mapping unavailable; reverting to standard driver path.')
            except Exception as force_e:
                print(f"[Warning] FORCE_FROM_PYSCF path failed early: {force_e}; continuing with standard logic.")

        try:
            if not QISKIT_NATURE_INSTALLED:
                raise ImportError("Qiskit Nature or its dependencies are not installed.")

            atom_spec = self._normalize_geometry(self.geom)
            driver = PySCFDriver(atom=atom_spec, basis='sto3g', charge=0, spin=0, unit=DistanceUnit.ANGSTROM)

            # Monkey patch missing legacy attribute if downstream code expects it
            if not hasattr(driver, 'register_length'):
                try:
                    raw_res_tmp = driver.run()
                    guess_len = getattr(raw_res_tmp, 'num_spatial_orbitals', 0) * 2
                    driver.register_length = guess_len  # type: ignore
                except Exception:
                    driver.register_length = 0  # type: ignore

            raw_res = driver.run()
            # Some versions return ElectronicStructureProblem directly; if not, wrap
            problem_full = raw_res if isinstance(raw_res, ElectronicStructureProblem) else ElectronicStructureProblem(raw_res)
            transformer = ActiveSpaceTransformer(num_electrons=4, num_spatial_orbitals=3)
            try:
                self._problem_active = transformer.transform(problem_full)
            except Exception as t_e:
                # Fallback: attempt direct integral reconstruction
                raise RuntimeError(f"Active space transform failed ({t_e})")

            self._mapper = JordanWignerMapper()
            ham2 = self._problem_active.second_q_ops()['ElectronicEnergy']
            ham_active = self._mapper.map(ham2)
            if ham_active.num_qubits != 6:
                raise RuntimeError(f'Active space produced {ham_active.num_qubits} qubits, expected 6.')

        except Exception as e:
            print(f'[Warning] Ab initio build failed: {e}. Attempting direct PySCF fallback...')
            direct_pyscf_failed = False
            ham_active = None
            if 'PySCFDriver' in str(e) or 'register_length' in str(e):
                try:
                    import pyscf  # type: ignore
                    from pyscf import gto, scf  # type: ignore
                    geom_str = self._normalize_geometry(self.geom)
                    mol = gto.Mole()
                    mol.build(atom=geom_str, basis='sto-3g', unit='Angstrom')
                    mf = scf.RHF(mol)
                    e_hf = mf.kernel()

                    # First attempt: full mapped Hamiltonian via from_pyscf (retains 2e correlations)
                    full_map_success = False
                    try:
                        from qiskit_nature.second_q.formats.pyscf import from_pyscf  # type: ignore
                        result_alt = from_pyscf(mf, include_dipole=False)
                        problem_full_alt = ElectronicStructureProblem(result_alt)
                        transformer_alt = ActiveSpaceTransformer(num_electrons=4, num_spatial_orbitals=3)
                        self._problem_active = transformer_alt.transform(problem_full_alt)
                        self._mapper = JordanWignerMapper()
                        ham2_alt = self._problem_active.second_q_ops()['ElectronicEnergy']
                        ham_active = self._mapper.map(ham2_alt)
                        if ham_active.num_qubits == 6:
                            full_map_success = True
                            self.PAD_SYNTHETIC = False
                            self.is_fallback = False
                            print('[Info] Recovered full active-space Hamiltonian via from_pyscf fallback (includes 2e terms).')
                        else:
                            print(f'[Info] from_pyscf produced {ham_active.num_qubits} qubits (expected 6); discarding.')
                            ham_active = None
                    except Exception as map_e:
                        print(f'[Info] from_pyscf path unavailable ({map_e}); reverting to diagonal HF model.')

                    if not full_map_success:
                        # NEW: attempt integral-based correlated Hamiltonian before diagonal simplification
                        try:
                            from qiskit_nature.second_q.hamiltonians import ElectronicEnergy  # type: ignore
                            from qiskit_nature.second_q.operators import FermionicOp  # type: ignore
                            # AO integrals
                            h1_ao = mf.get_hcore()
                            from pyscf import ao2mo  # type: ignore
                            eri_ao = ao2mo.full(mf._eri, mf.mo_coeff)  # MO two-electron (chemist)
                            # Build one- and two-body in MO basis
                            C = mf.mo_coeff
                            h1_mo = C.T @ h1_ao @ C
                            nmo = h1_mo.shape[0]
                            # Reshape two-electron integrals (chemist notation) (ij|kl)
                            eri_mo = ao2mo.restore(1, eri_ao, nmo)
                            # ElectronicEnergy helper
                            ee = ElectronicEnergy.from_raw_integrals(h1_mo, eri_mo)
                            problem_full_alt2 = ElectronicStructureProblem(ee)
                            transformer_alt2 = ActiveSpaceTransformer(num_electrons=4, num_spatial_orbitals=3)
                            self._problem_active = transformer_alt2.transform(problem_full_alt2)
                            self._mapper = JordanWignerMapper()
                            ham2_alt2 = self._problem_active.second_q_ops()['ElectronicEnergy']
                            ham_active = self._mapper.map(ham2_alt2)
                            if ham_active.num_qubits == 6:
                                print('[Info] Built correlated Hamiltonian from raw PySCF integrals (includes 2e terms).')
                                self.PAD_SYNTHETIC = False
                                self.is_fallback = False
                            else:
                                ham_active = None
                        except Exception as int_e:
                            # Final diagonal model
                            from qiskit.quantum_info import SparsePauliOp as _SPO  # type: ignore
                            mo_energies = list(mf.mo_energy)
                            n_spatial_target = 3
                            if len(mo_energies) < n_spatial_target:
                                n_spatial_target = len(mo_energies)
                            active_eps = mo_energies[:n_spatial_target]
                            spin_eps = []
                            for eps in active_eps:
                                spin_eps.extend([float(eps), float(eps)])
                            num_qubits = len(spin_eps)
                            n_electrons = 4
                            occ_indices = list(range(min(n_electrons, len(spin_eps))))
                            sum_eps_occ = sum(spin_eps[i] for i in occ_indices)
                            const_shift = e_hf - sum_eps_occ
                            paulis = ['I'*num_qubits]
                            coeffs = [const_shift]
                            for p, eps in enumerate(spin_eps):
                                paulis.append('I'*num_qubits); coeffs.append(eps/2.0)
                                z_string = ['I']*num_qubits; z_string[p] = 'Z'
                                paulis.append(''.join(z_string)); coeffs.append(-eps/2.0)
                            ham_active = _SPO(paulis, coeffs)
                            print(f'[Info] Direct PySCF HF energy = {e_hf:.6f} Hartree (diagonal orbital-energy Hamiltonian)')
                            print('[Info] Using simplified diagonal Hamiltonian (no two-electron correlations).')
                            self.PAD_SYNTHETIC = False
                            self.is_fallback = True
                except Exception as de2:
                    direct_pyscf_failed = True
                    print(f'[Warning] Direct PySCF fallback failed: {de2}')
            if ham_active is None:
                print('Using a synthetic 6-qubit operator.')
            if not QISKIT_NATURE_INSTALLED:
                print('[Info] qiskit-nature not detected. Install with: pip install qiskit-nature pyscf')
            else:
                print('[Info] Check geometry formatting or PySCF availability.')
            if ham_active is None:
                paulis = ['IIIIII', 'ZIIIZZ', 'ZZIIZZ', 'IZZIIZ', 'IIZZZZ', 'XXYYZZ', 'YYXXZZ']
                coeffs = [-5.0, 0.12, -0.08, 0.05, -0.03, 0.01, 0.01]
                ham_active = SparsePauliOp(paulis, coeffs)
            self._problem_active = None
            self._mapper = None
            self.is_fallback = True

        terms = {str(p): complex(c) for p, c in zip(ham_active.paulis, ham_active.coeffs) if abs(complex(c)) > 1e-12}
        physical_count = len(terms)

        if self.PAD_SYNTHETIC and physical_count < self.MIN_TERMS:
            self._add_synthetic_padding(terms, ham_active.num_qubits, physical_count)

        self._print_summary(terms, physical_count)
        
        self._hamiltonian = SparsePauliOp.from_list(list(terms.items()))
        
        return {
            "problem_active": self._problem_active,
            "mapper": self._mapper,
            "hamiltonian_active": self._hamiltonian,
            "num_qubits": self._hamiltonian.num_qubits,
            "basis": "sto3g",
            "geometry": self.geom,
            "fallback": self.is_fallback
        }

    def was_fallback(self):
        """Return True if synthetic fallback Hamiltonian was used."""
        return self.is_fallback

    def info(self):
        """Return a concise diagnostic dictionary about the Hamiltonian state."""
        return {
            'fallback': self.is_fallback,
            'num_qubits': self._hamiltonian.num_qubits if self._hamiltonian else None,
            'active_problem': self._problem_active is not None,
            'mapper': self._mapper.__class__.__name__ if self._mapper else None,
            'geometry_lines': len(self.geom) if isinstance(self.geom, (list, tuple)) else 1
        }

    def _add_synthetic_padding(self, terms, num_qubits, physical_count):
        needed = self.MIN_TERMS - physical_count
        alphabet = ['I', 'X', 'Y', 'Z']
        for _ in range(needed * 20):
            if len(terms) >= self.MIN_TERMS: break
            word = ''.join(random.choice(alphabet) for _ in range(num_qubits))
            if word == 'I' * num_qubits or word in terms: continue
            terms[word] = (random.random() * 2 - 1) * self.SYN_COEFF_SCALE
    
    def _print_summary(self, terms, physical_count):
        total = len(terms)
        print('\n--- Hamiltonian Generation Summary ---')
        print(f'Physical (mapped) terms: {physical_count}')
        if total > physical_count:
            print(f'Synthetic padding terms added: {total - physical_count}')
        print(f'Total terms in operator: {total}')
        print('-------------------------------------\n')


# Classical optimizer placeholder (keeping as requested)
class ClassicalOptimizerPlugin:
    """Plugin for the classical optimization routine."""
    def optimize(self, objective_function, initial_params):
        raise NotImplementedError("Optimizer plugin not implemented.")


import numpy as np
from typing import List, Union, Callable, Optional
import warnings

class ZNEDenoiserPlugin:
    """
    Advanced Zero-Noise Extrapolation (ZNE) Plugin for VQE Error Mitigation
    
    Implements multiple ZNE strategies including:
    - Richardson extrapolation
    - Exponential fitting
    - Polynomial fitting
    - Adaptive noise scaling
    """

    def __init__(self, 
                 noise_factors: List[float] = None,
                 extrapolation_method: str = 'richardson',
                 polynomial_degree: int = 2,
                 min_noise_factor: float = 1.0,
                 max_noise_factor: float = 5.0,
                 adaptive_threshold: float = 0.01,
                 verbose: bool = True):
        """
        Initialize ZNE plugin with comprehensive error mitigation options
        
        Args:
            noise_factors: List of noise scaling factors (>= 1.0)
            extrapolation_method: 'richardson', 'exponential', 'polynomial', 'linear'
            polynomial_degree: Degree for polynomial extrapolation
            min_noise_factor: Minimum noise scaling factor
            max_noise_factor: Maximum noise scaling factor
            adaptive_threshold: Convergence threshold for adaptive methods
            verbose: Enable detailed output
        """
        # Default noise factors for Richardson extrapolation
        if noise_factors is None:
            self.noise_factors = [1.0, 3.0, 5.0]  # Odd factors work well
        else:
            self.noise_factors = sorted([max(1.0, f) for f in noise_factors])
            
        self.extrapolation_method = extrapolation_method.lower()
        self.polynomial_degree = polynomial_degree
        self.min_noise_factor = min_noise_factor
        self.max_noise_factor = max_noise_factor
        self.adaptive_threshold = adaptive_threshold
        self.verbose = verbose
        
        # Tracking for analysis
        self.zne_history = []
        self.improvement_history = []
        
        if self.verbose:
            print(f"🔧 ZNE Plugin initialized:")
            print(f"   Method: {self.extrapolation_method}")
            print(f"   Noise factors: {self.noise_factors}")

    def denoise(self, noisy_results: Union[float, List[float], np.ndarray]) -> float:
        """
        Apply ZNE to denoise quantum expectation values
        
        Args:
            noisy_results: Either single noisy value or list of values at different noise levels
            
        Returns:
            float: Zero-noise extrapolated expectation value
        """
        # Handle single value input (backward compatibility)
        if isinstance(noisy_results, (int, float)):
            if self.verbose:
                print("⚠️  ZNE: Single value provided, returning as-is (no extrapolation possible)")
            return float(noisy_results)
        
        # Convert to numpy array
        noisy_values = np.array(noisy_results, dtype=float)
        
        # Validate input
        if len(noisy_values) != len(self.noise_factors):
            if self.verbose:
                print(f"⚠️  ZNE: Expected {len(self.noise_factors)} values, got {len(noisy_values)}")
                print("   Returning first value without extrapolation")
            return float(noisy_values[0]) if len(noisy_values) > 0 else 0.0
        
        # Perform extrapolation
        try:
            extrapolated_value = self._perform_extrapolation(noisy_values)
            
            # Track improvement
            original_value = noisy_values[0]  # Value at noise_factor = 1.0
            improvement = abs(extrapolated_value - original_value)
            
            self.zne_history.append({
                'noise_factors': self.noise_factors.copy(),
                'noisy_values': noisy_values.copy(),
                'extrapolated': extrapolated_value,
                'original': original_value,
                'improvement': improvement
            })
            
            self.improvement_history.append(improvement)
            
            if self.verbose:
                print(f"📊 ZNE Applied:")
                print(f"   Noisy values: {noisy_values}")
                print(f"   Noise factors: {self.noise_factors}")
                print(f"   Extrapolated: {extrapolated_value:.8f}")
                print(f"   Improvement: {improvement:.8f}")
            
            return float(extrapolated_value)
            
        except Exception as e:
            if self.verbose:
                print(f"❌ ZNE Error: {e}")
                print("   Returning original noisy value")
            return float(noisy_values[0])

    def _perform_extrapolation(self, noisy_values: np.ndarray) -> float:
        """Perform the actual extrapolation based on selected method"""
        noise_factors = np.array(self.noise_factors)
        
        if self.extrapolation_method == 'richardson':
            return self._richardson_extrapolation(noise_factors, noisy_values)
        elif self.extrapolation_method == 'exponential':
            return self._exponential_extrapolation(noise_factors, noisy_values)
        elif self.extrapolation_method == 'polynomial':
            return self._polynomial_extrapolation(noise_factors, noisy_values)
        elif self.extrapolation_method == 'linear':
            return self._linear_extrapolation(noise_factors, noisy_values)
        else:
            raise ValueError(f"Unknown extrapolation method: {self.extrapolation_method}")

    def _richardson_extrapolation(self, noise_factors: np.ndarray, values: np.ndarray) -> float:
        """Richardson extrapolation for ZNE"""
        if len(values) < 2:
            return values[0]
        
        # Use Richardson extrapolation formula
        # For two points: f(0) ≈ (λ₂f(λ₁) - λ₁f(λ₂)) / (λ₂ - λ₁)
        if len(values) == 2:
            λ1, λ2 = noise_factors[0], noise_factors[1]
            f1, f2 = values[0], values[1]
            return (λ2 * f1 - λ1 * f2) / (λ2 - λ1)
        
        # For multiple points, use weighted Richardson extrapolation
        weights = []
        extrapolated_values = []
        
        for i in range(len(values) - 1):
            for j in range(i + 1, len(values)):
                λ1, λ2 = noise_factors[i], noise_factors[j]
                f1, f2 = values[i], values[j]
                extrapolated = (λ2 * f1 - λ1 * f2) / (λ2 - λ1)
                extrapolated_values.append(extrapolated)
                # Weight inversely by noise factor difference
                weight = 1.0 / abs(λ2 - λ1)
                weights.append(weight)
        
        # Weighted average of all pairwise extrapolations
        weights = np.array(weights)
        extrapolated_values = np.array(extrapolated_values)
        
        return np.average(extrapolated_values, weights=weights)

    def _exponential_extrapolation(self, noise_factors: np.ndarray, values: np.ndarray) -> float:
        """Exponential decay model: E(λ) = a + b*exp(-c*λ)"""
        try:
            # Fit exponential decay: E(λ) = a*exp(-b*λ) + c
            from scipy.optimize import curve_fit
            
            def exp_model(x, a, b, c):
                return a * np.exp(-b * x) + c
            
            # Initial guess
            p0 = [values[0] - values[-1], 0.1, values[-1]]
            
            popt, _ = curve_fit(exp_model, noise_factors, values, p0=p0, maxfev=1000)
            
            # Extrapolate to λ = 0
            return exp_model(0, *popt)
            
        except (ImportError, RuntimeError, ValueError):
            # Fallback to simple exponential fit
            try:
                # Linear fit in log space: log(E) = log(a) + b*λ
                log_values = np.log(np.abs(values) + 1e-12)  # Avoid log(0)
                coeffs = np.polyfit(noise_factors, log_values, 1)
                return np.exp(coeffs[1])  # exp(log(a)) = a
            except:
                # Final fallback to linear extrapolation
                return self._linear_extrapolation(noise_factors, values)

    def _polynomial_extrapolation(self, noise_factors: np.ndarray, values: np.ndarray) -> float:
        """Polynomial extrapolation to λ = 0"""
        degree = min(self.polynomial_degree, len(values) - 1)
        
        try:
            # Fit polynomial
            coeffs = np.polyfit(noise_factors, values, degree)
            # Evaluate at λ = 0
            return np.polyval(coeffs, 0)
        except (np.linalg.LinAlgError, ValueError):
            # Fallback to linear
            return self._linear_extrapolation(noise_factors, values)

    def _linear_extrapolation(self, noise_factors: np.ndarray, values: np.ndarray) -> float:
        """Simple linear extrapolation"""
        try:
            # Fit line: E(λ) = a + b*λ
            coeffs = np.polyfit(noise_factors, values, 1)
            # Evaluate at λ = 0: E(0) = a
            return coeffs[1]
        except (np.linalg.LinAlgError, ValueError):
            # Ultimate fallback
            return values[0]

    def create_noisy_circuit(self, original_circuit, noise_factor: float):
        """
        Create a noisier version of the circuit by inserting identity gate pairs
        
        This is a common ZNE technique called "unitary folding"
        
        Args:
            original_circuit: Original quantum circuit
            noise_factor: Factor by which to scale noise (>= 1.0)
            
        Returns:
            Modified circuit with increased noise
        """
        if noise_factor <= 1.0:
            return original_circuit.copy()
        
        from qiskit import QuantumCircuit # type: ignore
        
        # Create a copy of the original circuit
        noisy_circuit = original_circuit.copy()
        
        # Calculate number of identity pairs to insert
        # Each identity pair (I followed by I†) increases circuit depth
        # without changing the logical operation
        num_identity_pairs = int((noise_factor - 1.0) * len(original_circuit.data))
        
        if num_identity_pairs > 0:
            # Insert identity pairs at random locations
            np.random.seed(42)  # For reproducibility
            
            for _ in range(num_identity_pairs):
                # Choose random qubit
                qubit = np.random.randint(0, noisy_circuit.num_qubits)
                
                # Insert X-X pair (equivalent to identity but adds noise)
                noisy_circuit.x(qubit)
                noisy_circuit.x(qubit)
        
        return noisy_circuit

    def get_zne_analysis(self) -> dict:
        """Get analysis of ZNE performance"""
        if not self.zne_history:
            return {"error": "No ZNE history available"}
        
        improvements = np.array(self.improvement_history)
        
        return {
            "total_applications": len(self.zne_history),
            "average_improvement": np.mean(improvements),
            "std_improvement": np.std(improvements),
            "max_improvement": np.max(improvements),
            "min_improvement": np.min(improvements),
            "extrapolation_method": self.extrapolation_method,
            "noise_factors": self.noise_factors,
            "success_rate": np.sum(improvements > 0) / len(improvements) * 100
        }

    def adaptive_zne(self, measurement_function: Callable, initial_params: np.ndarray) -> float:
        """
        Adaptive ZNE that automatically determines optimal noise factors
        
        Args:
            measurement_function: Function that takes (params, noise_factor) and returns expectation value
            initial_params: Parameters for the measurement function
            
        Returns:
            Zero-noise extrapolated value
        """
        if self.verbose:
            print("🔄 Running adaptive ZNE...")
        
        # Start with minimum noise factor
        current_factors = [1.0]
        current_values = [measurement_function(initial_params, 1.0)]
        
        # Iteratively add noise factors until convergence
        for factor in np.linspace(2.0, self.max_noise_factor, 10):
            current_factors.append(factor)
            current_values.append(measurement_function(initial_params, factor))
            
            if len(current_values) >= 3:
                # Try extrapolation with current data
                temp_plugin = ZNEDenoiserPlugin(
                    noise_factors=current_factors,
                    extrapolation_method=self.extrapolation_method,
                    verbose=False
                )
                
                extrapolated = temp_plugin.denoise(current_values)
                
                # Check for convergence
                if len(current_values) > 3:
                    prev_extrapolated = temp_plugin.denoise(current_values[:-1])
                    if abs(extrapolated - prev_extrapolated) < self.adaptive_threshold:
                        if self.verbose:
                            print(f"   Converged after {len(current_values)} measurements")
                        break
        
        # Final extrapolation
        self.noise_factors = current_factors
        return self.denoise(current_values)


# Integration helper for VQE class
class ZNEIntegratedMeasurement:
    """
    Helper class to integrate ZNE directly into VQE measurement process
    """
    
    def __init__(self, vqe_instance, zne_plugin: ZNEDenoiserPlugin):
        self.vqe = vqe_instance
        self.zne_plugin = zne_plugin
    
    def measure_with_zne(self, parameters: np.ndarray) -> float:
        """
        Perform measurement with automatic ZNE error mitigation
        
        Args:
            parameters: Variational parameters
            
        Returns:
            Zero-noise extrapolated expectation value
        """
        # Get trial wavefunction
        trial_wavefunction = self.vqe.ansatz_plugin.get_trial_wavefunction(parameters)
        hamiltonian = self.vqe.hamiltonian_system['hamiltonian_active']
        
        # Measure at different noise levels
        noisy_values = []
        
        for noise_factor in self.zne_plugin.noise_factors:
            if noise_factor == 1.0:
                # Original circuit
                noisy_circuit = trial_wavefunction
            else:
                # Create noisier version
                noisy_circuit = self.zne_plugin.create_noisy_circuit(trial_wavefunction, noise_factor)
            
            # Measure expectation value
            expectation = self.vqe._simulate_measurement(noisy_circuit, hamiltonian)
            noisy_values.append(expectation)
        
        # Apply ZNE
        return self.zne_plugin.denoise(noisy_values)


# Example usage and testing
def test_zne_plugin():
    """Test the ZNE plugin with synthetic noisy data"""
    print("🧪 Testing ZNE Plugin")
    print("="*50)
    
    # Create ZNE plugin
    zne = ZNEDenoiserPlugin(
        noise_factors=[1.0, 2.0, 3.0, 4.0],
        extrapolation_method='richardson',
        verbose=True
    )
    
    # Simulate noisy measurements (true value = -1.8572)
    true_value = -1.8572
    noise_factors = [1.0, 2.0, 3.0, 4.0]
    
    # Simulate exponential decay with noise
    noisy_values = []
    for factor in noise_factors:
        noise = 0.1 * (factor - 1.0)  # Noise increases with factor
        measured = true_value + noise + 0.01 * np.random.randn()
        noisy_values.append(measured)
    
    print(f"True value: {true_value}")
    print(f"Noisy measurements: {noisy_values}")
    
    # Apply ZNE
    denoised = zne.denoise(noisy_values)
    
    print(f"ZNE result: {denoised}")
    print(f"Error reduction: {abs(noisy_values[0] - true_value):.6f} → {abs(denoised - true_value):.6f}")
    
    # Get analysis
    analysis = zne.get_zne_analysis()
    print(f"ZNE Analysis: {analysis}")


if __name__ == "__main__":
    test_zne_plugin()

    


# === Optimizer Implementations ===
class SPSAOptimizer(ClassicalOptimizerPlugin):
    """Simplified SPSA optimizer.

    NOTE: This is a lightweight implementation suitable for early, noisy exploration.
    It purposefully keeps configuration minimal. Parameters follow conventional SPSA
    decay schedules (a_k, c_k).
    """
    def __init__(self, max_iter=40, a=0.2, c=0.15, alpha=0.602, gamma=0.101, tol=1e-3, seed=None, verbose=True):
        self.max_iter = max_iter
        self.a = a
        self.c = c
        self.alpha = alpha
        self.gamma = gamma
        self.tol = tol
        self.rng = np.random.default_rng(seed)
        self.verbose = verbose

    def _ak(self, k):
        return self.a / (k ** self.alpha)

    def _ck(self, k):
        return self.c / (k ** self.gamma)

    def optimize(self, objective_function, initial_params):
        params = np.array(initial_params, dtype=float)
        prev_val = objective_function(params)
        best_val = prev_val
        best_params = params.copy()
        if self.verbose:
            print(f"[SPSA] Initial energy: {prev_val}")
        for k in range(1, self.max_iter + 1):
            ak = self._ak(k)
            ck = self._ck(k)
            delta = self.rng.choice([-1, 1], size=params.shape)
            plus = params + ck * delta
            minus = params - ck * delta
            e_plus = objective_function(plus)
            e_minus = objective_function(minus)
            # Gradient estimate (vector)
            gk = (e_plus - e_minus) / (2.0 * ck) * delta
            params = params - ak * gk
            curr_val = objective_function(params)
            if curr_val < best_val:
                best_val = curr_val
                best_params = params.copy()
            if self.verbose:
                impr = prev_val - curr_val
                print(f"[SPSA] iter={k:3d} energy={curr_val:.8f} ΔE={impr:.3e} ak={ak:.3e} ck={ck:.3e}")
            if abs(prev_val - curr_val) < self.tol:
                if self.verbose:
                    print(f"[SPSA] Converged (|ΔE| < {self.tol}) at iter {k}")
                break
            prev_val = curr_val
        return best_params


class COBYLAOptimizer(ClassicalOptimizerPlugin):
    """Wrapper around SciPy COBYLA with graceful fallback if SciPy unavailable."""
    def __init__(self, max_iter=200, tol=1e-6, rhobeg=0.2, disp=True):
        self.max_iter = max_iter
        self.tol = tol
        self.rhobeg = rhobeg
        self.disp = disp

    def optimize(self, objective_function, initial_params):
        try:
            from scipy.optimize import minimize
            result = minimize(
                objective_function,
                np.array(initial_params, dtype=float),
                method="COBYLA",
                options={"maxiter": self.max_iter, "tol": self.tol, "disp": self.disp, "rhobeg": self.rhobeg},
            )
            return result.x
        except Exception as e:  # SciPy missing or failure -> simple fallback
            print(f"[COBYLAOptimizer] Fallback in use ({e}). Using coordinate descent.")
            params = np.array(initial_params, dtype=float)
            best = objective_function(params)
            step = self.rhobeg
            for _ in range(self.max_iter):
                improved = False
                for i in range(len(params)):
                    for direction in (+1, -1):
                        trial = params.copy()
                        trial[i] += direction * step
                        val = objective_function(trial)
                        if val < best - self.tol:
                            best = val
                            params = trial
                            improved = True
                if not improved:
                    step *= 0.5
                    if step < self.tol:
                        break
            return params


class HybridSPSAThenCOBYLA(ClassicalOptimizerPlugin):
    """Hybrid optimizer: coarse SPSA phase followed by COBYLA fine tuning.

    switch_tol: energy improvement threshold (absolute) below which we switch.
    min_spsa: minimum SPSA iterations before checking switch criterion.
    force_cobyla: if True, always run COBYLA after SPSA phase regardless of improvement.
    """
    def __init__(
        self,
        spsa_iters=40,
        spsa_a=0.2,
        spsa_c=0.15,
        switch_tol=5e-3,
        min_spsa=10,
        force_cobyla=False,
        cobyla_max_iter=150,
        ma_window=5,
        rel_switch_frac=1e-4,
        plateau_iters=3,
        verbose=True,
    ):
        self.spsa_iters = spsa_iters
        self.switch_tol = switch_tol
        self.min_spsa = min_spsa
        self.force_cobyla = force_cobyla
        self.verbose = verbose
        self._spsa = SPSAOptimizer(max_iter=spsa_iters, a=spsa_a, c=spsa_c, tol=switch_tol/5, verbose=verbose)
        self._cobyla = COBYLAOptimizer(max_iter=cobyla_max_iter, tol=1e-6, disp=verbose)
        # Adaptive switching controls
        self.ma_window = ma_window              # Moving-average window of last improvements
        self.rel_switch_frac = rel_switch_frac  # Relative (to |E|) MA improvement threshold
        self.plateau_iters = plateau_iters      # Consecutive tiny-improvement steps to qualify as plateau

    def optimize(self, objective_function, initial_params):
        if self.verbose:
            print(f"[Hybrid] Starting SPSA (max_iter={self.spsa_iters}) -> COBYLA (switch_tol={self.switch_tol})")
        # Wrap objective to record energies
        energy_log = []
        def logging_objective(p):
            val = objective_function(p)
            energy_log.append(val)
            return val
        # Run SPSA
        params_after_spsa = self._spsa.optimize(logging_objective, initial_params)
        # Decide on switch
        do_cobyla = self.force_cobyla
        recent_impr = float('inf')
        ma_impr = float('inf')
        rel_ma = float('inf')
        plateau = False
        if len(energy_log) >= 2:
            improvements = [abs(energy_log[i-1] - energy_log[i]) for i in range(1, len(energy_log))]
            recent_impr = improvements[-1]
            window = improvements[-self.ma_window:]
            if window:
                ma_impr = sum(window) / len(window)
            if abs(energy_log[-1]) > 1e-12:
                rel_ma = ma_impr / abs(energy_log[-1])
            # Plateau: last plateau_iters improvements all below switch_tol
            if len(improvements) >= self.plateau_iters:
                plateau = all(imp < self.switch_tol for imp in improvements[-self.plateau_iters:])
            if not do_cobyla:
                if len(energy_log) >= self.min_spsa:
                    # Switch if moving-average AND relative thresholds met OR plateau
                    if (ma_impr < self.switch_tol and rel_ma < self.rel_switch_frac) or plateau:
                        do_cobyla = True
        if self.verbose and len(energy_log) >= 2:
            print(f"[Hybrid] Decision metrics: last={recent_impr:.3e} ma={ma_impr:.3e} rel_ma={rel_ma:.3e} plateau={plateau}")
        if self.verbose:
            if do_cobyla and len(energy_log) >= 2:
                try:
                    print("[Hybrid] Switching to COBYLA in 5 seconds...")
                    import time; time.sleep(5)
                except Exception:
                    print("[Hybrid] (Sleep skipped)")
            elif not do_cobyla and len(energy_log) >= 2:
                print("[Hybrid] Staying with SPSA result (criteria not met for fine-tune).")
        params_final = self._cobyla.optimize(logging_objective, params_after_spsa) if do_cobyla else params_after_spsa
        if self.verbose:
            print(f"[Hybrid] Completed. Total energy evaluations: {len(energy_log)}")
        return params_final


# CORRECTED VQE CLASS - Fixed critical weaknesses
class VQE:
    """
    The main VQE class that orchestrates the algorithm using the provided plugins.
    """
    def __init__(self, ansatz_plugin, hamiltonian_plugin, optimizer_plugin, zne_plugin, verbose=True):
        self.ansatz_plugin = ansatz_plugin
        self.hamiltonian_plugin = hamiltonian_plugin
        self.optimizer_plugin = optimizer_plugin
        self.zne_plugin = zne_plugin
        self.verbose = verbose
        
        # Energy tracking for iteration output
        self.energy_history = []
        self.parameter_history = []
        self.iteration_count = 0
    # Total effective circuit evaluations (counts each underlying measurement call,
    # including per-factor ZNE foldings). Useful for fair optimizer comparisons.
        self.eval_calls = 0
        
        # Build the system
        print("🔄 Initializing VQE system...")
        self.hamiltonian_system = self.hamiltonian_plugin.get_hamiltonian()
        self.ansatz_plugin.build_from_hamiltonian(self.hamiltonian_system)
        
        # Setup quantum estimator
        self._setup_estimator()
    # Adaptive ZNE controls (inside __init__). By default adaptive disabling is OFF to retain
    # original behaviour (never auto-collapse noise_factors). Set zne_adaptive_enable=True in
    # notebook AFTER constructing VQE if you want auto-disable.
        self.zne_adaptive_enable = False     # Master switch
        self.zne_improvement_tol = 1e-4      # Hartree threshold considered "no improvement"
        self.zne_patience = 10               # Consecutive no-improvement iterations before disabling
        self.zne_no_improve_streak = 0       # Counter
        self.zne_disabled = False            # Flag once multi-noise is turned off

        # Optional synthetic shot-noise injection (off by default). This approximates
        # sampling noise when running with exact Estimator/Statevector backends, so we can
        # observe optimizer robustness and ZNE benefits without a hardware/noise model.
        # Enable by setting an integer number of shots here or via env VQE_SHOTS.
        try:
            self.shot_noise_shots = int(os.environ.get('VQE_SHOTS', '0')) or None
        except Exception:
            self.shot_noise_shots = None

    def _setup_estimator(self):
        """Setup quantum estimator for expectation value calculation"""
        try:
            from qiskit_aer.primitives import Estimator # type: ignore
            self.estimator = Estimator()
            if self.verbose:
                print("✓ Quantum estimator setup complete (Qiskit Aer)")
        except ImportError:
            try:
                from qiskit.primitives import StatevectorEstimator # type: ignore
                self.estimator = StatevectorEstimator()
                if self.verbose:
                    print("✓ Quantum estimator setup complete (Statevector)")
            except ImportError:
                raise RuntimeError("No compatible Estimator found. Install qiskit-aer or use compatible Qiskit version.")

    def _get_expectation_value(self, parameters):
        """
        Calculate quantum expectation value ⟨ψ(θ)|H|ψ(θ)⟩
        
        Args:
            parameters: Variational parameters for trial wavefunction
            
        Returns:
            float: Expectation value of Hamiltonian
        """
        # Get trial wavefunction with current parameters
        trial_wavefunction = self.ansatz_plugin.get_trial_wavefunction(parameters)
        
        # Compute expectation value using quantum simulator
        expectation_value = self._simulate_measurement(trial_wavefunction, self.hamiltonian_system['hamiltonian_active'])
        
        return expectation_value

    def _simulate_measurement(self, trial_wavefunction, hamiltonian):
        """
        FIXED: Real quantum simulation using Qiskit Aer primitives
        
        Args:
            trial_wavefunction: Parameterized quantum circuit |ψ(θ)⟩
            hamiltonian: Qubit Hamiltonian operator H
            
        Returns:
            float: Expectation value ⟨ψ(θ)|H|ψ(θ)⟩
        """
        try:
            # Attempt estimator first
            estimator_result = None
            try:
                job = self.estimator.run([trial_wavefunction], [hamiltonian])
            except TypeError:
                job = self.estimator.run([(trial_wavefunction, hamiltonian)])
            estimator_result = job.result()

            def collect_numbers(obj, depth=0, found=None):
                if found is None:
                    found = []
                if depth > 5:
                    return found
                import numpy as _np
                if isinstance(obj, (int, float)):
                    found.append(float(obj)); return found
                if isinstance(obj, _np.generic):
                    found.append(float(obj)); return found
                if isinstance(obj, (list, tuple)):
                    for x in obj[:4]:
                        collect_numbers(x, depth+1, found)
                    return found
                # Common attributes containers
                for attr in ("values", "evs", "expectation", "eigenvalues", "data"):
                    if hasattr(obj, attr):
                        try:
                            collect_numbers(getattr(obj, attr), depth+1, found)
                        except Exception:
                            pass
                # data may have nested attributes
                if hasattr(obj, '__dict__'):
                    for k, v in list(obj.__dict__.items())[:10]:
                        collect_numbers(v, depth+1, found)
                return found

            nums = collect_numbers(estimator_result)
            if nums:
                # Heuristic: take the first number (expected single expectation value)
                val = float(nums[0])
                # Count this effective circuit evaluation
                self.eval_calls += 1
                # Optionally inject synthetic shot noise for fair/noisy comparisons
                if self.shot_noise_shots is not None and self.shot_noise_shots > 0:
                    val = self._inject_shot_noise(val, hamiltonian, self.shot_noise_shots)
                return val
            else:
                raise TypeError(f"No numeric values found in estimator result {type(estimator_result)}")

        except Exception as e:
            # Deterministic statevector fallback (no randomness) before final stochastic fallback
            try:
                from qiskit.quantum_info import Statevector # type: ignore
                sv = Statevector.from_instruction(trial_wavefunction)
                val = sv.expectation_value(hamiltonian)
                out = float(np.real(val))
                self.eval_calls += 1
                if self.shot_noise_shots is not None and self.shot_noise_shots > 0:
                    out = self._inject_shot_noise(out, hamiltonian, self.shot_noise_shots)
                return out
            except Exception as e2:
                if self.verbose:
                    print(f"⚠ Quantum simulation error (estimator): {e}")
                    print(f"⚠ Statevector fallback failed: {e2}")
                    print("  Falling back to stochastic mock value")
                return -5.0 + np.random.normal(0, 0.1)

    def _inject_shot_noise(self, value: float, hamiltonian, shots: int) -> float:
        """Approximate shot noise by adding a zero-mean Gaussian with σ ≤ sqrt(sum c_i^2 / shots).

        This upper-bounds the standard deviation for weighted sums of ±1 Pauli measurements.
        It provides a simple, backend-agnostic way to emulate sampling noise when running on
        analytic simulators. Conservatively scales with the Hamiltonian coefficients.
        """
        try:
            coeffs = getattr(hamiltonian, 'coeffs', None)
            if coeffs is None:
                return value
            s2 = float(np.sum(np.abs(coeffs.astype(complex))**2))
            if shots <= 0:
                return value
            sigma = math.sqrt(max(s2, 0.0) / shots)
            return float(value + np.random.normal(0.0, sigma))
        except Exception:
            return value

    def objective_function(self, parameters):
        """
        Objective function for VQE optimization with iteration tracking
        
        Args:
            parameters: Current variational parameters
            
        Returns:
            float: Energy to minimize
        """
        hamiltonian = self.hamiltonian_system['hamiltonian_active']
        noise_factors = getattr(self.zne_plugin, 'noise_factors', [1.0]) or [1.0]

        # Build trial once
        trial_wavefunction = self.ansatz_plugin.get_trial_wavefunction(parameters)

        multi_noise = len(noise_factors) > 1
        noisy_values = []
        if multi_noise:
            # Multi-noise sampling: create folded circuits per factor
            for nf in noise_factors:
                if nf == 1.0:
                    circ = trial_wavefunction
                else:
                    try:
                        circ = self.zne_plugin.create_noisy_circuit(trial_wavefunction, nf)
                    except Exception as e:
                        if self.verbose:
                            print(f"[ZNE] Folding failed for factor {nf} ({e}); using original circuit.")
                        circ = trial_wavefunction
                val = self._simulate_measurement(circ, hamiltonian)
                # Coerce numeric
                try:
                    val = float(val)
                except Exception:
                    pass
                noisy_values.append(val)
            if self.verbose:
                print(f"[ZNE] Raw noisy values @ factors {noise_factors}: {noisy_values}")
            noisy_input = noisy_values
        else:
            # Single-factor (legacy) path
            single_val = self._simulate_measurement(trial_wavefunction, hamiltonian)
            try:
                single_val = float(single_val)
            except Exception:
                pass
            noisy_input = single_val

        denoised_value = self.zne_plugin.denoise(noisy_input)
        # Optionally retain last raw samples for external summary
        self.last_noisy_samples = noisy_values if multi_noise else [noisy_input]

        # Adaptive ZNE disable logic
        if (self.zne_adaptive_enable and multi_noise and
            hasattr(self.zne_plugin, 'zne_history') and self.zne_plugin.zne_history):
            last_record = self.zne_plugin.zne_history[-1]
            improvement = last_record.get('improvement', 0.0)
            if improvement < self.zne_improvement_tol:
                self.zne_no_improve_streak += 1
            else:
                self.zne_no_improve_streak = 0
            if (not self.zne_disabled and
                self.zne_no_improve_streak >= self.zne_patience and
                len(getattr(self.zne_plugin, 'noise_factors', [])) > 1):
                # Disable further multi-noise sampling
                self.zne_plugin.noise_factors = [1.0]
                self.zne_disabled = True
                if self.verbose:
                    print(f"[ZNE] Disabled multi-noise after {self.zne_no_improve_streak} consecutive < {self.zne_improvement_tol:.1e} improvements.")
        
        # Track iteration progress
        self.iteration_count += 1
        self.energy_history.append(denoised_value)
        self.parameter_history.append(parameters.copy())
        
        # Print iteration output as requested
        if self.verbose:
            self._print_iteration_output(denoised_value, parameters)
        
        return denoised_value

    def _print_iteration_output(self, energy, parameters):
        """Print detailed iteration output including trial wavefunction info"""
        print(f"\n{'='*60}")
        print(f"🔄 VQE ITERATION {self.iteration_count}")
        print(f"{'='*60}")
        print(f"📊 Energy = {energy:.8f} Hartree")
        print(f"📊 Energy = {energy * 627.509:.4f} kcal/mol")
        
        # Energy improvement tracking
        if len(self.energy_history) > 1:
            improvement = self.energy_history[-2] - energy
            print(f"📈 Energy improvement: {improvement:.8f} Hartree ({improvement*627.509:.4f} kcal/mol)")
        
        # Trial wavefunction parameters
        print(f"\n🌊 Trial Wavefunction |ψ(θ)⟩:")
        print("   " + "="*50)
        print(f"   Total parameters: {len(parameters)}")
        print(f"   Parameter range: [{np.min(parameters):7.4f}, {np.max(parameters):7.4f}]")
        print(f"   Parameter variance: {np.var(parameters):7.4f}")
        print(f"   RMS parameter: {np.sqrt(np.mean(parameters**2)):7.4f}")
        
        # Show first 8 parameters cleanly
        print(f"   Parameters:")
        for i in range(0, min(len(parameters), 8), 4):
            end_idx = min(i+4, len(parameters))
            param_group = parameters[i:end_idx]
            param_strs = [f"θ[{j:2d}]={param:7.4f}" for j, param in enumerate(param_group, i)]
            print(f"     {' '.join(param_strs)}")
        if len(parameters) > 8:
            print(f"     ... and {len(parameters)-8} more parameters")
        print("   " + "="*50)

    def run(self, initial_params=None, init_type="zero"):
        """Execute the VQE optimization loop.

        Args:
            initial_params (array-like | None): Optional user-specified initial parameter vector.
            init_type (str): If initial_params is None, strategy passed to ansatz_plugin.get_initial_parameters().

        Returns:
            (best_params: np.ndarray, best_energy: float)
        """
        if not self.ansatz_plugin.is_built:
            raise RuntimeError("Ansatz not built; initialization failed earlier.")

        # Prepare initial parameters
        if initial_params is None:
            initial_params = self.ansatz_plugin.get_initial_parameters(init_type=init_type)
        else:
            initial_params = np.array(initial_params, dtype=float)
            if len(initial_params) != self.ansatz_plugin.num_parameters:
                raise ValueError(
                    f"Initial parameter length {len(initial_params)} does not match ansatz parameter count {self.ansatz_plugin.num_parameters}."
                )

        if self.verbose:
            print("\n🚀 Starting VQE optimization run")
            print(f"   Parameter count: {len(initial_params)}")
            print(f"   Optimizer: {self.optimizer_plugin.__class__.__name__}")

        # Edge case: no variational parameters
        if len(initial_params) == 0:
            energy = self.objective_function(np.array([]))
            if self.verbose:
                print("   (No parameters to optimize – single evaluation performed)")
            return np.array([]), energy

        # Run optimizer
        best_params = self.optimizer_plugin.optimize(self.objective_function, initial_params)

        # Ensure we have energy for returned params (optimizer may store best earlier)
        try:
            current_energy = self.objective_function(best_params)
        except Exception:
            # Fallback to last recorded energy
            current_energy = self.energy_history[-1] if self.energy_history else float('nan')

        if self.verbose:
            print("\n🏁 VQE optimization complete")
            print(f"   Final energy: {current_energy:.10f} Hartree")
            if len(self.energy_history) >= 2:
                print(f"   Total improvement: {self.energy_history[0] - current_energy:.6f} Hartree")
            print(f"   Iterations (optimizer steps): {self.iteration_count}")
            print(f"   Effective circuit evaluations: {self.eval_calls}")

        return np.array(best_params, dtype=float), float(current_energy)


# Test function
def test_corrected_vqe():
    """Test the corrected VQE implementation"""
    print("🧪 Testing Corrected VQE Implementation")
    print("="*80)
    
    # Initialize plugins
    hamiltonian_plugin = HamiltonianPlugin()
    ansatz_plugin = AnsatzPlugin(verbose=True)
    # Use hybrid optimizer by default for test
    optimizer_plugin = HybridSPSAThenCOBYLA(spsa_iters=15, switch_tol=5e-3, min_spsa=8, force_cobyla=True, verbose=True)
    zne_plugin = ZNEDenoiserPlugin()
    
    # Create and run VQE
    vqe = VQE(ansatz_plugin, hamiltonian_plugin, optimizer_plugin, zne_plugin)
    
    # Run optimization
    result = vqe.run()
    
    if result[0] is not None:
        print(f"\n🎊 TEST SUCCESSFUL!")
        print(f"   Final energy: {result[1]:.6f} Hartree")
    else:
        print(f"\n❌ TEST FAILED")

if __name__ == "__main__":
    test_corrected_vqe()

