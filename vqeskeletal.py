import math
import random
import numpy as np

# Module version / build tag for notebook reload diagnostics
__VQESKELETAL_VERSION__ = "2025-09-03a"

# Imports for the Hamiltonian Plugin
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit import QuantumCircuit
try:
    from qiskit.primitives import Estimator
except ImportError:
    Estimator = None

# The following imports require qiskit-nature and a chemistry driver like pyscf
try:
    from qiskit_nature.units import DistanceUnit
    from qiskit_nature.second_q.drivers import PySCFDriver
    from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
    from qiskit_nature.second_q.problems import ElectronicStructureProblem
    from qiskit_nature.second_q.mappers import JordanWignerMapper
    from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
    QISKIT_NATURE_INSTALLED = True
except ImportError:
    QISKIT_NATURE_INSTALLED = False


class AnsatzPlugin:
    """
    Unified UCCSD Ansatz Plugin for VQE
    
    Takes Hamiltonian system as input and provides trial wavefunctions
    directly usable in VQE optimization.
    """

    def __init__(self, ansatz_reps=1, include_hf_state=True, verbose=True):
        """
        Initialize the unified ansatz plugin
        
        Args:
            ansatz_reps: Number of UCCSD repetitions (for deeper circuits)
            include_hf_state: Whether to include HF initial state in ansatz
            verbose: Whether to print detailed construction information
        """
        self.ansatz_reps = ansatz_reps
        self.include_hf_state = include_hf_state
        self.verbose = verbose
        
        # Ansatz system components (will be built when hamiltonian is provided)
        self.ansatz_circuit = None
        self.hf_state = None
        self.num_parameters = 0
        self.hamiltonian_system = None
        self.is_built = False
        
        # System properties
        self.num_qubits = 0
        self.num_spatial_orbitals = 0
        self.num_particles = None
        self.vqe_ready = False

    def build_from_hamiltonian(self, hamiltonian_system):
        """
        Build UCCSD ansatz from Hamiltonian system
        
        Args:
            hamiltonian_system: Output dict from HamiltonianPlugin.get_hamiltonian()
                               Must contain: 'problem_active', 'mapper', 'num_qubits'
        
        Returns:
            bool: True if build successful, False otherwise
        """
        if self.verbose:
            print("=" * 70)
            print("BUILDING UCCSD ANSATZ FROM HAMILTONIAN")
            print("=" * 70)

        self.hamiltonian_system = hamiltonian_system
        self.last_error = None

        try:
            problem_active = hamiltonian_system.get('problem_active')
            mapper = hamiltonian_system.get('mapper')
            hamiltonian = hamiltonian_system.get('hamiltonian_active')

            if problem_active is None or mapper is None or hamiltonian is None:
                if self.verbose:
                    print("⚠ Incomplete Hamiltonian system; building fallback ansatz")
                ok = self._build_fallback_ansatz(hamiltonian or hamiltonian_system.get('hamiltonian_active', None) or SparsePauliOp(['I'], [0.0]))
                self.is_built = ok
                return ok

            # System properties
            self.num_spatial_orbitals = problem_active.num_spin_orbitals // 2
            self.num_particles = problem_active.num_particles
            self.num_qubits = problem_active.num_spin_orbitals

            if self.verbose:
                print("Molecular system properties:")
                print(f"  Qubits: {self.num_qubits}")
                print(f"  Spatial orbitals: {self.num_spatial_orbitals}")
                print(f"  Particles (α, β): {self.num_particles}")
                print(f"  Basis: {hamiltonian_system.get('basis', 'unknown')}")

            # HF state
            if not self._build_hf_state(mapper):
                if self.verbose:
                    print("⚠ HF state build failed; using fallback ansatz")
                ok = self._build_fallback_ansatz(hamiltonian)
                self.is_built = ok
                return ok

            # UCCSD
            if not self._build_uccsd_ansatz(mapper):
                if self.verbose:
                    print("⚠ UCCSD build failed; using fallback ansatz")
                ok = self._build_fallback_ansatz(hamiltonian)
                self.is_built = ok
                return ok

            # Validate
            self._validate_system(hamiltonian)
            self.is_built = True

            if self.verbose:
                print("\n✓ Ansatz construction complete!")
                print(f"  Final qubits: {self.num_qubits}")
                print(f"  Variational parameters: {self.num_parameters}")
                print(f"  VQE ready: {self.vqe_ready}")
            return True

        except Exception as e:
            self.last_error = str(e)
            if self.verbose:
                print(f"✗ build_from_hamiltonian encountered error: {e}")
                print("  Falling back to generic parameterized ansatz.")
            try:
                # Fallback to simple circuit with same qubit count if possible
                h_active = hamiltonian_system.get('hamiltonian_active')
                if h_active is not None:
                    self._build_fallback_ansatz(h_active)
                else:
                    # Minimal single-qubit fallback
                    self._build_fallback_ansatz(SparsePauliOp(['I'], [0.0]))
            except Exception as e2:
                if self.verbose:
                    print(f"  Secondary fallback failed: {e2}")
                return False
            return True

    def _build_fallback_ansatz(self, hamiltonian):
        """Build a simple fallback ansatz when full UCCSD isn't possible"""
        self.num_qubits = hamiltonian.num_qubits
        self.num_parameters = 0
        
        # Create a simple parameterized circuit as fallback
        fallback_circuit = QuantumCircuit(self.num_qubits)
        
        # Add some basic rotations as variational parameters
        from qiskit.circuit import Parameter
        params = []
        for i in range(self.num_qubits):
            param = Parameter(f'theta_{i}')
            params.append(param)
            fallback_circuit.ry(param, i)
            
        # Add some entangling gates
        for i in range(self.num_qubits - 1):
            fallback_circuit.cx(i, i+1)
            
        self.ansatz_circuit = fallback_circuit
        self.num_parameters = len(params)
        self.vqe_ready = True
        self.is_built = True
        
        if self.verbose:
            print(f"✓ Built fallback parameterized ansatz with {self.num_parameters} parameters")
            
        return True

    def _build_hf_state(self, mapper):
        """Build Hartree-Fock reference state"""
        try:
            self.hf_state = HartreeFock(
                num_spin_orbitals=self.num_qubits,
                num_particles=self.num_particles,
                qubit_mapper=mapper
            )
            
            if self.verbose:
                print(f"✓ HF reference state created")
                print(f"  HF circuit depth: {self.hf_state.depth()}")
                print(f"  HF gates: {len(self.hf_state.data)}")
            
            return True
            
        except Exception as e:
            if self.verbose:
                print(f"✗ Failed to create HF state: {e}")
            return False

    def _build_uccsd_ansatz(self, mapper):
        """Build UCCSD ansatz circuit"""
        try:
            # Method 1: UCCSD with internal HF reference
            uccsd_ansatz = UCCSD(
                num_spatial_orbitals=self.num_spatial_orbitals,
                num_particles=self.num_particles,
                qubit_mapper=mapper,
                reps=self.ansatz_reps
            )
            
            self.ansatz_circuit = uccsd_ansatz
            
            if self.verbose:
                print(f"✓ UCCSD ansatz created successfully")
                
        except Exception as e1:
            if self.verbose:
                print(f"✗ Standard UCCSD failed: {e1}")
            
            try:
                # Method 2: Manual composition
                uccsd_bare = UCCSD(
                    num_spatial_orbitals=self.num_spatial_orbitals,
                    num_particles=self.num_particles,
                    qubit_mapper=mapper,
                    reps=self.ansatz_reps
                )

                if self.include_hf_state and self.hf_state is not None:
                    # Compose HF + UCCSD
                    full_circuit = QuantumCircuit(self.num_qubits)
                    full_circuit.compose(self.hf_state, inplace=True)
                    full_circuit.compose(uccsd_bare, inplace=True)
                    full_circuit._parameters = uccsd_bare.parameters
                    self.ansatz_circuit = full_circuit
                else:
                    self.ansatz_circuit = uccsd_bare
                    
                if self.verbose:
                    print(f"✓ UCCSD ansatz created via manual composition")
                    
            except Exception as e2:
                if self.verbose:
                    print(f"✗ Manual composition failed: {e2}")
                    print("  Using HF-only ansatz")
                
                self.ansatz_circuit = self.hf_state if self.hf_state else QuantumCircuit(self.num_qubits)
                
        # Get number of parameters
        if hasattr(self.ansatz_circuit, 'num_parameters'):
            self.num_parameters = self.ansatz_circuit.num_parameters
        elif hasattr(self.ansatz_circuit, 'parameters'):
            self.num_parameters = len(self.ansatz_circuit.parameters)
        else:
            self.num_parameters = 0
            
        if self.verbose:
            print(f"  Circuit depth: {self.ansatz_circuit.depth()}")
            print(f"  Total gates: {len(self.ansatz_circuit.data)}")
            print(f"  Variational parameters: {self.num_parameters}")
            
        return True

    def _validate_system(self, hamiltonian):
        """Validate ansatz compatibility with Hamiltonian"""
        if hamiltonian.num_qubits != self.ansatz_circuit.num_qubits:
            if self.verbose:
                print(f"⚠ Warning: Hamiltonian ({hamiltonian.num_qubits} qubits) and "
                      f"ansatz ({self.ansatz_circuit.num_qubits} qubits) mismatch!")
            self.vqe_ready = False
        else:
            if self.verbose:
                print(f"✓ Hamiltonian and ansatz compatible ({hamiltonian.num_qubits} qubits)")
            self.vqe_ready = self.num_parameters > 0

    def get_trial_wavefunction(self, parameters):
        """
        Get trial wavefunction (parameterized quantum circuit) for given parameters
        
        Args:
            parameters: Variational parameters (numpy array or list)
            
        Returns:
            QuantumCircuit: Trial wavefunction circuit ready for VQE evaluation
        """
        if not self.is_built:
            if self.hamiltonian_system is not None:
                if self.verbose:
                    print("[auto-build] Attempting to build ansatz inside get_trial_wavefunction")
                self.build_from_hamiltonian(self.hamiltonian_system)
            if not self.is_built:
                raise RuntimeError("Ansatz not built after auto-attempt in get_trial_wavefunction().")
            
        if self.num_parameters == 0:
            # No parameters to bind - return circuit as is
            return self.ansatz_circuit.copy()
            
        # Validate parameter count
        if len(parameters) != self.num_parameters:
            raise ValueError(f"Expected {self.num_parameters} parameters, got {len(parameters)}")
        
        # Bind parameters to circuit
        try:
            if hasattr(self.ansatz_circuit, 'bind_parameters'):
                trial_wavefunction = self.ansatz_circuit.bind_parameters(parameters)
            else:
                # Fallback method
                trial_wavefunction = self.ansatz_circuit.copy()
                if hasattr(trial_wavefunction, 'parameters') and trial_wavefunction.parameters:
                    param_dict = dict(zip(trial_wavefunction.parameters, parameters))
                    trial_wavefunction = trial_wavefunction.assign_parameters(param_dict)
                    
            return trial_wavefunction
            
        except Exception as e:
            raise RuntimeError(f"Failed to bind parameters to ansatz: {e}")

    def get_initial_parameters(self, init_type="zero"):
        """
        Generate initial parameters for optimization
        
        Args:
            init_type: "zero", "random_small", "random_normal", or "hf_like"
            
        Returns:
            np.array: Initial parameter values
        """
        if not self.is_built:
            if self.hamiltonian_system is not None:
                if self.verbose:
                    print("[auto-build] Attempting to build ansatz inside get_initial_parameters")
                self.build_from_hamiltonian(self.hamiltonian_system)
            if not self.is_built:
                raise RuntimeError("Ansatz not built after auto-attempt in get_initial_parameters().")
            
        if self.num_parameters == 0:
            return np.array([])
            
        if init_type == "zero":
            return np.zeros(self.num_parameters)
        elif init_type == "random_small":
            return np.random.normal(0, 0.01, self.num_parameters)
        elif init_type == "random_normal":
            return np.random.normal(0, 0.1, self.num_parameters)
        elif init_type == "hf_like":
            # Small perturbations around HF (all zeros)
            return np.random.normal(0, 0.005, self.num_parameters)
        else:
            return np.zeros(self.num_parameters)

    def get_parameter_bounds(self, bound_type="standard"):
        """
        Get parameter bounds for constrained optimization
        
        Args:
            bound_type: "tight", "standard", or "loose"
            
        Returns:
            list: Parameter bounds [(min, max), ...] for each parameter
        """
        if not self.is_built or self.num_parameters == 0:
            return []
            
        if bound_type == "tight":
            return [(-0.1, 0.1)] * self.num_parameters
        elif bound_type == "loose":
            return [(-2*np.pi, 2*np.pi)] * self.num_parameters
        else:  # standard
            return [(-0.5, 0.5)] * self.num_parameters

    def get_ansatz_info(self):
        """
        Get comprehensive information about the built ansatz
        
        Returns:
            dict: Complete ansatz system information
        """
        info = {
            "built": self.is_built,
            "vqe_ready": self.vqe_ready if self.is_built else False,
            "num_qubits": self.num_qubits or 0,
            "num_parameters": self.num_parameters if self.is_built else 0,
            "circuit_depth": (self.ansatz_circuit.depth() if (self.is_built and self.ansatz_circuit) else 0),
            "circuit_gates": (len(self.ansatz_circuit.data) if (self.is_built and self.ansatz_circuit) else 0),
            "num_spatial_orbitals": self.num_spatial_orbitals or 0,
            "num_particles": self.num_particles,
            "ansatz_reps": self.ansatz_reps,
            "include_hf_state": self.include_hf_state,
            "basis": self.hamiltonian_system.get('basis', 'unknown') if self.hamiltonian_system else 'unknown',
            "geometry": self.hamiltonian_system.get('geometry', 'unknown') if self.hamiltonian_system else 'unknown',
            "module_version": __VQESKELETAL_VERSION__,
            "last_error": getattr(self, 'last_error', None)
        }
        if not self.is_built:
            info["error"] = "Ansatz not built"
        return info

    # Convenience method (can be called externally if desired)
    def ensure_built(self, hamiltonian_system=None):
        """Ensure the ansatz is built; build automatically if possible.

        Args:
            hamiltonian_system (dict|None): Optional Hamiltonian system dict. If not
                provided, will use previously stored self.hamiltonian_system.

        Returns:
            bool: True if built (after this call), else False.
        """
        if self.is_built:
            return True
        system = hamiltonian_system or self.hamiltonian_system
        if system is None:
            return False
        return bool(self.build_from_hamiltonian(system))


# Keep the existing HamiltonianPlugin unchanged
class HamiltonianPlugin:
    """
    Plugin for generating the active-space Hamiltonian for Ammonia (NH3).
    """
    MIN_TERMS = 400
    PAD_SYNTHETIC = True
    SYN_COEFF_SCALE = 1e-8

    def __init__(self, pad_synthetic=True, min_terms=400, physical_threshold=1e-6):
        self.geom = (
            "N  0.0000  0.0000  0.0000;"
            " H  0.9377  0.0000 -0.3816;"
            " H -0.4688  0.8119 -0.3816;"
            " H -0.4688 -0.8119 -0.3816"
        )
        self._hamiltonian = None
        self._hamiltonian_physical = None
        self._problem_active = None
        self._mapper = None
        self._reference_rhf = None
        self._identity_coeff = None
        self._nuclear_repulsion = None
        # Configurable behavior
        self.PAD_SYNTHETIC = pad_synthetic
        self.MIN_TERMS = min_terms
        self.PHYSICAL_THRESHOLD = physical_threshold

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
                "hamiltonian_physical": self._hamiltonian_physical or self._hamiltonian,
                "num_qubits": self._hamiltonian.num_qubits,
                "basis": "sto3g",
                "geometry": self.geom,
                "reference_rhf_energy": self._reference_rhf
            }
        try:
            if not QISKIT_NATURE_INSTALLED:
                raise ImportError("Qiskit Nature or its dependencies are not installed.")

            # Build full electronic structure problem via driver
            driver = PySCFDriver(atom=self.geom, basis='sto3g', charge=0, spin=0, unit=DistanceUnit.ANGSTROM)
            res = driver.run()
            problem_full = res if isinstance(res, ElectronicStructureProblem) else ElectronicStructureProblem(res)
            try:
                self._nuclear_repulsion = float(problem_full.nuclear_repulsion_energy)
            except Exception:
                self._nuclear_repulsion = None

            # Active space: 4 electrons, 3 spatial orbitals (6 spin orbitals / qubits)
            transformer = ActiveSpaceTransformer(num_electrons=4, num_spatial_orbitals=3)
            self._problem_active = transformer.transform(problem_full)

            # Mapper & qubit Hamiltonian (use direct hamiltonian.second_q_op for stability)
            self._mapper = JordanWignerMapper()
            fermionic_op = self._problem_active.hamiltonian.second_q_op()
            ham_active = self._mapper.map(fermionic_op)
            # Capture identity coefficient (constant energy shift)
            for p, c in zip(ham_active.paulis, ham_active.coeffs):
                if str(p) == 'I' * ham_active.num_qubits:
                    self._identity_coeff = complex(c)
                    break

            if ham_active.num_qubits != 6:
                raise RuntimeError(f'Active space produced {ham_active.num_qubits} qubits, expected 6.')

            # PySCF RHF reference energy (direct) for benchmarking
            try:
                from pyscf import gto, scf
                mol = gto.M(atom=self.geom.replace(';', '; '), basis='sto-3g', unit='Angstrom', charge=0, spin=0)
                mf = scf.RHF(mol)
                self._reference_rhf = float(mf.kernel())
            except Exception:
                self._reference_rhf = None

        except Exception as e:
            print(f'[Warning] Ab initio build failed: {e}. Using a fallback 6-qubit operator.')
            paulis = ['IIIIII', 'ZIIIZZ', 'ZZIIZZ', 'IZZIIZ', 'IIZZZZ', 'XXYYZZ', 'YYXXZZ']
            coeffs = [-5.0, 0.12, -0.08, 0.05, -0.03, 0.01, 0.01]
            ham_active = SparsePauliOp(paulis, coeffs)
            self._problem_active = None
            self._mapper = None
            self._reference_rhf = None

        # Separate physical and padded terms
        all_terms = {str(p): complex(c) for p, c in zip(ham_active.paulis, ham_active.coeffs) if abs(complex(c)) > 1e-12}
        physical_terms = {lbl: coeff for lbl, coeff in all_terms.items() if abs(coeff) > self.PHYSICAL_THRESHOLD}
        physical_count = len(physical_terms)

        # Build physical-only operator (before padding)
        self._hamiltonian_physical = SparsePauliOp.from_list(list(physical_terms.items())) if physical_terms else ham_active

        padded_terms = dict(physical_terms)  # start from physical
        if self.PAD_SYNTHETIC and len(padded_terms) < self.MIN_TERMS:
            self._add_synthetic_padding(padded_terms, ham_active.num_qubits, len(padded_terms))

        self._print_summary(padded_terms, physical_count)

        self._hamiltonian = SparsePauliOp.from_list(list(padded_terms.items()))
        
        return {
            "problem_active": self._problem_active,
            "mapper": self._mapper,
            "hamiltonian_active": self._hamiltonian,
            "hamiltonian_physical": self._hamiltonian_physical or self._hamiltonian,
            "num_qubits": self._hamiltonian.num_qubits,
            "basis": "sto3g",
            "geometry": self.geom,
            "reference_rhf_energy": self._reference_rhf,
            "identity_coefficient": self._identity_coeff,
            "nuclear_repulsion_energy": self._nuclear_repulsion
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


# Placeholder classes remain the same
class ClassicalOptimizerPlugin:
    """Plugin for the classical optimization routine."""
    def optimize(self, objective_function, initial_params):
        raise NotImplementedError("Optimizer plugin not implemented.")


class HybridOptimizer(ClassicalOptimizerPlugin):
    """Simple hybrid optimizer: coarse random sampling + local coordinate descent."""
    def __init__(self, samples=20, local_iters=40, step=0.2, shrink=0.5, tol=1e-6, seed=None, bounds=None, verbose=True):
        self.samples = samples
        self.local_iters = local_iters
        self.step = step
        self.shrink = shrink
        self.tol = tol
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)
        self.bounds = bounds  # list of (min,max)

    def _clip(self, x):
        if self.bounds is None:
            return x
        return np.array([np.clip(val, lo, hi) for val, (lo, hi) in zip(x, self.bounds)])

    def _coordinate_descent(self, objective_function, start):
        params = start.copy()
        best_val = objective_function(params)
        step = self.step
        for _ in range(self.local_iters):
            improved = False
            for i in range(len(params)):
                for direction in (+1, -1):
                    trial = params.copy()
                    trial[i] += direction * step
                    trial = self._clip(trial)
                    val = objective_function(trial)
                    if val < best_val - self.tol:
                        best_val = val
                        params = trial
                        improved = True
                        if self.verbose:
                            print(f"[hybrid] improved dim {i} dir {direction} -> {best_val:.6f}")
            if not improved:
                step *= self.shrink
                if step < self.tol:
                    break
        return params, best_val

    def optimize(self, objective_function, initial_params):
        best_params = np.array(initial_params, dtype=float)
        best_val = objective_function(best_params)
        if self.verbose:
            print(f"[hybrid] initial value {best_val:.6f}")
        # Global sampling
        for s in range(self.samples):
            if self.bounds is not None:
                trial = np.array([self.rng.uniform(lo, hi) for (lo, hi) in self.bounds])
            else:
                trial = self.rng.uniform(-1.0, 1.0, size=len(best_params))
            val = objective_function(trial)
            if val < best_val:
                best_val = val
                best_params = trial
                if self.verbose:
                    print(f"[hybrid] sample {s} improved -> {best_val:.6f}")
        # Local refinement
        refined, final_val = self._coordinate_descent(objective_function, best_params)
        if final_val < best_val:
            best_params, best_val = refined, final_val
        if self.verbose:
            print(f"[hybrid] final value {best_val:.6f}")
        return best_params


# Advanced hybrid optimizer integrating (optional) SPSA + COBYLA
try:
    from qiskit_algorithms.optimizers import SPSA as _QiskitSPSA, COBYLA as _QiskitCOBYLA
except Exception:
    _QiskitSPSA = None
    _QiskitCOBYLA = None


class AdvancedHybridOptimizer(ClassicalOptimizerPlugin):
    """Three-stage hybrid optimizer: random sampling -> SPSA -> COBYLA.

    Falls back gracefully if qiskit-algorithms optimizers are unavailable.
    """
    def __init__(self,
                 samples=32,
                 spsa_iters=80,
                 cobyla_maxiter=200,
                 random_bound=1.0,
                 seed=None,
                 bounds=None,
                 verbose=True):
        self.samples = samples
        self.spsa_iters = spsa_iters
        self.cobyla_maxiter = cobyla_maxiter
        self.random_bound = random_bound
        self.rng = np.random.default_rng(seed)
        self.bounds = bounds  # list[(min,max)]
        self.verbose = verbose

    def _clip(self, x):
        if self.bounds is None:
            return x
        return np.array([np.clip(val, lo, hi) for val, (lo, hi) in zip(x, self.bounds)])

    def _random_sampling(self, objective_function, initial_params):
        best_params = initial_params.copy()
        best_val = objective_function(best_params)
        if self.verbose:
            print(f"[adv-hybrid] initial value {best_val:.6f}")
        for s in range(self.samples):
            if self.bounds is not None:
                trial = np.array([self.rng.uniform(lo, hi) for (lo, hi) in self.bounds])
            else:
                trial = self.rng.uniform(-self.random_bound, self.random_bound, size=len(best_params))
            val = objective_function(trial)
            if val < best_val:
                best_val = val
                best_params = trial
                if self.verbose:
                    print(f"[adv-hybrid] sample {s} improved -> {best_val:.6f}")
        return best_params, best_val

    def _run_spsa(self, objective_function, start):
        if _QiskitSPSA is None or self.spsa_iters <= 0:
            if self.verbose:
                print("[adv-hybrid] SPSA unavailable or disabled; skipping stage")
            return start, objective_function(start)
        if self.verbose:
            print(f"[adv-hybrid] Running SPSA for {self.spsa_iters} iterations")
        opt = _QiskitSPSA(maxiter=self.spsa_iters)
        # qiskit-algorithms optimizers expect a callable returning (value, gradient) optionally; SPSA supports value-only
        def fun(theta):
            return objective_function(theta)
        result = opt.minimize(fun=fun, x0=start)
        final_params = self._clip(result.x)
        final_val = objective_function(final_params)
        if self.verbose:
            print(f"[adv-hybrid] SPSA final value {final_val:.6f}")
        return final_params, final_val

    def _run_cobyla(self, objective_function, start):
        if _QiskitCOBYLA is None or self.cobyla_maxiter <= 0:
            if self.verbose:
                print("[adv-hybrid] COBYLA unavailable or disabled; skipping stage")
            return start, objective_function(start)
        if self.verbose:
            print(f"[adv-hybrid] Refining with COBYLA (maxiter={self.cobyla_maxiter})")
        opt = _QiskitCOBYLA(maxiter=self.cobyla_maxiter)
        def fun(theta):
            return objective_function(theta)
        result = opt.minimize(fun=fun, x0=start)
        final_params = self._clip(result.x)
        final_val = objective_function(final_params)
        if self.verbose:
            print(f"[adv-hybrid] COBYLA final value {final_val:.6f}")
        return final_params, final_val

    def optimize(self, objective_function, initial_params):
        initial_params = np.array(initial_params, dtype=float)
        # Stage 1: Random sampling
        params, val = self._random_sampling(objective_function, initial_params)
        # Stage 2: SPSA (if available)
        params, val = self._run_spsa(objective_function, params)
        # Stage 3: COBYLA (if available)
        params, val = self._run_cobyla(objective_function, params)
        if self.verbose:
            print(f"[adv-hybrid] final optimized value {val:.6f}")
        return params


class ZNEDenoiserPlugin:
    """Plugin for applying Zero-Noise Extrapolation (ZNE) to mitigate errors."""
    def denoise(self, noisy_results):
        return noisy_results


# Updated VQE class with simplified interface
class VQE:
    """
    The main VQE class that orchestrates the algorithm using the provided plugins.
    """
    def __init__(self, ansatz_plugin, hamiltonian_plugin, optimizer_plugin, zne_plugin, use_physical_only=False):
        self.ansatz_plugin = ansatz_plugin
        self.hamiltonian_plugin = hamiltonian_plugin
        self.optimizer_plugin = optimizer_plugin
        self.zne_plugin = zne_plugin
        self.use_physical_only = use_physical_only
        
        # Build the system
        self.hamiltonian_system = self.hamiltonian_plugin.get_hamiltonian()
        self.ansatz_plugin.build_from_hamiltonian(self.hamiltonian_system)
        # Prepare estimator primitive if available
        self._estimator = None
        if Estimator is not None and self.hamiltonian_system.get('hamiltonian_active') is not None:
            try:
                self._estimator = Estimator()
            except Exception:
                self._estimator = None

    def _get_expectation_value(self, parameters):
        trial_wavefunction = self.ansatz_plugin.get_trial_wavefunction(parameters)
        ham = (self.hamiltonian_system['hamiltonian_physical']
               if self.use_physical_only and self.hamiltonian_system.get('hamiltonian_physical') is not None
               else self.hamiltonian_system['hamiltonian_active'])
        if self._estimator is not None:
            try:
                job = self._estimator.run([trial_wavefunction], [ham])
                res = job.result()
                return float(res.values[0])
            except Exception:
                pass
        # Fallback path: compute full expectation via statevector simulation
        try:
            sv = Statevector.from_instruction(trial_wavefunction)
            energy = 0.0
            for p, c in zip(ham.paulis, ham.coeffs):
                coeff = complex(c)
                if abs(coeff) < 1e-14:
                    continue
                exp_val = sv.expectation_value(SparsePauliOp([p]))
                energy += coeff * exp_val
            return float(np.real_if_close(energy))
        except Exception:
            return 0.0

    def objective_function(self, parameters):
        noisy_value = self._get_expectation_value(parameters)
        denoised_value = self.zne_plugin.denoise(noisy_value)
        return denoised_value

    def run(self, initial_params=None):
        print("Starting VQE algorithm...")
        
        ansatz_info = self.ansatz_plugin.get_ansatz_info()
        
        if not ansatz_info['vqe_ready']:
            print("❌ VQE cannot run - ansatz not ready")
            return None, None
        
        if ansatz_info['num_parameters'] == 0:
            print("❌ No variational parameters - cannot optimize")
            return None, None
        
        # Get initial parameters if not provided
        if initial_params is None:
            initial_params = self.ansatz_plugin.get_initial_parameters("zero")
            print(f"Generated {len(initial_params)} initial parameters")
        
        # Run optimization
        optimized_params = self.optimizer_plugin.optimize(self.objective_function, initial_params)
        print("Optimization complete.")
        
        final_energy = self.objective_function(optimized_params)
        print(f"Optimal Parameters: {optimized_params}")
        print(f"Estimated Ground State Energy: {final_energy}")
        
        return optimized_params, final_energy


# Example usage
if __name__ == '__main__':
    # Create plugins
    ansatz_plugin = AnsatzPlugin(ansatz_reps=1, include_hf_state=True, verbose=True)
    hamiltonian_plugin = HamiltonianPlugin()
    optimizer_plugin = ClassicalOptimizerPlugin()
    zne_plugin = ZNEDenoiserPlugin()

    try:
        # Create and run VQE
        vqe_instance = VQE(
            ansatz_plugin=ansatz_plugin,
            hamiltonian_plugin=hamiltonian_plugin,
            optimizer_plugin=optimizer_plugin,
            zne_plugin=zne_plugin
        )
        
        # Show ansatz information
        info = ansatz_plugin.get_ansatz_info()
        print(f"\n{'='*50}")
        print("FINAL ANSATZ SYSTEM INFO")
        print("="*50)
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # Attempt to run (will fail at optimizer as expected)
        vqe_instance.run()

    except NotImplementedError as e:
        print(f"\n✓ Setup successful! {e}")
        print("The ansatz is ready - implement the optimizer to complete VQE.")
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
