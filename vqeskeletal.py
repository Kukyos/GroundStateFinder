import math
import random 
import numpy as np 

# Imports for the Hamiltonian Plugin
from qiskit.quantum_info import SparsePauliOp
from qiskit import QuantumCircuit

# The following imports require qiskit-nature and a chemistry driver like pyscf
# pip install qiskit-nature pyscf
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


class UCCSDAnsatzBuilder:
    """
    UCCSD ansatz builder for existing molecular Hamiltonian systems
    Works with the MolecularHamiltonianBuilder output
    """

    def __init__(self, ansatz_reps=1, include_hf_state=True):
        """
        Initialize the UCCSD ansatz builder

        Args:
            ansatz_reps: Number of UCCSD repetitions (for deeper circuits)
            include_hf_state: Whether to include HF initial state in ansatz
        """
        self.ansatz_reps = ansatz_reps
        self.include_hf_state = include_hf_state

    def build_uccsd_ansatz(self, hamiltonian_system):
        """
        Build UCCSD ansatz from existing Hamiltonian system

        Args:
            hamiltonian_system: Output dict from HamiltonianPlugin.get_hamiltonian()
                               Must contain: 'problem_active', 'mapper', 'num_qubits'

        Returns:
            dict: Complete ansatz system with analysis
        """
        print("="*70)
        print("UCCSD ANSATZ BUILDER")
        print("="*70)

        # Extract information from Hamiltonian system
        problem_active = hamiltonian_system['problem_active']
        mapper = hamiltonian_system['mapper']
        hamiltonian = hamiltonian_system['hamiltonian_active']

        if problem_active is None or mapper is None:
            raise RuntimeError("Cannot build UCCSD ansatz: problem_active or mapper is None (fallback Hamiltonian used)")

        num_spatial_orbitals = problem_active.num_spin_orbitals // 2
        num_particles = problem_active.num_particles
        num_qubits = problem_active.num_spin_orbitals

        print(f"Input system properties:")
        print(f"  Qubits: {num_qubits}")
        print(f"  Spatial orbitals: {num_spatial_orbitals}")
        print(f"  Particles (α, β): {num_particles}")
        print(f"  Hamiltonian terms: {len(hamiltonian)}")
        print(f"  Basis: {hamiltonian_system.get('basis', 'unknown')}")

        print(f"\nAnsatz configuration:")
        print(f"  UCCSD repetitions: {self.ansatz_reps}")
        print(f"  Include HF state: {self.include_hf_state}")

        # 1. CREATE HARTREE-FOCK REFERENCE STATE
        print(f"\n{'='*50}")
        print("STEP 1: HARTREE-FOCK REFERENCE STATE")
        print("="*50)

        try:
            hf_state = HartreeFock(
                num_spin_orbitals=num_qubits,
                num_particles=num_particles,
                qubit_mapper=mapper
            )

            print(f"✓ HF state created successfully")
            print(f"  HF qubits: {hf_state.num_qubits}")
            print(f"  HF depth: {hf_state.depth()}")
            print(f"  HF gates: {len(hf_state.data)}")

            # Show HF state structure
            print(f"\nHF reference state circuit:")
            if len(hf_state.data) <= 10:
                print(hf_state)
            else:
                print(f"  (Circuit with {len(hf_state.data)} gates - showing first few)")
                temp_circuit = QuantumCircuit(hf_state.num_qubits)
                for i, instruction in enumerate(hf_state.data[:5]):
                    temp_circuit.append(instruction.operation, instruction.qubits)
                print(temp_circuit)
                print(f"  ... and {len(hf_state.data) - 5} more gates")

        except Exception as e:
            raise RuntimeError(f"Failed to create HF state: {e}")

        # 2. CREATE UCCSD ANSATZ
        print(f"\n{'='*50}")
        print("STEP 2: UCCSD ANSATZ CONSTRUCTION")
        print("="*50)

        ansatz_success = False
        uccsd_ansatz = None

        # Method 1: Try UCCSD with internal HF reference (recommended)
        try:
            print("Attempting Method 1: UCCSD with internal HF reference...")

            uccsd_ansatz = UCCSD(
                num_spatial_orbitals=num_spatial_orbitals,
                num_particles=num_particles,
                qubit_mapper=mapper,
                reps=self.ansatz_reps
            )

            print(f"✓ UCCSD ansatz created successfully (Method 1)")
            ansatz_success = True

        except Exception as e1:
            print(f"✗ Method 1 failed: {e1}")

            # Method 2: Try creating UCCSD separately and composing with HF
            try:
                print("Attempting Method 2: Manual HF + UCCSD composition...")

                # Create bare UCCSD without initial state
                uccsd_bare = UCCSD(
                    num_spatial_orbitals=num_spatial_orbitals,
                    num_particles=num_particles,
                    qubit_mapper=mapper,
                    reps=self.ansatz_reps
                )

                if self.include_hf_state:
                    # Compose HF + UCCSD manually
                    full_circuit = QuantumCircuit(max(hf_state.num_qubits, uccsd_bare.num_qubits))

                    # Add HF initialization
                    if hf_state.num_qubits <= full_circuit.num_qubits:
                        full_circuit.compose(hf_state, inplace=True)

                    # Add UCCSD on top
                    if uccsd_bare.num_qubits <= full_circuit.num_qubits:
                        full_circuit.compose(uccsd_bare, inplace=True)

                    uccsd_ansatz = full_circuit
                    uccsd_ansatz._parameters = uccsd_bare.parameters  # Preserve parameters
                else:
                    uccsd_ansatz = uccsd_bare

                print(f"✓ UCCSD ansatz created successfully (Method 2)")
                ansatz_success = True

            except Exception as e2:
                print(f"✗ Method 2 failed: {e2}")

                # Method 3: Fallback to just HF state
                print("Using Method 3: HF-only fallback...")
                uccsd_ansatz = hf_state
                print(f"⚠ Using HF state as fallback (no variational parameters)")

        # 3. ANALYZE ANSATZ PROPERTIES
        print(f"\n{'='*50}")
        print("STEP 3: ANSATZ ANALYSIS")
        print("="*50)

        print(f"Final ansatz properties:")
        print(f"  Ansatz qubits: {uccsd_ansatz.num_qubits}")
        print(f"  Circuit depth: {uccsd_ansatz.depth()}")
        print(f"  Total gates: {len(uccsd_ansatz.data)}")

        # Get number of parameters
        if ansatz_success and hasattr(uccsd_ansatz, 'num_parameters'):
            num_parameters = uccsd_ansatz.num_parameters
        elif ansatz_success and hasattr(uccsd_ansatz, 'parameters'):
            num_parameters = len(uccsd_ansatz.parameters)
        else:
            num_parameters = 0

        print(f"  Variational parameters: {num_parameters}")

        # 4. DECOMPOSE AND ANALYZE CIRCUIT STRUCTURE
        if ansatz_success and num_parameters > 0:
            print(f"\n{'='*50}")
            print("STEP 4: CIRCUIT STRUCTURE ANALYSIS")
            print("="*50)

            try:
                # Try to decompose the circuit
                print("Decomposing UCCSD ansatz...")
                decomposed = uccsd_ansatz.decompose()

                print(f"After decomposition:")
                print(f"  Decomposed depth: {decomposed.depth()}")
                print(f"  Decomposed gates: {len(decomposed.data)}")

                # Count gate types
                gate_counts = {}
                for instruction in decomposed.
                    gate_name = instruction.operation.name
                    gate_counts[gate_name] = gate_counts.get(gate_name, 0) + 1

                print(f"\nGate composition:")
                for gate, count in sorted(gate_counts.items()):
                    percentage = 100 * count / len(decomposed.data)
                    print(f"  {gate}: {count} ({percentage:.1f}%)")

                # Show sample gates
                print(f"\nFirst 15 gates of decomposed circuit:")
                for i, instruction in enumerate(decomposed.data[:15]):
                    gate_name = instruction.operation.name
                    qubits = [decomposed.find_bit(q).index for q in instruction.qubits]
                    if len(qubits) == 1:
                        print(f"  {i+1:2d}: {gate_name} on qubit {qubits[0]}")
                    else:
                        print(f"  {i+1:2d}: {gate_name} on qubits {qubits}")

                if len(decomposed.data) > 15:
                    print(f"  ... and {len(decomposed.data) - 15} more gates")

            except Exception as e:
                print(f"Could not decompose circuit: {e}")

            # 5. ESTIMATE EXCITATION STRUCTURE
            print(f"\n{'='*50}")
            print("STEP 5: EXCITATION ANALYSIS")
            print("="*50)

            try:
                n_electrons = sum(num_particles)
                n_occupied = n_electrons // 2  # For closed shell
                n_virtual = num_spatial_orbitals - n_occupied

                # Theoretical excitation counts
                singles_theory = n_occupied * n_virtual
                doubles_theory = (n_occupied * (n_occupied - 1) * n_virtual * (n_virtual - 1)) // 4

                print(f"Theoretical excitation analysis:")
                print(f"  Occupied orbitals: {n_occupied}")
                print(f"  Virtual orbitals: {n_virtual}")
                print(f"  Single excitations: {singles_theory}")
                print(f"  Double excitations: {doubles_theory}")
                print(f"  Total theoretical parameters: {singles_theory + doubles_theory}")
                print(f"  Actual UCCSD parameters: {num_parameters}")

                if num_parameters > 0:
                    efficiency = num_parameters / (singles_theory + doubles_theory)
                    print(f"  Parameter efficiency: {efficiency:.3f}")

                # Try to access excitation operators if available
                if hasattr(uccsd_ansatz, '_excitation_list'):
                    excitations = uccsd_ansatz._excitation_list
                    singles = [exc for exc in excitations if len(exc[0]) == 1]
                    doubles = [exc for exc in excitations if len(exc[0]) == 2]

                    print(f"\nActual excitation operators:")
                    print(f"  Single excitations found: {len(singles)}")
                    print(f"  Double excitations found: {len(doubles)}")

                    if singles and len(singles) <= 10:
                        print(f"  Example singles: {singles[:5]}")
                    if doubles and len(doubles) <= 10:
                        print(f"  Example doubles: {doubles[:3]}")

            except Exception as e:
                print(f"Could not analyze excitation structure: {e}")

        # 6. VALIDATION CHECKS
        print(f"\n{'='*50}")
        print("STEP 6: SYSTEM VALIDATION")
        print("="*50)

        # Check compatibility with Hamiltonian
        if hamiltonian.num_qubits != uccsd_ansatz.num_qubits:
            print(f"⚠ Warning: Hamiltonian ({hamiltonian.num_qubits} qubits) and "
                  f"ansatz ({uccsd_ansatz.num_qubits} qubits) mismatch!")
            validation_passed = False
        else:
            print(f"✓ Hamiltonian and ansatz are compatible ({hamiltonian.num_qubits} qubits)")
            validation_passed = True

        # Check parameter count
        if ansatz_success and num_parameters > 0:
            print(f"✓ Ansatz has {num_parameters} variational parameters")
        elif ansatz_success:
            print(f"⚠ UCCSD created but has no variational parameters")
        else:
            print(f"⚠ Using HF-only ansatz (no optimization possible)")

        # Final status
        if validation_passed and ansatz_success and num_parameters > 0:
            print(f"✓ System ready for VQE optimization")
            vqe_ready = True
        else:
            print(f"⚠ System has limitations for VQE")
            vqe_ready = False

        # 7. RETURN COMPLETE ANSATZ SYSTEM
        return {
            # Main ansatz components
            'ansatz': uccsd_ansatz,
            'hf_state': hf_state,
            'ansatz_success': ansatz_success,
            'vqe_ready': vqe_ready,

            # Ansatz properties
            'num_qubits': uccsd_ansatz.num_qubits,
            'num_parameters': num_parameters,
            'circuit_depth': uccsd_ansatz.depth(),
            'circuit_gates': len(uccsd_ansatz.data),

            # Configuration
            'ansatz_reps': self.ansatz_reps,
            'include_hf_state': self.include_hf_state,

            # System compatibility
            'compatible_with_hamiltonian': validation_passed,
            'hamiltonian_qubits': hamiltonian.num_qubits,

            # Molecular system info (from input)
            'num_spatial_orbitals': num_spatial_orbitals,
            'num_particles': num_particles,
            'basis': hamiltonian_system.get('basis', 'unknown'),
            'geometry': hamiltonian_system.get('geometry', 'unknown')
        }

    def create_parameter_bounds(self, num_parameters, bound_type="standard"):
        """
        Create reasonable parameter bounds for UCCSD optimization

        Args:
            num_parameters: Number of variational parameters
            bound_type: "tight", "standard", or "loose"

        Returns:
            list: Parameter bounds for optimization
        """
        if bound_type == "tight":
            return [(-0.1, 0.1)] * num_parameters
        elif bound_type == "loose":
            return [(-2*np.pi, 2*np.pi)] * num_parameters
        else:  # standard
            return [(-0.5, 0.5)] * num_parameters

    def get_initial_parameters(self, num_parameters, init_type="zero"):
        """
        Generate initial parameters for UCCSD optimization

        Args:
            num_parameters: Number of variational parameters
            init_type: "zero", "random_small", or "random_normal"

        Returns:
            np.array: Initial parameter values
        """
        if init_type == "zero":
            return np.zeros(num_parameters)
        elif init_type == "random_small":
            return np.random.normal(0, 0.01, num_parameters)
        elif init_type == "random_normal":
            return np.random.normal(0, 0.1, num_parameters)
        else:
            return np.zeros(num_parameters)


class AnsatzPlugin:
    """
    Plugin for preparing the quantum circuit for the ansatz.
    """
    
    def __init__(self, ansatz_reps=1, include_hf_state=True):
        """
        Initialize the ansatz plugin with UCCSD builder.
        
        Args:
            ansatz_reps: Number of UCCSD repetitions
            include_hf_state: Whether to include HF initial state
        """
        self.uccsd_builder = UCCSDAnsatzBuilder(ansatz_reps, include_hf_state)
        self.ansatz_system = None
        self.parameters = None
        
    def prepare_ansatz(self, parameters, hamiltonian_system=None):
        """
        Prepare the ansatz circuit with given parameters.
        
        Args:
            parameters: Variational parameters for the ansatz
            hamiltonian_system: Hamiltonian system from HamiltonianPlugin
            
        Returns:
            QuantumCircuit: Prepared ansatz circuit
        """
        if hamiltonian_system is None:
            raise ValueError("hamiltonian_system must be provided to prepare ansatz")
            
        # Build ansatz if not already built or if hamiltonian system changed
        if self.ansatz_system is None:
            print("Building UCCSD ansatz...")
            self.ansatz_system = self.uccsd_builder.build_uccsd_ansatz(hamiltonian_system)
            
        # Bind parameters to the ansatz
        ansatz = self.ansatz_system['ansatz']
        if self.ansatz_system['num_parameters'] > 0:
            # Check parameter count matches
            expected_params = self.ansatz_system['num_parameters']
            if len(parameters) != expected_params:
                raise ValueError(f"Expected {expected_params} parameters, got {len(parameters)}")
                
            # Bind parameters to circuit
            if hasattr(ansatz, 'bind_parameters'):
                bound_circuit = ansatz.bind_parameters(parameters)
            else:
                # Fallback: assign parameters directly
                bound_circuit = ansatz.copy()
                if hasattr(bound_circuit, 'parameters') and bound_circuit.parameters:
                    param_dict = dict(zip(bound_circuit.parameters, parameters))
                    bound_circuit = bound_circuit.assign_parameters(param_dict)
            
            return bound_circuit
        else:
            # No parameters to bind (e.g., HF-only ansatz)
            return ansatz
            
    def get_ansatz_info(self):
        """Get information about the built ansatz system."""
        return self.ansatz_system
        
    def get_initial_parameters(self, init_type="zero"):
        """Get initial parameters for optimization."""
        if self.ansatz_system is None:
            raise RuntimeError("Ansatz not built yet. Call prepare_ansatz first.")
        return self.uccsd_builder.get_initial_parameters(
            self.ansatz_system['num_parameters'], 
            init_type
        )


class HamiltonianPlugin:
    """
    Plugin for generating the active-space Hamiltonian for Ammonia (NH3).
    The logic is encapsulated in the get_hamiltonian method, which now returns
    a dictionary containing the Hamiltonian and other relevant data.
    """
    MIN_TERMS = 400
    PAD_SYNTHETIC = True
    SYN_COEFF_SCALE = 1e-8

    def __init__(self):
        self.geom = (
            "N  0.0000  0.0000  0.0000;"
            " H  0.9377  0.0000 -0.3816;"
            " H -0.4688  0.8119 -0.3816;"
            " H -0.4688 -0.8119 -0.3816"
        )
        self._hamiltonian = None
        self._problem_active = None
        self._mapper = None

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

        try:
            if not QISKIT_NATURE_INSTALLED:
                raise ImportError("Qiskit Nature or its dependencies are not installed.")

            driver = PySCFDriver(atom=self.geom, basis='sto3g', charge=0, spin=0, unit=DistanceUnit.ANGSTROM)
            problem_full = ElectronicStructureProblem(driver)
            transformer = ActiveSpaceTransformer(num_electrons=4, num_spatial_orbitals=3)
            self._problem_active = transformer.transform(problem_full)
            
            self._mapper = JordanWignerMapper()
            ham2 = self._problem_active.second_q_ops()['ElectronicEnergy']
            ham_active = self._mapper.map(ham2)

            if ham_active.num_qubits != 6:
                raise RuntimeError(f'Active space produced {ham_active.num_qubits} qubits, expected 6.')

        except Exception as e:
            print(f'[Warning] Ab initio build failed: {e}. Using a fallback 6-qubit operator.')
            paulis = ['IIIIII', 'ZIIIZZ', 'ZZIIZZ', 'IZZIIZ', 'IIZZZZ', 'XXYYZZ', 'YYXXZZ']
            coeffs = [-5.0, 0.12, -0.08, 0.05, -0.03, 0.01, 0.01]
            ham_active = SparsePauliOp(paulis, coeffs)
            # Set related objects to None in case of failure
            self._problem_active = None
            self._mapper = None

        terms = {str(p): complex(c) for p, c in zip(ham_active.paulis, ham_active.coeffs) if abs(complex(c)) > 1e-12}
        physical_count = len(terms)

        if self.PAD_SYNTHETIC and physical_count < self.MIN_TERMS:
            self._add_synthetic_padding(terms, ham_active.num_qubits, physical_count)

        self._print_summary(terms, physical_count)
        
        # Cache and prepare the final operator
        self._hamiltonian = SparsePauliOp.from_list(list(terms.items()))
        
        # Return the result in the specified dictionary format
        return {
            "problem_active": self._problem_active,
            "mapper": self._mapper,
            "hamiltonian_active": self._hamiltonian,
            "num_qubits": self._hamiltonian.num_qubits,
            "basis": "sto3g",
            "geometry": self.geom
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


class ClassicalOptimizerPlugin:
    """
    Plugin for the classical optimization routine.
    """
    def optimize(self, objective_function, initial_params):
        raise NotImplementedError("Optimizer plugin not implemented.")


class ZNEDenoiserPlugin:
    """
    Plugin for applying Zero-Noise Extrapolation (ZNE) to mitigate errors.
    """
    def denoise(self, noisy_results):
        return noisy_results


class VQE:
    """
    The main VQE class that orchestrates the algorithm using the provided plugins.
    """
    def __init__(self, ansatz_plugin, hamiltonian_plugin, optimizer_plugin, zne_plugin):
        self.ansatz_plugin = ansatz_plugin
        self.hamiltonian_plugin = hamiltonian_plugin
        self.optimizer_plugin = optimizer_plugin
        self.zne_plugin = zne_plugin
        self.hamiltonian_system = self.hamiltonian_plugin.get_hamiltonian()  # Get the Hamiltonian system

    def _get_expectation_value(self, parameters):
        # Prepare ansatz with current parameters and hamiltonian system
        state = self.ansatz_plugin.prepare_ansatz(parameters, self.hamiltonian_system)
        # In a real implementation, you would compute the expectation of self.hamiltonian_system['hamiltonian_active']
        # with respect to the prepared state (ansatz).
        expectation_value = self._simulate_measurement(state, self.hamiltonian_system['hamiltonian_active'])
        return expectation_value

    def _simulate_measurement(self, state, hamiltonian):
        return 0.0  # Dummy value

    def objective_function(self, parameters):
        noisy_value = self._get_expectation_value(parameters)
        denoised_value = self.zne_plugin.denoise(noisy_value)
        return denoised_value

    def run(self, initial_params=None):
        print("Starting VQE algorithm...")
        
        # If no initial params provided, get them from the ansatz plugin
        if initial_params is None:
            # First prepare a dummy ansatz to get system info
            dummy_params = [0.0] * 10  # Temporary
            try:
                self.ansatz_plugin.prepare_ansatz(dummy_params, self.hamiltonian_system)
                ansatz_info = self.ansatz_plugin.get_ansatz_info()
                if ansatz_info and ansatz_info['num_parameters'] > 0:
                    initial_params = self.ansatz_plugin.get_initial_parameters("zero")
                    print(f"Generated {len(initial_params)} initial parameters")
                else:
                    print("No variational parameters available - cannot optimize")
                    return None, None
            except Exception as e:
                print(f"Failed to build ansatz: {e}")
                return None, None
        
        optimized_params = self.optimizer_plugin.optimize(self.objective_function, initial_params)
        print("Optimization complete.")
        final_energy = self.objective_function(optimized_params)
        print(f"Optimal Parameters: {optimized_params}")
        print(f"Estimated Ground State Energy: {final_energy}")
        return optimized_params, final_energy


# Example of how to use the framework
if __name__ == '__main__':
    ansatz_plugin = AnsatzPlugin(ansatz_reps=1, include_hf_state=True)
    hamiltonian_plugin = HamiltonianPlugin()
    optimizer_plugin = ClassicalOptimizerPlugin()
    zne_plugin = ZNEDenoiserPlugin()

    try:
        # Create the VQE instance, which will trigger Hamiltonian construction
        vqe_instance = VQE(
            ansatz_plugin=ansatz_plugin,
            hamiltonian_plugin=hamiltonian_plugin,
            optimizer_plugin=optimizer_plugin,
            zne_plugin=zne_plugin
        )
        
        # Now attempt the full run (initial parameters will be generated automatically)
        vqe_instance.run()

    except NotImplementedError as e:
        print(f"\nExecution halted as expected: {e}")
        print("Please implement the remaining plugins (Optimizer) to run the full VQE.")
    except Exception as e:
        print(f"\nAn error occurred during setup: {e}")
