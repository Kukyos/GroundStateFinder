import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from hamiltonian import HamiltonianPlugin

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
if __name__ == "__main__":
    print("🚀 Testing AnsatzPlugin integration...")
    
    # Step 1: Create Hamiltonian
    h_plugin = HamiltonianPlugin(auto_active=True, active_electrons=4, active_orbitals=3)
    hamiltonian_system = h_plugin.get_hamiltonian()
    
    print(f"🔬 Hamiltonian created with {hamiltonian_system['num_qubits']} qubits")
    print(f"🧪 Molecule: {hamiltonian_system.get('geometry', 'Unknown')}")
    
    # Step 2: Build Ansatz
    a_plugin = AnsatzPlugin(ansatz_reps=1, include_hf_state=True, verbose=True)
    success = a_plugin.build_from_hamiltonian(hamiltonian_system)
    
    if success:
        print("\n" + "="*60)
        print("🎯 ANSATZ BUILD SUCCESS!")
        print("="*60)
        
        # Get info
        info = a_plugin.get_ansatz_info()
        print(f"✅ VQE Ready: {info['vqe_ready']}")
        print(f"🎛️  Parameters: {info['num_parameters']}")
        print(f"📏 Circuit Depth: {info['circuit_depth']}")
        print(f"🚪 Gates: {info['circuit_gates']}")
        print(f"⚛️  Qubits: {info['num_qubits']}")
        
        # Display the quantum circuit
        print("\n" + "="*60)
        print("🎨 QUANTUM CIRCUIT VISUALIZATION")
        print("="*60)
        
        try:
            # Get the ansatz circuit
            circuit = a_plugin.ansatz_circuit
            
            # Print circuit as text
            print("Circuit (text representation):")
            print("-" * 40)
            print(circuit)
            print("-" * 40)
            
            # Try to create a visual diagram (this works in Jupyter/Colab)
            try:
                print("\n🎭 Circuit Diagram:")
                circuit_diagram = circuit.draw(output='text', fold=120)
                print(circuit_diagram)
            except:
                print("Circuit diagram not available in this environment")
            
            # Show circuit statistics
            print(f"\n📊 Circuit Statistics:")
            print(f"   • Total gates: {len(circuit.data)}")
            print(f"   • Depth: {circuit.depth()}")
            print(f"   • Width (qubits): {circuit.num_qubits}")
            print(f"   • Parameters: {len(circuit.parameters)}")
            
            # Show gate breakdown
            gate_counts = {}
            for instruction in circuit.data:
                gate_name = instruction.operation.name
                gate_counts[gate_name] = gate_counts.get(gate_name, 0) + 1
            
            print(f"\n🔧 Gate Breakdown:")
            for gate, count in sorted(gate_counts.items()):
                print(f"   • {gate}: {count}")
            
        except Exception as e:
            print(f"Could not display circuit: {e}")
        
        # Get and display initial parameters
        if info['num_parameters'] > 0:
            initial_params = a_plugin.get_initial_parameters("random_small")
            print(f"\n🎲 Initial Parameters ({len(initial_params)} total):")
            if len(initial_params) <= 10:
                for i, param in enumerate(initial_params):
                    print(f"   θ_{i}: {param:.6f}")
            else:
                print(f"   First 5: {[f'{p:.6f}' for p in initial_params[:5]]}")
                print(f"   Last 5:  {[f'{p:.6f}' for p in initial_params[-5:]]}")
            
            # Show parameter bounds
            bounds = a_plugin.get_parameter_bounds("standard")
            print(f"📏 Parameter bounds: [{bounds[0][0]:.1f}, {bounds[0][1]:.1f}]")
            
    else:
        print("❌ Ansatz build failed!")
        
    print("\n🏁 Test complete!")


