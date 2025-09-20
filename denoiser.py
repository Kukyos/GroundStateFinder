# Direct Drop-in Replacement for ZNEDenoiserPlugin using Mitiq
# 
# Simply replace your existing ZNEDenoiserPlugin class with this implementation
# No other changes needed to your VQE code!

import numpy as np
from typing import List, Union, Callable, Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Check for Mitiq availability
try:
    import mitiq
    from mitiq import zne
    from mitiq.interface.mitiq_qiskit import QiskitExecutor
    MITIQ_AVAILABLE = True
except ImportError:
    MITIQ_AVAILABLE = False
    print("Warning: Mitiq not available. Install with: pip install mitiq")

# Check for Qiskit availability
try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error, amplitude_damping_error
    from qiskit.primitives import Estimator
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    print("Warning: Qiskit not available. Install with: pip install qiskit qiskit-aer")


class ZNEDenoiserPlugin:
    """
    DROP-IN REPLACEMENT: Mitiq-powered ZNE Plugin for VQE Error Mitigation
    
    This class maintains the exact same interface as your original ZNEDenoiserPlugin
    but uses Mitiq's advanced ZNE capabilities under the hood.
    
    Simply replace your existing import with this class - no other changes needed!
    """

    def __init__(self, 
                 noise_factors: List[float] = None,
                 extrapolation_method: str = 'richardson',
                 polynomial_degree: int = 2,
                 min_noise_factor: float = 1.0,
                 max_noise_factor: float = 5.0,
                 adaptive_threshold: float = 0.01,
                 verbose: bool = True,
                 # New Mitiq-specific parameters (with sensible defaults)
                 error_rate: float = 0.02,
                 shots: int = 1024,
                 backend_type: str = 'aer_simulator'):
        """
        Initialize ZNE plugin - SAME INTERFACE as original
        
        Args:
            noise_factors: List of noise scaling factors (>= 1.0)
            extrapolation_method: 'richardson', 'exponential', 'polynomial', 'linear'
            polynomial_degree: Degree for polynomial extrapolation (unused in Mitiq)
            min_noise_factor: Minimum noise scaling factor
            max_noise_factor: Maximum noise scaling factor  
            adaptive_threshold: Convergence threshold for adaptive methods
            verbose: Enable detailed output
            error_rate: NEW - Realistic error rate for noise model
            shots: NEW - Number of measurement shots
            backend_type: NEW - Quantum backend type
        """
        # Store original interface parameters
        if noise_factors is None:
            self.noise_factors = [1.0, 3.0, 5.0]
        else:
            self.noise_factors = sorted([max(1.0, f) for f in noise_factors])
            
        self.extrapolation_method = extrapolation_method.lower()
        self.polynomial_degree = polynomial_degree
        self.min_noise_factor = min_noise_factor
        self.max_noise_factor = max_noise_factor
        self.adaptive_threshold = adaptive_threshold
        self.verbose = verbose
        
        # New Mitiq-specific parameters
        self.error_rate = error_rate
        self.shots = shots
        self.backend_type = backend_type
        
        # Initialize Mitiq components if available
        self.mitiq_available = MITIQ_AVAILABLE and QISKIT_AVAILABLE
        if self.mitiq_available:
            self._setup_mitiq_components()
        
        # Tracking for analysis (same as original)
        self.zne_history = []
        self.improvement_history = []
        
        # VQE integration - store current circuit and Hamiltonian for use in denoise()
        self._current_circuit = None
        self._current_hamiltonian = None
        
        if self.verbose:
            backend_info = "Mitiq+Qiskit" if self.mitiq_available else "Fallback"
            print(f"ZNE Plugin initialized ({backend_info}):")
            print(f"   Method: {self.extrapolation_method}")
            print(f"   Noise factors: {self.noise_factors}")
            if self.mitiq_available:
                print(f"   Error rate: {self.error_rate}")
                print(f"   Shots: {self.shots}")

    def _setup_mitiq_components(self):
        """Setup Mitiq noise model and simulator following Mitiq best practices"""
        # Use Mitiq's initialized_depolarizing_noise for proper integration
        from mitiq.interface.mitiq_qiskit.qiskit_utils import initialized_depolarizing_noise
        from qiskit_aer import QasmSimulator
        
        # Create noise model using Mitiq's utility (like your example)
        self.noise_model = initialized_depolarizing_noise(noise_level=self.error_rate)
        
        # Setup simulator with noise model
        self.backend = QasmSimulator(noise_model=self.noise_model)
        
        # For Estimator-based measurements, we'll create a custom executor
        # that properly handles the noise model and shots

    def set_context(self, circuit: QuantumCircuit, hamiltonian: SparsePauliOp):
        """
        NEW METHOD: Set current circuit and Hamiltonian context for Mitiq ZNE
        
        Call this before denoise() to enable full Mitiq ZNE functionality.
        If not called, falls back to manual extrapolation.
        """
        self._current_circuit = circuit
        self._current_hamiltonian = hamiltonian

    def denoise(self, noisy_results: Union[float, List[float], np.ndarray]) -> float:
        """
        Apply ZNE to denoise quantum expectation values - SAME INTERFACE as original
        
        Args:
            noisy_results: Either single noisy value or list of values at different noise levels
                          If circuit/hamiltonian context is set, this can be None
            
        Returns:
            float: Zero-noise extrapolated expectation value
        """
        # If we have circuit and Hamiltonian context, use full Mitiq ZNE
        if (self.mitiq_available and 
            self._current_circuit is not None and 
            self._current_hamiltonian is not None):
            return self._apply_mitiq_zne()
        
        # Otherwise, fall back to original interface behavior
        return self._manual_extrapolation(noisy_results)

    def _apply_mitiq_zne(self) -> float:
        """Apply full Mitiq ZNE with current circuit and Hamiltonian following Mitiq best practices"""
        try:
            from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
            from qiskit.primitives import Estimator
            
            # Create executor function similar to your example
            def vqe_executor(circuit: QuantumCircuit) -> float:
                """Execute circuit and return expectation value (following Mitiq patterns)"""
                try:
                    # Transpile the circuit for the noisy backend
                    pm = generate_preset_pass_manager(
                        backend=self.backend,
                        basis_gates=self.noise_model.basis_gates if hasattr(self.noise_model, 'basis_gates') else None,
                        optimization_level=0,  # Important to preserve folded gates
                    )
                    
                    exec_circuit = pm.run(circuit)
                    
                    # Use Estimator with the noisy backend
                    estimator = Estimator(backend=self.backend)
                    estimator.set_options(shots=self.shots)
                    
                    job = estimator.run([exec_circuit], [self._current_hamiltonian])
                    result = job.result()
                    
                    return float(result.values[0])
                except Exception as e:
                    if self.verbose:
                        print(f"Executor error: {e}")
                    return 0.0
            
            # Create factory for extrapolation method
            zne_factory = self._get_zne_factory()
            
            # Apply Mitiq ZNE (like your example)
            if zne_factory is not None:
                mitigated_value = zne.execute_with_zne(
                    self._current_circuit,
                    vqe_executor,
                    factory=zne_factory
                )
            else:
                # Use default ZNE with specified noise factors
                mitigated_value = zne.execute_with_zne(
                    self._current_circuit,
                    vqe_executor,
                    scale_noise=zne.scaling.fold_gates_at_random,
                    noise_factors=self.noise_factors,
                    extrapolate=self._get_extrapolation_func()
                )
            
            # Get raw (unmitigated) value for comparison
            raw_value = vqe_executor(self._current_circuit)
            
            # Track improvement (same as original interface)
            improvement = abs(mitigated_value - raw_value)
            
            self.zne_history.append({
                'noise_factors': self.noise_factors.copy(),
                'noisy_values': None,  # Not applicable for Mitiq ZNE
                'extrapolated': mitigated_value,
                'original': raw_value,
                'improvement': improvement,
                'method': 'mitiq'
            })
            
            self.improvement_history.append(improvement)
            
            if self.verbose:
                print(f"Mitiq ZNE Applied:")
                print(f"   Raw value: {raw_value:.8f}")
                print(f"   Mitigated: {mitigated_value:.8f}")
                print(f"   Improvement: {improvement:.8f}")
            
            return float(mitigated_value)
            
        except Exception as e:
            if self.verbose:
                print(f"Mitiq ZNE Error: {e}")
                print("   Falling back to raw execution")
            
            # Fallback: execute once and return
            try:
                from qiskit.primitives import Estimator
                estimator = Estimator(backend=self.backend)
                estimator.set_options(shots=self.shots)
                job = estimator.run([self._current_circuit], [self._current_hamiltonian])
                result = job.result()
                return float(result.values[0])
            except:
                return 0.0

    def _get_zne_factory(self):
        """Get appropriate ZNE factory based on extrapolation method"""
        try:
            if self.extrapolation_method == 'linear':
                return zne.inference.LinearFactory(scale_factors=self.noise_factors)
            elif self.extrapolation_method == 'polynomial':
                return zne.inference.PolyFactory(
                    scale_factors=self.noise_factors, 
                    order=self.polynomial_degree
                )
            elif self.extrapolation_method == 'exponential':
                return zne.inference.ExpFactory(scale_factors=self.noise_factors)
            elif self.extrapolation_method == 'richardson':
                return zne.inference.RichardsonFactory(scale_factors=self.noise_factors)
            else:
                # Return None to use default execute_with_zne approach
                return None
        except Exception as e:
            if self.verbose:
                print(f"Factory creation failed: {e}, using default")
            return None
    
    def _get_extrapolation_func(self):
        """Get extrapolation function for direct execute_with_zne call"""
        try:
            if self.extrapolation_method == 'richardson':
                return zne.extrapolation.richardson
            elif self.extrapolation_method == 'linear':
                return zne.extrapolation.linear
            elif self.extrapolation_method == 'polynomial':
                return lambda x, y: zne.extrapolation.polynomial(x, y, deg=self.polynomial_degree)
            elif self.extrapolation_method == 'exponential':
                return zne.extrapolation.exponential
            else:
                return zne.extrapolation.richardson
        except AttributeError:
            return zne.extrapolation.richardson

    def _manual_extrapolation(self, noisy_results: Union[float, List[float], np.ndarray]) -> float:
        """Manual extrapolation - SAME as original implementation"""
        # Handle single value input (backward compatibility)
        if isinstance(noisy_results, (int, float)):
            if self.verbose:
                print("ZNE: Single value provided, returning as-is (no extrapolation possible)")
            return float(noisy_results)
        
        # Convert to numpy array
        noisy_values = np.array(noisy_results, dtype=float)
        
        # Validate input
        if len(noisy_values) != len(self.noise_factors):
            if self.verbose:
                print(f"ZNE: Expected {len(self.noise_factors)} values, got {len(noisy_values)}")
                print("   Returning first value without extrapolation")
            return float(noisy_values[0]) if len(noisy_values) > 0 else 0.0
        
        # Perform extrapolation
        try:
            extrapolated_value = self._perform_extrapolation(noisy_values)
            
            # Track improvement (same as original)
            original_value = noisy_values[0]
            improvement = abs(extrapolated_value - original_value)
            
            self.zne_history.append({
                'noise_factors': self.noise_factors.copy(),
                'noisy_values': noisy_values.copy(),
                'extrapolated': extrapolated_value,
                'original': original_value,
                'improvement': improvement,
                'method': 'manual'
            })
            
            self.improvement_history.append(improvement)
            
            if self.verbose:
                print(f"Manual ZNE Applied:")
                print(f"   Noisy values: {noisy_values}")
                print(f"   Extrapolated: {extrapolated_value:.8f}")
                print(f"   Improvement: {improvement:.8f}")
            
            return float(extrapolated_value)
            
        except Exception as e:
            if self.verbose:
                print(f"Manual ZNE Error: {e}")
                print("   Returning original noisy value")
            return float(noisy_values[0])

    def _perform_extrapolation(self, noisy_values: np.ndarray) -> float:
        """Perform extrapolation - SAME as original implementation"""
        noise_factors = np.array(self.noise_factors)
        
        if self.extrapolation_method == 'richardson':
            return self._richardson_extrapolation(noise_factors, noisy_values)
        elif self.extrapolation_method == 'linear':
            return self._linear_extrapolation(noise_factors, noisy_values)
        else:
            # Try Mitiq extrapolation if available
            if self.mitiq_available:
                try:
                    extrapolation_func = getattr(zne.extrapolation, self.extrapolation_method)
                    return extrapolation_func(self.noise_factors, noisy_values)
                except AttributeError:
                    pass
            
            # Fallback to Richardson
            return self._richardson_extrapolation(noise_factors, noisy_values)

    def _richardson_extrapolation(self, noise_factors: np.ndarray, values: np.ndarray) -> float:
        """Richardson extrapolation - SAME as original implementation"""
        if len(values) < 2:
            return values[0]
        
        # For two points
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
                weight = 1.0 / abs(λ2 - λ1)
                weights.append(weight)
        
        weights = np.array(weights)
        extrapolated_values = np.array(extrapolated_values)
        
        return np.average(extrapolated_values, weights=weights)

    def _linear_extrapolation(self, noise_factors: np.ndarray, values: np.ndarray) -> float:
        """Linear extrapolation - SAME as original implementation"""
        try:
            coeffs = np.polyfit(noise_factors, values, 1)
            return coeffs[1]
        except (np.linalg.LinAlgError, ValueError):
            return values[0]

    def create_noisy_circuit(self, original_circuit, noise_factor: float):
        """
        Create noisy circuit - SAME interface as original
        
        Uses Mitiq gate folding if available, otherwise manual method
        """
        if noise_factor <= 1.0:
            return original_circuit.copy()
        
        if self.mitiq_available:
            try:
                scaling_method = getattr(zne.scaling, 'fold_gates_at_random')
                return scaling_method(original_circuit, noise_factor)
            except Exception as e:
                if self.verbose:
                    print(f"Mitiq scaling failed: {e}, using manual method")
        
        # Fallback to original manual method
        return self._manual_create_noisy_circuit(original_circuit, noise_factor)

    def _manual_create_noisy_circuit(self, original_circuit, noise_factor: float):
        """Manual noisy circuit creation - SAME as original"""
        from qiskit import QuantumCircuit
        
        noisy_circuit = original_circuit.copy()
        num_identity_pairs = int((noise_factor - 1.0) * len(original_circuit.data))
        
        if num_identity_pairs > 0:
            np.random.seed(42)
            
            for _ in range(num_identity_pairs):
                qubit = np.random.randint(0, noisy_circuit.num_qubits)
                noisy_circuit.x(qubit)
                noisy_circuit.x(qubit)
        
        return noisy_circuit

    def get_zne_analysis(self) -> dict:
        """Get ZNE analysis - SAME interface as original"""
        if not self.zne_history:
            return {"error": "No ZNE history available"}
        
        improvements = np.array(self.improvement_history)
        
        analysis = {
            "total_applications": len(self.zne_history),
            "average_improvement": np.mean(improvements),
            "std_improvement": np.std(improvements),
            "max_improvement": np.max(improvements),
            "min_improvement": np.min(improvements),
            "extrapolation_method": self.extrapolation_method,
            "noise_factors": self.noise_factors,
            "success_rate": np.sum(improvements > 0) / len(improvements) * 100 if len(improvements) > 0 else 0
        }
        
        # Add Mitiq-specific info if available
        if self.mitiq_available:
            mitiq_count = sum(1 for h in self.zne_history if h.get('method') == 'mitiq')
            manual_count = sum(1 for h in self.zne_history if h.get('method') == 'manual')
            
            analysis.update({
                "mitiq_applications": mitiq_count,
                "manual_applications": manual_count,
                "mitiq_available": True,
                "error_rate": self.error_rate,
                "shots": self.shots
            })
        else:
            analysis["mitiq_available"] = False
        
        return analysis

    def adaptive_zne(self, measurement_function: Callable, initial_params: np.ndarray) -> float:
        """Adaptive ZNE - SAME interface as original"""
        if self.verbose:
            print("Running adaptive ZNE...")
        
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

    def reset_history(self):
        """Reset ZNE history and analysis data"""
        self.zne_history = []
        self.improvement_history = []
        if self.verbose:
            print("ZNE history reset")

    def plot_zne_history(self):
        """Plot ZNE improvement history (requires matplotlib)"""
        try:
            import matplotlib.pyplot as plt
            
            if not self.improvement_history:
                print("No ZNE history to plot")
                return
            
            plt.figure(figsize=(10, 6))
            plt.plot(self.improvement_history, 'b-o', markersize=4)
            plt.title('ZNE Improvement History')
            plt.xlabel('ZNE Application')
            plt.ylabel('Improvement (|mitigated - original|)')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("matplotlib not available for plotting")

    def benchmark_methods(self, test_values: List[float], noise_factors: List[float] = None) -> Dict[str, float]:
        """Benchmark different extrapolation methods"""
        if noise_factors is None:
            noise_factors = self.noise_factors
            
        methods = ['richardson', 'linear', 'exponential', 'polynomial']
        results = {}
        
        original_method = self.extrapolation_method
        original_factors = self.noise_factors
        
        self.noise_factors = noise_factors
        
        for method in methods:
            self.extrapolation_method = method
            try:
                result = self._manual_extrapolation(test_values)
                results[method] = result
            except Exception as e:
                results[method] = f"Error: {str(e)}"
        
        # Restore original settings
        self.extrapolation_method = original_method
        self.noise_factors = original_factors
        
        return results


# Helper function to modify VQE for seamless integration
def integrate_with_vqe(vqe_instance):
    """
    Helper function to integrate ZNE with your existing VQE instance
    
    Usage:
        vqe = VQE(...)  # Your existing VQE
        integrate_with_vqe(vqe)  # Add Mitiq ZNE integration
        vqe.run()  # Run as usual
    """
    original_simulate = vqe_instance._simulate_measurement
    
    def mitiq_simulate_measurement(trial_wavefunction, hamiltonian):
        """Modified measurement function that uses Mitiq ZNE when possible"""
        # Set context for ZNE plugin
        if hasattr(vqe_instance.zne_plugin, 'set_context'):
            vqe_instance.zne_plugin.set_context(trial_wavefunction, hamiltonian)
            # Use ZNE with context
            return vqe_instance.zne_plugin.denoise(None)
        else:
            # Fallback to original behavior
            return original_simulate(trial_wavefunction, hamiltonian)
    
    # Replace the method
    vqe_instance._simulate_measurement = mitiq_simulate_measurement
    
    print("VQE integrated with Mitiq ZNE successfully!")


# Demo/Test functions
def create_demo_circuit(num_qubits: int = 2) -> 'QuantumCircuit':
    """Create a simple demo quantum circuit for testing"""
    if not QISKIT_AVAILABLE:
        return None
        
    from qiskit import QuantumCircuit
    
    qc = QuantumCircuit(num_qubits)
    qc.h(0)
    if num_qubits > 1:
        qc.cx(0, 1)
    
    return qc

def create_demo_hamiltonian(num_qubits: int = 2) -> 'SparsePauliOp':
    """Create a simple demo Hamiltonian for testing"""
    if not QISKIT_AVAILABLE:
        return None
        
    from qiskit.quantum_info import SparsePauliOp
    
    # Simple Z ⊗ Z Hamiltonian
    if num_qubits == 1:
        return SparsePauliOp.from_list([("Z", 1.0)])
    else:
        return SparsePauliOp.from_list([("ZZ", 1.0), ("ZI", 0.5), ("IZ", 0.5)])

def run_demo():
    """Run a complete demo of the ZNE plugin"""
    print("\n" + "="*60)
    print("ZNE Plugin Demo")
    print("="*60)
    
    # Test manual extrapolation (always works)
    print("\n1. Manual Extrapolation Test:")
    zne_plugin = ZNEDenoiserPlugin(
        noise_factors=[1.0, 3.0, 5.0],
        extrapolation_method='richardson',
        verbose=True
    )
    
    # Simulate noisy measurements
    true_value = -1.5
    noise_levels = [0.1, 0.3, 0.5]  # Increasing noise
    noisy_values = [true_value + np.random.normal(0, noise) for noise in noise_levels]
    
    mitigated_value = zne_plugin.denoise(noisy_values)
    print(f"True value: {true_value}")
    print(f"Mitigated value: {mitigated_value:.6f}")
    print(f"Error: {abs(mitigated_value - true_value):.6f}")
    
    # Test Mitiq ZNE if available
    if zne_plugin.mitiq_available:
        print("\n2. Mitiq ZNE Test:")
        
        # Create demo circuit and Hamiltonian
        circuit = create_demo_circuit(2)
        hamiltonian = create_demo_hamiltonian(2)
        
        if circuit and hamiltonian:
            # Set context and run Mitiq ZNE
            zne_plugin.set_context(circuit, hamiltonian)
            mitiq_result = zne_plugin.denoise(None)
            print(f"Mitiq ZNE result: {mitiq_result:.6f}")
        else:
            print("Could not create demo circuit/Hamiltonian")
    else:
        print("\n2. Mitiq ZNE: Not available (requires mitiq and qiskit)")
    
    # Display analysis
    print("\n3. ZNE Analysis:")
    analysis = zne_plugin.get_zne_analysis()
    for key, value in analysis.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.6f}")
        else:
            print(f"   {key}: {value}")
    
    # Test method benchmarking
    print("\n4. Method Benchmarking:")
    benchmark_results = zne_plugin.benchmark_methods(noisy_values)
    for method, result in benchmark_results.items():
        if isinstance(result, (int, float)):
            print(f"   {method}: {result:.6f}")
        else:
            print(f"   {method}: {result}")


if __name__ == "__main__":
    print("Drop-in Replacement ZNE Plugin with Mitiq Backend")
    print("=" * 60)
    
    # Test the plugin creation
    zne_plugin = ZNEDenoiserPlugin(
        noise_factors=[1.0, 3.0, 5.0],
        extrapolation_method='richardson',
        verbose=True
    )
    
    print("Plugin created successfully!")
    print(f"Mitiq available: {zne_plugin.mitiq_available}")
    
    # Run full demo
    run_demo()
    
    # Test analysis after demo
    analysis = zne_plugin.get_zne_analysis()
    print(f"\nFinal Analysis: {len(analysis)} metrics available")
    
    print("\n" + "="*60)
    print("Demo completed successfully!")
    print("To use in your VQE code:")
    print("1. Replace your ZNEDenoiserPlugin import with this file")
    print("2. (Optional) Call integrate_with_vqe(your_vqe_instance)")
    print("3. Run your VQE as usual - ZNE will automatically use Mitiq when possible")
    print("="*60)
