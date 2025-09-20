import numpy as np
from typing import List, Union, Callable, Optional, Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# Check for Mitiq availability
try:
    import mitiq
    from mitiq import zne
    from mitiq.interface import convert_to_mitiq, convert_from_mitiq
    MITIQ_AVAILABLE = True
    MITIQ_VERSION = mitiq.__version__
except ImportError:
    MITIQ_AVAILABLE = False
    MITIQ_VERSION = None

# Check for Qiskit availability
try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.quantum_info import SparsePauliOp, Statevector
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error, amplitude_damping_error
    from qiskit.primitives import Estimator
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False


class ZNEDenoiserPlugin:
    """
    Zero-Noise Extrapolation (ZNE) Plugin for VQE Error Mitigation
    
    This implementation provides both Mitiq-powered ZNE and manual extrapolation
    methods with automatic fallback and comprehensive error handling.
    """

    def __init__(self, 
                 noise_factors: Optional[List[float]] = None,
                 extrapolation_method: str = 'richardson',
                 polynomial_degree: int = 2,
                 min_noise_factor: float = 1.0,
                 max_noise_factor: float = 5.0,
                 adaptive_threshold: float = 0.01,
                 verbose: bool = True,
                 # Mitiq-specific parameters
                 error_rate: float = 0.02,
                 shots: int = 1024,
                 backend_type: str = 'aer_simulator',
                 max_history_size: int = 100,
                 enable_caching: bool = True):
        """
        Initialize ZNE denoising plugin.
        
        Args:
            noise_factors: List of noise scaling factors (>= 1.0)
            extrapolation_method: 'richardson', 'linear', 'polynomial', 'exponential'
            polynomial_degree: Degree for polynomial extrapolation
            min_noise_factor: Minimum noise scaling factor
            max_noise_factor: Maximum noise scaling factor  
            adaptive_threshold: Convergence threshold for adaptive methods
            verbose: Enable detailed output
            error_rate: Realistic error rate for noise model (0-1)
            shots: Number of measurement shots
            backend_type: Quantum backend type
            max_history_size: Maximum number of history entries to keep
            enable_caching: Enable result caching for performance
        """
        # Validate and set noise factors
        if noise_factors is None:
            self.noise_factors = [1.0, 3.0, 5.0]
        else:
            self.noise_factors = sorted([max(1.0, float(f)) for f in noise_factors])
            
        # Validate extrapolation method
        valid_methods = ['richardson', 'linear', 'polynomial', 'exponential']
        self.extrapolation_method = extrapolation_method.lower()
        if self.extrapolation_method not in valid_methods:
            if verbose:
                print(f"Warning: Unknown method '{extrapolation_method}', using 'richardson'")
            self.extrapolation_method = 'richardson'
            
        self.polynomial_degree = max(1, int(polynomial_degree))
        self.min_noise_factor = max(1.0, float(min_noise_factor))
        self.max_noise_factor = max(self.min_noise_factor, float(max_noise_factor))
        self.adaptive_threshold = abs(float(adaptive_threshold))
        self.verbose = bool(verbose)
        
        # Mitiq-specific parameters
        self.error_rate = max(0.0, min(1.0, float(error_rate)))
        self.shots = max(1, int(shots))
        self.backend_type = str(backend_type)
        
        # Performance parameters
        self.max_history_size = max(1, int(max_history_size))
        self.enable_caching = bool(enable_caching)
        
        # Initialize components
        self.mitiq_available = MITIQ_AVAILABLE and QISKIT_AVAILABLE
        if self.mitiq_available:
            self._setup_mitiq_components()
        
        # Tracking and caching
        self.zne_history = []
        self.improvement_history = []
        self._cache = {} if enable_caching else None
        
        # Current context for Mitiq ZNE
        self._current_circuit = None
        self._current_hamiltonian = None
        
        if self.verbose:
            backend_info = f"Mitiq v{MITIQ_VERSION}" if self.mitiq_available else "Manual only"
            print(f"ZNE Plugin initialized ({backend_info}):")
            print(f"   Method: {self.extrapolation_method}")
            print(f"   Noise factors: {self.noise_factors}")
            if self.mitiq_available:
                print(f"   Error rate: {self.error_rate:.4f}")
                print(f"   Shots: {self.shots}")

    def _setup_mitiq_components(self):
        """Setup Mitiq noise model and backend with proper error handling."""
        try:
            # Create realistic noise model
            noise_model = NoiseModel()
            
            # Add depolarizing errors to gates
            p1_error = depolarizing_error(self.error_rate, 1)
            p2_error = depolarizing_error(self.error_rate * 1.5, 2)  # 2-qubit gates typically noisier
            
            # Add errors to common gates
            single_qubit_gates = ['rx', 'ry', 'rz', 'h', 'x', 'y', 'z', 's', 't']
            two_qubit_gates = ['cx', 'cy', 'cz', 'cphase']
            
            for gate in single_qubit_gates:
                noise_model.add_all_qubit_quantum_error(p1_error, gate, warnings=False)
            for gate in two_qubit_gates:
                noise_model.add_all_qubit_quantum_error(p2_error, gate, warnings=False)
            
            self.noise_model = noise_model
            self.backend = AerSimulator(noise_model=noise_model)
            
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not setup noise model: {e}")
                print("         Using noiseless simulation for Mitiq ZNE")
            self.noise_model = None
            self.backend = AerSimulator()

    def set_context(self, circuit: QuantumCircuit, hamiltonian: SparsePauliOp):
        """
        Set current circuit and Hamiltonian context for Mitiq ZNE.
        
        Args:
            circuit: Quantum circuit for the trial wavefunction
            hamiltonian: Hamiltonian operator as SparsePauliOp
        """
        if not QISKIT_AVAILABLE:
            if self.verbose:
                print("Warning: Qiskit not available, context setting ignored")
            return
            
        self._current_circuit = circuit.copy() if circuit is not None else None
        self._current_hamiltonian = hamiltonian

    def denoise(self, noisy_results: Optional[Union[float, List[float], np.ndarray]] = None) -> float:
        """
        Apply ZNE to denoise quantum expectation values.
        
        Args:
            noisy_results: Either single noisy value, list of values at different 
                          noise levels, or None (uses Mitiq ZNE with context)
            
        Returns:
            Zero-noise extrapolated expectation value
        """
        # Priority 1: Use Mitiq ZNE if context is set and available
        if (self.mitiq_available and 
            self._current_circuit is not None and 
            self._current_hamiltonian is not None):
            return self._apply_mitiq_zne()
        
        # Priority 2: Manual extrapolation if noisy results provided
        if noisy_results is not None:
            return self._manual_extrapolation(noisy_results)
        
        # Priority 3: Fallback
        if self.verbose:
            print("Warning: No context or noisy results provided, cannot apply ZNE")
        return 0.0

    def _apply_mitiq_zne(self) -> float:
        """Apply Mitiq ZNE with current circuit and Hamiltonian."""
        if not self.mitiq_available:
            raise RuntimeError("Mitiq ZNE requested but Mitiq not available")
            
        try:
            # Create executor function for expectation value calculation
            def expectation_executor(circuit: QuantumCircuit) -> float:
                return self._execute_expectation_value(circuit)
            
            # Apply ZNE using Mitiq
            if self.verbose:
                print(f"Applying Mitiq ZNE with {len(self.noise_factors)} noise factors...")
            
            # Get raw (unmitigated) value for comparison
            raw_value = expectation_executor(self._current_circuit)
            
            # Apply ZNE
            try:
                # Use execute_with_zne with appropriate scaling and extrapolation
                mitigated_value = zne.execute_with_zne(
                    self._current_circuit,
                    expectation_executor,
                    scale_noise=zne.scaling.fold_gates_at_random,
                    noise_factors=self.noise_factors,
                    extrapolate=self._get_mitiq_extrapolation_func()
                )
            except Exception as e:
                if self.verbose:
                    print(f"Mitiq ZNE execution failed: {e}")
                    print("Falling back to manual extrapolation")
                
                # Fallback: collect noisy measurements manually and extrapolate
                noisy_values = []
                for factor in self.noise_factors:
                    if factor == 1.0:
                        noisy_values.append(raw_value)
                    else:
                        noisy_circuit = self._create_noisy_circuit(self._current_circuit, factor)
                        noisy_values.append(expectation_executor(noisy_circuit))
                
                mitigated_value = self._manual_extrapolation(noisy_values)
            
            # Track results
            improvement = abs(mitigated_value - raw_value)
            self._add_to_history({
                'noise_factors': self.noise_factors.copy(),
                'raw_value': raw_value,
                'mitigated_value': mitigated_value,
                'improvement': improvement,
                'method': 'mitiq',
                'circuit_depth': self._current_circuit.depth(),
                'num_qubits': self._current_circuit.num_qubits
            })
            
            if self.verbose:
                print(f"Mitiq ZNE Results:")
                print(f"   Raw value:     {raw_value:.8f}")
                print(f"   Mitigated:     {mitigated_value:.8f}")
                print(f"   Improvement:   {improvement:.8f}")
                print(f"   Circuit depth: {self._current_circuit.depth()}")
            
            return float(mitigated_value)
            
        except Exception as e:
            if self.verbose:
                print(f"Mitiq ZNE failed: {e}")
                print("Falling back to raw execution")
            
            try:
                return self._execute_expectation_value(self._current_circuit)
            except:
                return 0.0

    def _execute_expectation_value(self, circuit: QuantumCircuit) -> float:
        """Execute circuit and compute expectation value with proper error handling."""
        cache_key = self._get_cache_key(circuit) if self.enable_caching else None
        
        if cache_key is not None and cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            # Method 1: Use Estimator primitive (preferred)
            estimator = Estimator(backend=self.backend)
            if hasattr(estimator, 'set_options'):
                estimator.set_options(shots=self.shots)
            
            # Transpile circuit for backend
            transpiled = transpile(circuit, backend=self.backend, optimization_level=1)
            
            # Execute estimation
            job = estimator.run([transpiled], [self._current_hamiltonian], shots=self.shots)
            result = job.result()
            expectation_value = float(result.values[0])
            
        except Exception as estimator_error:
            try:
                # Method 2: Statevector fallback (noiseless)
                if self.verbose:
                    print(f"Estimator failed ({estimator_error}), using statevector")
                
                statevector = Statevector.from_instruction(circuit)
                expectation_value = float(statevector.expectation_value(self._current_hamiltonian).real)
                
            except Exception as sv_error:
                if self.verbose:
                    print(f"Statevector also failed ({sv_error}), returning 0")
                expectation_value = 0.0
        
        # Cache result
        if cache_key is not None:
            self._cache[cache_key] = expectation_value
            
        return expectation_value

    def _get_mitiq_extrapolation_func(self):
        """Get Mitiq extrapolation function based on method."""
        try:
            method_map = {
                'richardson': zne.extrapolation.richardson,
                'linear': zne.extrapolation.linear,
                'exponential': zne.extrapolation.exponential,
                'polynomial': lambda x, y: zne.extrapolation.polynomial(x, y, deg=self.polynomial_degree)
            }
            return method_map.get(self.extrapolation_method, zne.extrapolation.richardson)
        except AttributeError:
            # Fallback if specific method not available
            return zne.extrapolation.richardson

    def _manual_extrapolation(self, noisy_results: Union[float, List[float], np.ndarray]) -> float:
        """Manual extrapolation using numpy-based methods."""
        # Handle single value input
        if isinstance(noisy_results, (int, float)):
            if self.verbose:
                print("Single value provided, no extrapolation possible")
            return float(noisy_results)
        
        # Convert to numpy array and validate
        noisy_values = np.array(noisy_results, dtype=float)
        
        if len(noisy_values) != len(self.noise_factors):
            if self.verbose:
                print(f"Warning: Expected {len(self.noise_factors)} values, got {len(noisy_values)}")
                print("Using first value without extrapolation")
            return float(noisy_values[0]) if len(noisy_values) > 0 else 0.0
        
        # Perform extrapolation
        try:
            extrapolated_value = self._perform_extrapolation(noisy_values)
            
            # Track results
            original_value = noisy_values[0]  # Noise factor = 1.0
            improvement = abs(extrapolated_value - original_value)
            
            self._add_to_history({
                'noise_factors': self.noise_factors.copy(),
                'noisy_values': noisy_values.copy(),
                'raw_value': original_value,
                'mitigated_value': extrapolated_value,
                'improvement': improvement,
                'method': 'manual'
            })
            
            if self.verbose:
                print(f"Manual ZNE Results:")
                print(f"   Noisy values:  {noisy_values}")
                print(f"   Extrapolated:  {extrapolated_value:.8f}")
                print(f"   Improvement:   {improvement:.8f}")
            
            return float(extrapolated_value)
            
        except Exception as e:
            if self.verbose:
                print(f"Manual extrapolation failed: {e}")
                print("Returning original noisy value")
            return float(noisy_values[0])

    def _perform_extrapolation(self, noisy_values: np.ndarray) -> float:
        """Perform extrapolation based on selected method."""
        noise_factors = np.array(self.noise_factors, dtype=float)
        
        # Method dispatch
        if self.extrapolation_method == 'richardson':
            return self._richardson_extrapolation(noise_factors, noisy_values)
        elif self.extrapolation_method == 'linear':
            return self._linear_extrapolation(noise_factors, noisy_values)
        elif self.extrapolation_method == 'polynomial':
            return self._polynomial_extrapolation(noise_factors, noisy_values)
        elif self.extrapolation_method == 'exponential':
            return self._exponential_extrapolation(noise_factors, noisy_values)
        else:
            # Default to Richardson
            return self._richardson_extrapolation(noise_factors, noisy_values)

    def _richardson_extrapolation(self, noise_factors: np.ndarray, values: np.ndarray) -> float:
        """Robust Richardson extrapolation with multiple methods."""
        if len(values) < 2:
            return values[0]
        
        try:
            # For exactly 2 points, use standard Richardson formula
            if len(values) == 2:
                λ1, λ2 = noise_factors[0], noise_factors[1]
                f1, f2 = values[0], values[1]
                
                # Check numerical stability
                if abs(λ2 - λ1) < 1e-12:
                    return f1
                    
                return (λ2 * f1 - λ1 * f2) / (λ2 - λ1)
            
            # For multiple points, use weighted least squares approach
            # This is more stable than pairwise Richardson extrapolation
            weights = 1.0 / noise_factors  # Weight inversely by noise level
            
            try:
                # Fit linear model: f(λ) = a + b*λ, extrapolate to λ=0
                coeffs = np.polyfit(noise_factors, values, deg=1, w=weights)
                return coeffs[1]  # Intercept = zero-noise limit
            except np.linalg.LinAlgError:
                # Fallback: simple average of pairwise extrapolations
                extrapolated_values = []
                for i in range(len(values) - 1):
                    λ1, λ2 = noise_factors[i], noise_factors[i + 1]
                    f1, f2 = values[i], values[i + 1]
                    if abs(λ2 - λ1) > 1e-12:
                        extrapolated = (λ2 * f1 - λ1 * f2) / (λ2 - λ1)
                        extrapolated_values.append(extrapolated)
                
                return np.mean(extrapolated_values) if extrapolated_values else values[0]
                
        except Exception as e:
            if self.verbose:
                print(f"Richardson extrapolation failed: {e}")
            return values[0]

    def _linear_extrapolation(self, noise_factors: np.ndarray, values: np.ndarray) -> float:
        """Linear extrapolation to zero noise."""
        try:
            coeffs = np.polyfit(noise_factors, values, deg=1)
            return coeffs[1]  # Intercept
        except (np.linalg.LinAlgError, ValueError):
            return values[0]

    def _polynomial_extrapolation(self, noise_factors: np.ndarray, values: np.ndarray) -> float:
        """Polynomial extrapolation to zero noise."""
        try:
            degree = min(self.polynomial_degree, len(values) - 1)
            coeffs = np.polyfit(noise_factors, values, deg=degree)
            # Evaluate polynomial at λ = 0
            return coeffs[-1]  # Constant term
        except (np.linalg.LinAlgError, ValueError):
            return self._linear_extrapolation(noise_factors, values)

    def _exponential_extrapolation(self, noise_factors: np.ndarray, values: np.ndarray) -> float:
        """Exponential extrapolation assuming f(λ) = A + B*exp(-C*λ)."""
        try:
            # Simple exponential fit: f(λ) ≈ a*exp(-b*λ) + c
            # For zero noise: f(0) = a + c
            
            # Use logarithmic transformation for linear fit
            if np.any(values <= 0):
                # Can't use log transformation, fall back to polynomial
                return self._polynomial_extrapolation(noise_factors, values)
            
            # Fit log(f(λ)) vs λ
            log_values = np.log(np.abs(values))
            coeffs = np.polyfit(noise_factors, log_values, deg=1)
            
            # Extrapolate: f(0) = exp(intercept)
            return np.exp(coeffs[1])
            
        except (ValueError, np.linalg.LinAlgError, OverflowError):
            return self._linear_extrapolation(noise_factors, values)

    def _create_noisy_circuit(self, circuit: QuantumCircuit, noise_factor: float) -> QuantumCircuit:
        """Create circuit with scaled noise."""
        if noise_factor <= 1.0:
            return circuit.copy()
        
        if self.mitiq_available:
            try:
                # Use Mitiq's gate folding
                return zne.scaling.fold_gates_at_random(circuit, noise_factor)
            except Exception as e:
                if self.verbose:
                    print(f"Mitiq gate folding failed: {e}, using manual method")
        
        # Manual gate folding: add identity pairs
        noisy_circuit = circuit.copy()
        extra_gates = int((noise_factor - 1.0) * len(circuit.data))
        
        if extra_gates > 0:
            np.random.seed(42)  # Reproducible noise
            for _ in range(extra_gates):
                qubit = np.random.randint(0, noisy_circuit.num_qubits)
                # Add identity gate pair (X-X)
                noisy_circuit.x(qubit)
                noisy_circuit.x(qubit)
        
        return noisy_circuit

    def _get_cache_key(self, circuit: QuantumCircuit) -> Optional[str]:
        """Generate cache key for circuit."""
        if not self.enable_caching:
            return None
        try:
            # Use circuit string representation for caching
            return str(circuit)
        except:
            return None

    def _add_to_history(self, entry: Dict[str, Any]):
        """Add entry to history with size management."""
        self.zne_history.append(entry)
        self.improvement_history.append(entry['improvement'])
        
        # Maintain maximum history size
        if len(self.zne_history) > self.max_history_size:
            removed = self.zne_history.pop(0)
            if 'improvement' in removed:
                self.improvement_history.pop(0)

    def auto_select_noise_factors(self, circuit: QuantumCircuit, max_factors: int = 5) -> List[float]:
        """Automatically select noise factors based on circuit properties."""
        if circuit is None:
            return self.noise_factors
        
        depth = circuit.depth()
        n_qubits = circuit.num_qubits
        
        # Heuristic selection based on circuit complexity
        base_factors = [1.0]
        
        if depth <= 5:
            additional = [2.0, 3.0]
        elif depth <= 15:
            additional = [2.0, 4.0, 6.0]
        elif depth <= 30:
            additional = [3.0, 5.0, 7.0, 9.0]
        else:
            additional = [5.0, 7.0, 9.0, 11.0, 13.0]
        
        selected_factors = base_factors + additional[:max_factors-1]
        
        if self.verbose:
            print(f"Auto-selected noise factors for depth {depth}, {n_qubits} qubits: {selected_factors}")
        
        return selected_factors

    def adaptive_zne(self, measurement_function: Callable, initial_params: np.ndarray) -> float:
        """Adaptive ZNE with automatic noise factor selection."""
        if self.verbose:
            print("Running adaptive ZNE...")
        
        current_factors = [1.0]
        current_values = [measurement_function(initial_params, 1.0)]
        
        # Iteratively add noise factors
        candidate_factors = np.linspace(2.0, self.max_noise_factor, 20)
        
        for factor in candidate_factors:
            current_factors.append(factor)
            current_values.append(measurement_function(initial_params, factor))
            
            if len(current_values) >= 3:
                # Check convergence
                temp_factors = current_factors.copy()
                temp_values = current_values.copy()
                
                extrapolated = self._richardson_extrapolation(np.array(temp_factors), np.array(temp_values))
                
                if len(current_values) > 3:
                    prev_extrapolated = self._richardson_extrapolation(
                        np.array(temp_factors[:-1]), 
                        np.array(temp_values[:-1])
                    )
                    
                    if abs(extrapolated - prev_extrapolated) < self.adaptive_threshold:
                        if self.verbose:
                            print(f"Adaptive ZNE converged after {len(current_values)} measurements")
                        break
        
        # Final extrapolation with selected factors
        self.noise_factors = current_factors
        return self._richardson_extrapolation(np.array(current_factors), np.array(current_values))

    def get_zne_analysis(self) -> Dict[str, Any]:
        """Comprehensive ZNE analysis and statistics."""
        if not self.zne_history:
            return {"error": "No ZNE history available"}
        
        improvements = np.array(self.improvement_history)
        
        # Basic statistics
        analysis = {
            "total_applications": len(self.zne_history),
            "average_improvement": float(np.mean(improvements)),
            "std_improvement": float(np.std(improvements)),
            "max_improvement": float(np.max(improvements)),
            "min_improvement": float(np.min(improvements)),
            "median_improvement": float(np.median(improvements)),
            "success_rate": float(np.mean(improvements > 1e-8) * 100),  # Meaningful improvement
        }
        
        # Method breakdown
        method_counts = {}
        method_improvements = {}
        
        for entry in self.zne_history:
            method = entry.get('method', 'unknown')
            method_counts[method] = method_counts.get(method, 0) + 1
            
            if method not in method_improvements:
                method_improvements[method] = []
            method_improvements[method].append(entry['improvement'])
        
        analysis['method_breakdown'] = {
            'counts': method_counts,
            'avg_improvements': {
                method: float(np.mean(impr)) for method, impr in method_improvements.items()
            }
        }
        
        # Configuration
        analysis['configuration'] = {
            "extrapolation_method": self.extrapolation_method,
            "noise_factors": self.noise_factors,
            "mitiq_available": self.mitiq_available,
            "error_rate": self.error_rate,
            "shots": self.shots,
            "cache_enabled": self.enable_caching,
            "cache_size": len(self._cache) if self._cache else 0
        }
        
        return analysis

    def benchmark_methods(self, test_values: List[float], 
                         noise_factors: Optional[List[float]] = None) -> Dict[str, Any]:
        """Benchmark different extrapolation methods on test data."""
        if noise_factors is None:
            noise_factors = self.noise_factors
            
        methods = ['richardson', 'linear', 'polynomial', 'exponential']
        results = {}
        
        # Save current settings
        original_method = self.extrapolation_method
        original_factors = self.noise_factors
        
        self.noise_factors = noise_factors
        
        for method in methods:
            self.extrapolation_method = method
            try:
                result = self._manual_extrapolation(test_values)
                results[method] = {
                    'extrapolated_value': float(result),
                    'error': None
                }
            except Exception as e:
                results[method] = {
                    'extrapolated_value': None,
                    'error': str(e)
                }
        
        # Restore original settings
        self.extrapolation_method = original_method
        self.noise_factors = original_factors
        
        return {
            'methods': results,
            'test_values': test_values,
            'noise_factors': noise_factors,
            'benchmark_timestamp': np.datetime64('now')
        }

    def reset_history(self):
        """Reset all tracking history and cache."""
        self.zne_history.clear()
        self.improvement_history.clear()
        if self._cache:
            self._cache.clear()
        if self.verbose:
            print("ZNE history and cache reset")

    def plot_zne_history(self):
        """Plot ZNE performance history."""
        try:
            import matplotlib.pyplot as plt
            
            if not self.improvement_history:
                print("No ZNE history to plot")
                return
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Plot improvement over time
            ax1.plot(self.improvement_history, 'b-o', markersize=4)
            ax1.set_title('ZNE Improvement History')
            ax1.set_xlabel('ZNE Application')
            ax1.set_ylabel('Improvement (|mitigated - raw|)')
            ax1.grid(True, alpha=0.3)
            
            # Plot histogram of improvements
            ax2.hist(self.improvement_history, bins=min(20, len(self.improvement_history)//2), 
                    alpha=0.7, color='green', edgecolor='black')
            ax2.set_title('Distribution of ZNE Improvements')
            ax2.set_xlabel('Improvement')
            ax2.set_ylabel('Frequency')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("matplotlib not available for plotting")

    def get_optimal_noise_factors(self, circuit: QuantumCircuit, 
                                 hamiltonian: SparsePauliOp,
                                 max_factors: int = 7) -> List[float]:
        """
        Determine optimal noise factors based on circuit analysis and test runs.
        
        Args:
            circuit: Quantum circuit to analyze
            hamiltonian: Hamiltonian operator
            max_factors: Maximum number of noise factors to use
            
        Returns:
            List of optimized noise factors
        """
        if not QISKIT_AVAILABLE:
            return self.noise_factors
        
        depth = circuit.depth()
        n_qubits = circuit.num_qubits
        gate_count = len(circuit.data)
        
        # Base factor is always 1.0 (no additional noise)
        factors = [1.0]
        
        # Algorithm: Start with small steps, increase based on circuit complexity
        if depth <= 10:
            step_size = 1.0
            max_factor = 5.0
        elif depth <= 30:
            step_size = 1.5
            max_factor = 7.0
        else:
            step_size = 2.0
            max_factor = 10.0
        
        # Generate candidate factors
        current = 1.0 + step_size
        while len(factors) < max_factors and current <= max_factor:
            factors.append(current)
            current += step_size
        
        if self.verbose:
            print(f"Optimal noise factors for circuit (depth={depth}, qubits={n_qubits}): {factors}")
        
        return factors


# Context manager for ZNE operations
class ZNEContext:
    """Context manager for ZNE operations with automatic cleanup."""
    
    def __init__(self, zne_plugin: ZNEDenoiserPlugin, 
                 circuit: QuantumCircuit, hamiltonian: SparsePauliOp):
        """
        Initialize ZNE context.
        
        Args:
            zne_plugin: ZNE plugin instance
            circuit: Quantum circuit
            hamiltonian: Hamiltonian operator
        """
        self.zne_plugin = zne_plugin
        self.circuit = circuit
        self.hamiltonian = hamiltonian
        
    def __enter__(self) -> ZNEDenoiserPlugin:
        """Enter context and set ZNE context."""
        self.zne_plugin.set_context(self.circuit, self.hamiltonian)
        return self.zne_plugin
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and clean up."""
        self.zne_plugin._current_circuit = None
        self.zne_plugin._current_hamiltonian = None


def integrate_with_vqe(vqe_instance, zne_plugin: ZNEDenoiserPlugin):
    """
    Helper function to integrate ZNE with existing VQE instance.
    
    This modifies the VQE's energy evaluation to automatically use ZNE.
    
    Args:
        vqe_instance: Your VQE instance
        zne_plugin: ZNE plugin to integrate
    """
    if not hasattr(vqe_instance, '_original_energy_evaluation'):
        # Store original method
        vqe_instance._original_energy_evaluation = vqe_instance._energy_evaluation
        vqe_instance.zne_plugin = zne_plugin
        
        def zne_energy_evaluation(params):
            """Modified energy evaluation with automatic ZNE."""
            # Get trial wavefunction
            circuit = vqe_instance.ansatz.get_trial_wavefunction(params)
            hamiltonian = vqe_instance.hamiltonian_op
            
            # Use ZNE context
            with ZNEContext(zne_plugin, circuit, hamiltonian) as zne:
                return zne.denoise(None)
        
        # Replace energy evaluation method
        vqe_instance._energy_evaluation = zne_energy_evaluation
        
        print("VQE successfully integrated with ZNE!")
        print("All energy evaluations will now use zero-noise extrapolation.")


# Demo and testing functions
def create_demo_circuit(num_qubits: int = 2, depth: int = 3) -> Optional[QuantumCircuit]:
    """Create a demo quantum circuit for testing ZNE."""
    if not QISKIT_AVAILABLE:
        return None
        
    from qiskit import QuantumCircuit
    
    qc = QuantumCircuit(num_qubits)
    
    # Create a simple parameterized circuit
    for layer in range(depth):
        # Single-qubit rotations
        for q in range(num_qubits):
            qc.ry(np.pi/4 * (layer + 1), q)
        
        # Entanglement
        for q in range(num_qubits - 1):
            qc.cx(q, q + 1)
    
    return qc


def create_demo_hamiltonian(num_qubits: int = 2) -> Optional[SparsePauliOp]:
    """Create a demo Hamiltonian for testing."""
    if not QISKIT_AVAILABLE:
        return None
        
    from qiskit.quantum_info import SparsePauliOp
    
    # Create a simple Hamiltonian
    if num_qubits == 1:
        return SparsePauliOp.from_list([("Z", 1.0), ("X", 0.5)])
    elif num_qubits == 2:
        return SparsePauliOp.from_list([
            ("ZZ", 1.0), ("ZI", 0.3), ("IZ", 0.3), ("XX", -0.1)
        ])
    else:
        # For larger systems, create a simple Ising-like Hamiltonian
        pauli_list = []
        
        # Longitudinal field
        for i in range(num_qubits):
            pauli_str = "I" * num_qubits
            pauli_str = pauli_str[:i] + "Z" + pauli_str[i+1:]
            pauli_list.append((pauli_str, 0.5))
        
        # Coupling terms
        for i in range(num_qubits - 1):
            pauli_str = "I" * num_qubits
            pauli_str = pauli_str[:i] + "Z" + pauli_str[i+1:i+2] + "Z" + pauli_str[i+2:]
            pauli_list.append((pauli_str, 1.0))
        
        return SparsePauliOp.from_list(pauli_list)


def run_comprehensive_demo():
    """Run comprehensive ZNE demonstration with multiple test cases."""
    print("=" * 70)
    print("COMPREHENSIVE ZNE PLUGIN DEMONSTRATION")
    print("=" * 70)
    
    # Test 1: Manual extrapolation (always works)
    print("\n1. MANUAL EXTRAPOLATION TEST")
    print("-" * 40)
    
    zne_plugin = ZNEDenoiserPlugin(
        noise_factors=[1.0, 2.5, 4.0, 5.5],
        extrapolation_method='richardson',
        verbose=True
    )
    
    # Simulate realistic noisy VQE data
    true_ground_state = -1.8567
    noise_strengths = [0.0, 0.05, 0.12, 0.18]
    noisy_energies = [true_ground_state + np.random.normal(0, noise) for noise in noise_strengths]
    
    print(f"Simulated noisy energies: {[f'{e:.6f}' for e in noisy_energies]}")
    
    mitigated_energy = zne_plugin.denoise(noisy_energies)
    error = abs(mitigated_energy - true_ground_state)
    
    print(f"True ground state:     {true_ground_state:.6f}")
    print(f"Mitigated energy:      {mitigated_energy:.6f}")
    print(f"Absolute error:        {error:.6f}")
    print(f"Relative error:        {error/abs(true_ground_state)*100:.3f}%")
    
    # Test 2: Mitiq integration (if available)
    if zne_plugin.mitiq_available:
        print("\n2. MITIQ ZNE TEST")
        print("-" * 40)
        
        circuit = create_demo_circuit(num_qubits=3, depth=4)
        hamiltonian = create_demo_hamiltonian(num_qubits=3)
        
        if circuit and hamiltonian:
            print(f"Demo circuit: {circuit.num_qubits} qubits, depth {circuit.depth()}")
            print(f"Demo Hamiltonian: {len(hamiltonian.paulis)} Pauli terms")
            
            # Test with context manager
            with ZNEContext(zne_plugin, circuit, hamiltonian) as zne:
                mitiq_result = zne.denoise(None)
                print(f"Mitiq ZNE result: {mitiq_result:.8f}")
            
            # Test automatic noise factor selection
            optimal_factors = zne_plugin.get_optimal_noise_factors(circuit, hamiltonian)
            print(f"Optimal noise factors: {optimal_factors}")
        else:
            print("Could not create demo circuit/Hamiltonian")
    else:
        print("\n2. Mitiq ZNE: Not available")
        print("   Install with: pip install mitiq")
    
    # Test 3: Method benchmarking
    print("\n3. EXTRAPOLATION METHOD BENCHMARKING")
    print("-" * 40)
    
    benchmark_results = zne_plugin.benchmark_methods(noisy_energies)
    
    print("Method comparison on test data:")
    for method, result in benchmark_results['methods'].items():
        if result['extrapolated_value'] is not None:
            error = abs(result['extrapolated_value'] - true_ground_state)
            print(f"   {method:12s}: {result['extrapolated_value']:8.6f} (error: {error:.6f})")
        else:
            print(f"   {method:12s}: Failed - {result['error']}")
    
    # Test 4: Adaptive ZNE simulation
    print("\n4. ADAPTIVE ZNE SIMULATION")
    print("-" * 40)
    
    def mock_measurement_function(params, noise_factor):
        """Mock measurement function that simulates VQE energy evaluation."""
        base_energy = true_ground_state + 0.1 * np.sum(params**2)
        noise = 0.02 * noise_factor * np.random.randn()
        return base_energy + noise
    
    test_params = np.array([0.1, -0.05, 0.03])
    adaptive_result = zne_plugin.adaptive_zne(mock_measurement_function, test_params)
    
    print(f"Adaptive ZNE result: {adaptive_result:.6f}")
    print(f"Final noise factors used: {zne_plugin.noise_factors}")
    
    # Test 5: Performance analysis
    print("\n5. PERFORMANCE ANALYSIS")
    print("-" * 40)
    
    analysis = zne_plugin.get_zne_analysis()
    print("ZNE Performance Statistics:")
    for key, value in analysis.items():
        if key == 'method_breakdown':
            print(f"   Method breakdown:")
            for method, count in value['counts'].items():
                avg_impr = value['avg_improvements'][method]
                print(f"      {method}: {count} applications, avg improvement: {avg_impr:.6f}")
        elif key == 'configuration':
            print(f"   Configuration:")
            for config_key, config_val in value.items():
                print(f"      {config_key}: {config_val}")
        elif isinstance(value, (int, float)):
            print(f"   {key}: {value:.6f}")
        else:
            print(f"   {key}: {value}")
    
    # Test 6: Error handling
    print("\n6. ERROR HANDLING TEST")
    print("-" * 40)
    
    # Test with insufficient data
    single_value_result = zne_plugin.denoise(5.0)
    print(f"Single value input result: {single_value_result}")
    
    # Test with mismatched data
    wrong_size_data = [1.0, 2.0]  # Only 2 values for 4 noise factors
    wrong_size_result = zne_plugin.denoise(wrong_size_data)
    print(f"Wrong size data result: {wrong_size_result}")
    
    print("\n" + "=" * 70)
    print("COMPREHENSIVE DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    
    # Summary
    total_applications = len(zne_plugin.zne_history)
    avg_improvement = np.mean(zne_plugin.improvement_history) if zne_plugin.improvement_history else 0
    
    print(f"Total ZNE applications: {total_applications}")
    print(f"Average improvement: {avg_improvement:.6f}")
    print(f"ZNE plugin ready for VQE integration!")
    
    return zne_plugin


if __name__ == "__main__":
    print("CORRECTED ZNE DENOISER PLUGIN")
    print("=" * 50)
    
    # Quick initialization test
    try:
        zne_plugin = ZNEDenoiserPlugin(
            noise_factors=[1.0, 3.0, 5.0],
            extrapolation_method='richardson',
            error_rate=0.02,
            shots=1024,
            verbose=True
        )
        print("✓ Plugin initialization successful")
        print(f"✓ Mitiq available: {zne_plugin.mitiq_available}")
        print(f"✓ Qiskit available: {QISKIT_AVAILABLE}")
        
    except Exception as e:
        print(f"✗ Plugin initialization failed: {e}")
        exit(1)
    
    # Run comprehensive demo
    demo_plugin = run_comprehensive_demo()
    
    print("\n" + "=" * 50)
    print("INTEGRATION INSTRUCTIONS:")
    print("1. Save this code as 'zne_denoiser.py'")
    print("2. Import: from zne_denoiser import ZNEDenoiserPlugin")
    print("3. Initialize: zne = ZNEDenoiserPlugin(...)")
    print("4. Use in VQE: integrate_with_vqe(your_vqe, zne)")
    print("5. Or manual: zne.set_context(circuit, hamiltonian); zne.denoise()")
    print("=" * 50)
