import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, List, Tuple, Dict, Any, Optional
from hamiltonian import HamiltonianPlugin
from ansatz import *
from vqe import *
from denoiser import *

class ClassicalOptimizerPlugin(ABC):
    """Base class for classical optimizers used in VQE."""
    
    @abstractmethod
    def optimize(self, objective_function: Callable[[np.ndarray], float], initial_params: np.ndarray) -> np.ndarray:
        """
        Optimize the objective function starting from initial parameters.
        
        Args:
            objective_function: Function to minimize
            initial_params: Initial parameter values
            
        Returns:
            Optimized parameters
        """
        pass
    
    def get_optimizer_info(self) -> Dict[str, Any]:
        """Return information about this optimizer."""
        return {"optimizer_type": self.__class__.__name__}


class SPSAOptimizer(ClassicalOptimizerPlugin):
    """
    Simultaneous Perturbation Stochastic Approximation (SPSA) optimizer.
    
    This is a gradient-free optimizer particularly suited for noisy objective functions
    like those encountered in VQE with quantum hardware or simulation noise.
    """
    
    def __init__(self, max_iter: int = 100, a: float = 0.2, c: float = 0.15, 
                 alpha: float = 0.602, gamma: float = 0.101, tol: float = 1e-6,
                 bounds: Optional[List[Tuple[float, float]]] = None,
                 seed: Optional[int] = None, verbose: bool = True):
        """
        Initialize SPSA optimizer.
        
        Args:
            max_iter: Maximum number of iterations
            a: Step size scaling parameter
            c: Perturbation size scaling parameter  
            alpha: Step size decay exponent (should be ~0.602)
            gamma: Perturbation decay exponent (should be ~0.101)
            tol: Convergence tolerance
            bounds: Optional parameter bounds as [(min, max), ...]
            seed: Random seed for reproducibility
            verbose: Whether to print progress
        """
        self.max_iter = max_iter
        self.a = a
        self.c = c
        self.alpha = alpha
        self.gamma = gamma
        self.tol = tol
        self.bounds = bounds
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)
        
        # Optimization history
        self.history = []
        self.best_value = float('inf')
        self.best_params = None
        
    def _ak(self, k: int) -> float:
        """Compute step size at iteration k."""
        return self.a / ((k + 1) ** self.alpha)
    
    def _ck(self, k: int) -> float:
        """Compute perturbation size at iteration k.""" 
        return self.c / ((k + 1) ** self.gamma)
    
    def _project_bounds(self, params: np.ndarray) -> np.ndarray:
        """Project parameters to satisfy bounds constraints."""
        if self.bounds is None:
            return params
        
        result = params.copy()
        for i, (low, high) in enumerate(self.bounds):
            result[i] = np.clip(result[i], low, high)
        return result
    
    def optimize(self, objective_function: Callable[[np.ndarray], float], 
                 initial_params: np.ndarray) -> np.ndarray:
        """Run SPSA optimization."""
        params = np.array(initial_params, dtype=float)
        params = self._project_bounds(params)  # Ensure initial params satisfy bounds
        
        # Initialize tracking
        self.history = []
        prev_val = objective_function(params)
        self.best_value = prev_val
        self.best_params = params.copy()
        self.history.append(prev_val)
        
        if self.verbose:
            print(f"[SPSA] Initial energy: {prev_val:.8f}")
            
        for k in range(self.max_iter):
            ak = self._ak(k)
            ck = self._ck(k)
            
            # Generate simultaneous perturbation
            delta = self.rng.choice([-1, 1], size=params.shape).astype(float)
            
            # Evaluate function at perturbed points
            plus_params = self._project_bounds(params + ck * delta)
            minus_params = self._project_bounds(params - ck * delta)
            
            e_plus = objective_function(plus_params)
            e_minus = objective_function(minus_params)
            
            # SPSA gradient approximation (corrected formula)
            # Avoid division by zero and handle the case where delta = 0 (shouldn't happen with ±1)
            gk = np.where(delta != 0, (e_plus - e_minus) / (2.0 * ck * delta), 0.0)
            
            # Update parameters
            params = self._project_bounds(params - ak * gk)
            
            # Evaluate new parameters
            curr_val = objective_function(params)
            self.history.append(curr_val)
            
            # Track best solution
            if curr_val < self.best_value:
                self.best_value = curr_val
                self.best_params = params.copy()
                
            # Progress reporting
            if self.verbose:
                impr = prev_val - curr_val
                print(f"[SPSA] iter={k+1:3d} energy={curr_val:.8f} ΔE={impr:+.3e} "
                      f"ak={ak:.3e} ck={ck:.3e}")
                      
            # Convergence check
            if abs(prev_val - curr_val) < self.tol:
                if self.verbose:
                    print(f"[SPSA] Converged (|ΔE| < {self.tol}) at iter {k+1}")
                break
                
            prev_val = curr_val
            
        return self.best_params
    
    def get_optimizer_info(self) -> Dict[str, Any]:
        """Return optimizer information and statistics."""
        return {
            "optimizer_type": "SPSA",
            "max_iter": self.max_iter,
            "a": self.a, "c": self.c, "alpha": self.alpha, "gamma": self.gamma,
            "tol": self.tol,
            "bounds": self.bounds is not None,
            "best_value": self.best_value,
            "iterations_run": len(self.history) - 1 if self.history else 0,
            "history_length": len(self.history)
        }


class COBYLAOptimizer(ClassicalOptimizerPlugin):
    """
    Constrained Optimization BY Linear Approximation (COBYLA) optimizer.
    
    A derivative-free optimizer that works well for fine-tuning near optimal points.
    Falls back to coordinate descent if SciPy is not available.
    """
    
    def __init__(self, max_iter: int = 200, tol: float = 1e-6, 
                 rhobeg: float = 0.2, rhoend: Optional[float] = None,
                 bounds: Optional[List[Tuple[float, float]]] = None,
                 disp: bool = True):
        """
        Initialize COBYLA optimizer.
        
        Args:
            max_iter: Maximum number of iterations
            tol: Convergence tolerance
            rhobeg: Initial step size
            rhoend: Final step size (defaults to tol)
            bounds: Optional parameter bounds
            disp: Whether to display progress
        """
        self.max_iter = max_iter
        self.tol = tol
        self.rhobeg = rhobeg
        self.rhoend = rhoend if rhoend is not None else tol
        self.bounds = bounds
        self.disp = disp
        
        # Optimization tracking
        self.history = []
        self.success = False
        self.message = ""
        
    def _setup_constraints(self):
        """Setup constraint functions for bounds if provided."""
        if self.bounds is None:
            return []
            
        constraints = []
        for i, (low, high) in enumerate(self.bounds):
            if low is not None:
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, idx=i, bound=low: x[idx] - bound
                })
            if high is not None:
                constraints.append({
                    'type': 'ineq', 
                    'fun': lambda x, idx=i, bound=high: bound - x[idx]
                })
        return constraints
    
    def _coordinate_descent_fallback(self, objective_function: Callable[[np.ndarray], float],
                                   initial_params: np.ndarray) -> np.ndarray:
        """Simple coordinate descent fallback when SciPy is unavailable."""
        params = np.array(initial_params, dtype=float)
        best_val = objective_function(params)
        step = self.rhobeg
        
        if self.disp:
            print("[COBYLA] Using coordinate descent fallback")
            print(f"[COBYLA] Initial energy: {best_val:.8f}")
        
        for iteration in range(self.max_iter):
            improved = False
            
            # Try improving each parameter individually
            for i in range(len(params)):
                original_val = params[i]
                
                # Try both directions
                for direction in [+1, -1]:
                    trial_params = params.copy()
                    trial_params[i] = original_val + direction * step
                    
                    # Apply bounds if specified
                    if self.bounds and i < len(self.bounds):
                        low, high = self.bounds[i]
                        if low is not None:
                            trial_params[i] = max(trial_params[i], low)
                        if high is not None:
                            trial_params[i] = min(trial_params[i], high)
                    
                    trial_val = objective_function(trial_params)
                    self.history.append(trial_val)
                    
                    if trial_val < best_val - self.tol:
                        best_val = trial_val
                        params = trial_params.copy()
                        improved = True
                        
                        if self.disp:
                            print(f"[COBYLA] iter={iteration:3d} energy={best_val:.8f} "
                                  f"improved param {i}")
                        break  # Found improvement for this parameter
            
            # Reduce step size if no improvement found
            if not improved:
                step *= 0.5
                if step < self.rhoend:
                    if self.disp:
                        print(f"[COBYLA] Converged at iteration {iteration}")
                    break
        
        self.success = True
        self.message = "Coordinate descent completed"
        return params
    
    def optimize(self, objective_function: Callable[[np.ndarray], float],
                 initial_params: np.ndarray) -> np.ndarray:
        """Run COBYLA optimization with graceful fallback."""
        params = np.array(initial_params, dtype=float)
        self.history = []
        
        # Store initial evaluation
        initial_val = objective_function(params)
        self.history.append(initial_val)
        
        try:
            from scipy.optimize import minimize
            
            # Track function evaluations
            def tracking_objective(p):
                val = objective_function(p)
                self.history.append(val)
                return val
            
            # Setup constraints
            constraints = self._setup_constraints()
            
            # Run SciPy COBYLA
            result = minimize(
                tracking_objective,
                params,
                method="COBYLA",
                constraints=constraints,
                options={
                    "maxiter": self.max_iter,
                    "tol": self.tol,
                    "disp": self.disp,
                    "rhobeg": self.rhobeg,
                    "rhoend": self.rhoend
                }
            )
            
            self.success = result.success
            self.message = result.message
            
            if self.disp:
                print(f"[COBYLA] {'Success' if result.success else 'Warning'}: {result.message}")
                print(f"[COBYLA] Final energy: {result.fun:.8f}")
                print(f"[COBYLA] Function evaluations: {len(self.history)}")
                
            return result.x
            
        except ImportError:
            if self.disp:
                print("[COBYLA] SciPy not available, using coordinate descent fallback")
            return self._coordinate_descent_fallback(objective_function, params)
            
        except Exception as e:
            if self.disp:
                print(f"[COBYLA] SciPy optimization failed ({e}), using fallback")
            return self._coordinate_descent_fallback(objective_function, params)
    
    def get_optimizer_info(self) -> Dict[str, Any]:
        """Return optimizer information and statistics."""
        return {
            "optimizer_type": "COBYLA",
            "max_iter": self.max_iter,
            "tol": self.tol,
            "rhobeg": self.rhobeg,
            "rhoend": self.rhoend,
            "bounds": self.bounds is not None,
            "success": self.success,
            "message": self.message,
            "function_evaluations": len(self.history)
        }


class HybridSPSAThenCOBYLA(ClassicalOptimizerPlugin):
    """
    Hybrid optimizer that combines SPSA for global exploration with COBYLA for local refinement.
    
    The optimizer automatically switches from SPSA to COBYLA when convergence metrics
    indicate that fine-tuning would be beneficial.
    """
    
    def __init__(self,
                 # SPSA parameters
                 spsa_max_iter: int = 100,
                 spsa_a: float = 0.2,
                 spsa_c: float = 0.15,
                 spsa_alpha: float = 0.602,
                 spsa_gamma: float = 0.101,
                 # COBYLA parameters  
                 cobyla_max_iter: int = 200,
                 cobyla_tol: float = 1e-8,
                 cobyla_rhobeg: float = 0.1,
                 # Switching parameters
                 switch_tol: float = 1e-3,
                 min_spsa_iters: int = 10,
                 ma_window: int = 5,
                 rel_switch_threshold: float = 1e-4,
                 plateau_iters: int = 5,
                 force_cobyla: bool = True,
                 # General parameters
                 bounds: Optional[List[Tuple[float, float]]] = None,
                 seed: Optional[int] = None,
                 verbose: bool = True):
        """
        Initialize hybrid optimizer.
        
        Args:
            spsa_max_iter: Maximum SPSA iterations
            spsa_a, spsa_c, spsa_alpha, spsa_gamma: SPSA parameters
            cobyla_max_iter: Maximum COBYLA iterations  
            cobyla_tol: COBYLA tolerance
            cobyla_rhobeg: COBYLA initial step size
            switch_tol: Absolute improvement threshold for switching
            min_spsa_iters: Minimum SPSA iterations before considering switch
            ma_window: Moving average window for switch decision
            rel_switch_threshold: Relative improvement threshold
            plateau_iters: Consecutive iterations with small improvement to detect plateau
            force_cobyla: Whether to always run COBYLA phase
            bounds: Parameter bounds
            seed: Random seed
            verbose: Whether to print progress
        """
        # Store all parameters
        self.switch_tol = switch_tol
        self.min_spsa_iters = min_spsa_iters
        self.ma_window = ma_window
        self.rel_switch_threshold = rel_switch_threshold
        self.plateau_iters = plateau_iters
        self.force_cobyla = force_cobyla
        self.verbose = verbose
        
        # Create optimizers
        self.spsa = SPSAOptimizer(
            max_iter=spsa_max_iter,
            a=spsa_a, c=spsa_c, alpha=spsa_alpha, gamma=spsa_gamma,
            tol=switch_tol / 10,  # Tighter tolerance for SPSA
            bounds=bounds,
            seed=seed,
            verbose=verbose
        )
        
        self.cobyla = COBYLAOptimizer(
            max_iter=cobyla_max_iter,
            tol=cobyla_tol,
            rhobeg=cobyla_rhobeg,
            bounds=bounds,
            disp=verbose
        )
        
        # Tracking
        self.switch_info = {}
        self.total_history = []
        
    def _compute_switch_metrics(self, energy_history: List[float]) -> Dict[str, Any]:
        """Compute metrics to decide whether to switch to COBYLA."""
        if len(energy_history) < 2:
            return {
                'should_switch': False,
                'reason': 'insufficient_data',
                'recent_improvement': float('inf'),
                'ma_improvement': float('inf'),
                'rel_ma_improvement': float('inf'),
                'plateau_detected': False
            }
        
        # Compute improvements
        improvements = np.abs(np.diff(energy_history))
        recent_improvement = improvements[-1] if len(improvements) > 0 else float('inf')
        
        # Moving average improvement
        window_size = min(self.ma_window, len(improvements))
        ma_improvement = np.mean(improvements[-window_size:]) if window_size > 0 else float('inf')
        
        # Relative improvement (avoid division by zero)
        current_energy = energy_history[-1]
        rel_ma_improvement = ma_improvement / max(abs(current_energy), 1e-12)
        
        # Plateau detection
        plateau_window = min(self.plateau_iters, len(improvements))
        plateau_detected = (plateau_window > 0 and 
                          all(imp < self.switch_tol for imp in improvements[-plateau_window:]))
        
        # Decision logic
        ma_stable = ma_improvement < self.switch_tol
        rel_stable = rel_ma_improvement < self.rel_switch_threshold
        
        should_switch = (ma_stable and rel_stable) or plateau_detected
        
        reason = 'none'
        if plateau_detected:
            reason = 'plateau'
        elif ma_stable and rel_stable:
            reason = 'stable_convergence'
            
        return {
            'should_switch': should_switch,
            'reason': reason,
            'recent_improvement': recent_improvement,
            'ma_improvement': ma_improvement,
            'rel_ma_improvement': rel_ma_improvement,
            'plateau_detected': plateau_detected,
            'ma_stable': ma_stable,
            'rel_stable': rel_stable
        }
    
    def optimize(self, objective_function: Callable[[np.ndarray], float],
                 initial_params: np.ndarray) -> np.ndarray:
        """Run hybrid optimization: SPSA followed by COBYLA."""
        if self.verbose:
            print("=" * 70)
            print("HYBRID OPTIMIZATION: SPSA → COBYLA")
            print("=" * 70)
            print(f"SPSA phase: max_iter={self.spsa.max_iter}")
            print(f"Switch criteria: tol={self.switch_tol}, ma_window={self.ma_window}")
            print(f"COBYLA phase: max_iter={self.cobyla.max_iter}")
        
        self.total_history = []
        
        # Phase 1: SPSA with early switching logic
        if self.verbose:
            print("\n" + "=" * 40)
            print("Phase 1: SPSA Exploration")
            print("=" * 40)
        
        params = np.array(initial_params, dtype=float)
        switched_early = False
        spsa_iters_completed = 0
        
        # Run SPSA with manual iteration control for switch detection
        self.spsa.history = []
        prev_val = objective_function(params)
        self.spsa.best_value = prev_val
        self.spsa.best_params = params.copy()
        self.total_history.append(prev_val)
        
        if self.verbose:
            print(f"[SPSA] Initial energy: {prev_val:.8f}")
        
        for k in range(self.spsa.max_iter):
            # SPSA iteration
            ak = self.spsa._ak(k)
            ck = self.spsa._ck(k)
            
            delta = self.spsa.rng.choice([-1, 1], size=params.shape).astype(float)
            plus_params = self.spsa._project_bounds(params + ck * delta)
            minus_params = self.spsa._project_bounds(params - ck * delta)
            
            e_plus = objective_function(plus_params)
            e_minus = objective_function(minus_params)
            self.total_history.extend([e_plus, e_minus])
            
            gk = np.where(delta != 0, (e_plus - e_minus) / (2.0 * ck * delta), 0.0)
            params = self.spsa._project_bounds(params - ak * gk)
            
            curr_val = objective_function(params)
            self.total_history.append(curr_val)
            
            if curr_val < self.spsa.best_value:
                self.spsa.best_value = curr_val
                self.spsa.best_params = params.copy()
            
            spsa_iters_completed = k + 1
            
            # Progress reporting
            if self.verbose:
                impr = prev_val - curr_val
                print(f"[SPSA] iter={k+1:3d} energy={curr_val:.8f} ΔE={impr:+.3e}")
            
            # Check for early switch (after minimum iterations)
            if k >= self.min_spsa_iters - 1:  # k is 0-indexed
                # Use only energy evaluations from main parameter updates for switch decision
                main_evaluations = [self.total_history[i] for i in range(0, len(self.total_history), 3)]
                metrics = self._compute_switch_metrics(main_evaluations)
                
                if metrics['should_switch'] and not self.force_cobyla:
                    switched_early = True
                    self.switch_info = {
                        'switched_early': True,
                        'switch_iteration': k + 1,
                        'reason': metrics['reason'],
                        'metrics': metrics
                    }
                    if self.verbose:
                        print(f"[Hybrid] Early switch triggered at iteration {k+1}: {metrics['reason']}")
                        print(f"         MA improvement: {metrics['ma_improvement']:.3e}")
                        print(f"         Rel improvement: {metrics['rel_ma_improvement']:.3e}")
                        print(f"         Plateau: {metrics['plateau_detected']}")
                    break
            
            # SPSA convergence check
            if abs(prev_val - curr_val) < self.spsa.tol:
                if self.verbose:
                    print(f"[SPSA] Converged at iteration {k+1}")
                break
                
            prev_val = curr_val
        
        # Finalize SPSA results
        params = self.spsa.best_params.copy()
        
        # Phase 2: Decide on COBYLA
        run_cobyla = self.force_cobyla or switched_early
        
        if not switched_early and len(self.total_history) >= 6:  # Need at least a few evaluations
            main_evaluations = [self.total_history[i] for i in range(0, len(self.total_history), 3)]
            metrics = self._compute_switch_metrics(main_evaluations)
            if metrics['should_switch']:
                run_cobyla = True
                self.switch_info = {
                    'switched_early': False,
                    'switch_iteration': spsa_iters_completed,
                    'reason': metrics['reason'],
                    'metrics': metrics
                }
        
        if not run_cobyla:
            self.switch_info = {
                'switched_early': False,
                'switch_iteration': None,
                'reason': 'no_switch_needed',
                'metrics': {}
            }
            if self.verbose:
                print("\n[Hybrid] No COBYLA refinement needed - SPSA result sufficient")
            return params
        
        # Phase 2: COBYLA refinement
        if self.verbose:
            print("\n" + "=" * 40)
            print("Phase 2: COBYLA Refinement") 
            print("=" * 40)
            print(f"Starting COBYLA from energy: {self.spsa.best_value:.8f}")
        
        # Track COBYLA evaluations separately
        cobyla_start_evals = len(self.total_history)
        
        def tracking_objective(p):
            val = objective_function(p)
            self.total_history.append(val)
            return val
        
        final_params = self.cobyla.optimize(tracking_objective, params)
        
        cobyla_evals = len(self.total_history) - cobyla_start_evals
        
        if self.verbose:
            final_energy = self.total_history[-1] if self.total_history else float('inf')
            improvement = self.spsa.best_value - final_energy
            print(f"\n" + "=" * 70)
            print("HYBRID OPTIMIZATION COMPLETE")
            print("=" * 70)
            print(f"SPSA iterations: {spsa_iters_completed}")
            print(f"COBYLA evaluations: {cobyla_evals}")
            print(f"Total function evaluations: {len(self.total_history)}")
            print(f"Initial energy: {self.total_history[0]:.8f}")
            print(f"SPSA final: {self.spsa.best_value:.8f}")
            print(f"Final energy: {final_energy:.8f}")
            print(f"Total improvement: {self.total_history[0] - final_energy:.3e}")
            print(f"COBYLA improvement: {improvement:+.3e}")
            print(f"Switch reason: {self.switch_info.get('reason', 'N/A')}")
        
        return final_params
    
    def get_optimizer_info(self) -> Dict[str, Any]:
        """Return comprehensive optimizer information."""
        return {
            "optimizer_type": "Hybrid_SPSA_COBYLA",
            "spsa_info": self.spsa.get_optimizer_info(),
            "cobyla_info": self.cobyla.get_optimizer_info(),
            "switch_info": self.switch_info,
            "total_evaluations": len(self.total_history),
            "switch_tol": self.switch_tol,
            "ma_window": self.ma_window,
            "min_spsa_iters": self.min_spsa_iters,
            "force_cobyla": self.force_cobyla
        }


# Testing and demonstration
if __name__ == "__main__":
    print("Testing VQE Optimizers")
    print("=" * 50)
    
    # Test objective function (noisy quadratic)
    def test_objective(params):
        # Simulate VQE-like objective with noise
        base_energy = np.sum(params**2) + 2.0 * np.sum(params)
        noise = 0.001 * np.random.randn()  # Small noise
        return base_energy + noise
    
    # Test parameters
    initial_params = np.array([0.5, -0.3, 0.2, 0.1])
    bounds = [(-1.0, 1.0) for _ in initial_params]
    
    # Test 1: SPSA only
    print("\n1. Testing SPSA Optimizer")
    print("-" * 30)
    spsa = SPSAOptimizer(max_iter=50, bounds=bounds, verbose=True, seed=42)
    result_spsa = spsa.optimize(test_objective, initial_params)
    print(f"SPSA Result: {result_spsa}")
    print(f"Final energy: {test_objective(result_spsa):.6f}")
    print(f"Info: {spsa.get_optimizer_info()}")
    
    # Test 2: COBYLA only  
    print("\n2. Testing COBYLA Optimizer")
    print("-" * 30)
    cobyla = COBYLAOptimizer(max_iter=100, bounds=bounds, disp=True)
    result_cobyla = cobyla.optimize(test_objective, initial_params)
    print(f"COBYLA Result: {result_cobyla}")
    print(f"Final energy: {test_objective(result_cobyla):.6f}")
    print(f"Info: {cobyla.get_optimizer_info()}")
    
    # Test 3: Hybrid optimizer
    print("\n3. Testing Hybrid Optimizer")
    print("-" * 30)
    hybrid = HybridSPSAThenCOBYLA(
        spsa_max_iter=30,
        cobyla_max_iter=50,
        bounds=bounds,
        force_cobyla=True,
        verbose=True,
        seed=42
    )
    result_hybrid = hybrid.optimize(test_objective, initial_params)
    print(f"Hybrid Result: {result_hybrid}")
    print(f"Final energy: {test_objective(result_hybrid):.6f}")
    print(f"Info: {hybrid.get_optimizer_info()}")
    
    print("\n" + "=" * 50)
    print("All optimizer tests completed!")
