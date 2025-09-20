from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.units import DistanceUnit


class HamiltonianPlugin:
    def __init__(self, auto_active=True, active_electrons=4, active_orbitals=3):
        # Default geometry = NH3
        self.geom = [
            "N 0.000000 0.000000 0.000000",
            "H 0.000000 0.937700 -0.381600",
            "H 0.812100 -0.468800 -0.381600",
            "H -0.812100 -0.468800 -0.381600"
        ]
        self._hamiltonian = None
        self._problem_active = None
        self._mapper = None
        self.is_fallback = False
        self._hamiltonian_system = None  # Cache the dictionary

        # settings
        self.auto_active = auto_active
        self.active_electrons = active_electrons
        self.active_orbitals = active_orbitals
        self.reduction_applied = False

    def _normalize_geometry(self, geom):
        if isinstance(geom, str):
            return geom
        if isinstance(geom, (list, tuple)) and all(isinstance(x, str) for x in geom):
            return "\n".join(geom)
        raise ValueError("Geometry must be str or list of str.")

    def _compute_hamiltonian(self):
        """Internal method to compute the hamiltonian operator"""
        if self._hamiltonian is not None:
            return

        try:
            atom_spec = self._normalize_geometry(self.geom)
            driver = PySCFDriver(
                atom=atom_spec,
                basis='sto3g',
                charge=0,
                spin=0,
                unit=DistanceUnit.ANGSTROM
            )
            problem = driver.run()

            # auto-detect
            num_electrons = problem.num_particles[0] + problem.num_particles[1]
            num_orbitals = problem.num_spatial_orbitals

            if self.auto_active:
                # Fixed: Use AND instead of OR for proper active space reduction
                if (self.active_electrons < num_electrons) and (self.active_orbitals < num_orbitals):
                    transformer = ActiveSpaceTransformer(
                        num_electrons=self.active_electrons,
                        num_spatial_orbitals=self.active_orbitals
                    )
                    problem = transformer.transform(problem)
                    self.reduction_applied = True
                else:
                    self.reduction_applied = False

            self._problem_active = problem
            self._mapper = JordanWignerMapper()
            h2_op, _ = problem.second_q_ops()
            self._hamiltonian = self._mapper.map(h2_op)
            self.is_fallback = False

        except Exception as e:
            print(f"[Warning] Build failed: {e}")
            paulis = ['IIIIII', 'ZIIIZZ', 'ZZIIZZ', 'IZZIIZ', 'IIZZZZ', 'XXYYZZ', 'YYXXZZ']
            coeffs = [-5.0, 0.12, -0.08, 0.05, -0.03, 0.01, 0.01]
            self._hamiltonian = SparsePauliOp(paulis, coeffs)
            self.is_fallback = True
            # Set fallback values for compatibility
            self._problem_active = None
            self._mapper = None

    def get_hamiltonian(self):
        """
        Returns the hamiltonian system dictionary required by AnsatzPlugin
        """
        # Compute hamiltonian if not already done
        if self._hamiltonian is None:
            self._compute_hamiltonian()

        # Build and cache the system dictionary
        if self._hamiltonian_system is None:
            geometry_str = "; ".join(self.geom) if isinstance(self.geom, list) else str(self.geom)
            
            self._hamiltonian_system = {
                # Required keys for AnsatzPlugin (UCCSD)
                'problem_active': self._problem_active,
                'mapper': self._mapper,
                'hamiltonian_active': self._hamiltonian,
                
                # Required keys for GenericAnsatzPlugin
                'num_qubits': self._hamiltonian.num_qubits,
                
                # Optional metadata
                'basis': 'sto3g',
                'geometry': geometry_str,
                'active_electrons': self.active_electrons if self.reduction_applied else None,
                'active_orbitals': self.active_orbitals if self.reduction_applied else None,
                'reduction_applied': self.reduction_applied,
                'is_fallback': self.is_fallback,
                'mapper_type': self._mapper.__class__.__name__ if self._mapper else None
            }
        
        return self._hamiltonian_system

    def get_hamiltonian_operator(self):
        """
        Returns just the hamiltonian operator (for backward compatibility)
        """
        if self._hamiltonian is None:
            self._compute_hamiltonian()
        return self._hamiltonian

    def print_hamiltonian(self):
        """Print the Hamiltonian in a formatted way"""
        if self._hamiltonian is None:
            self._compute_hamiltonian()
        
        print("--- Hamiltonian Terms ---")
        for pauli, coeff in zip(self._hamiltonian.paulis, self._hamiltonian.coeffs):
            sign = "+" if coeff.real >= 0 else ""
            print(f"{sign}{coeff.real:.6f} * {pauli}")
        print("-" * 26)

    def info(self):
        return {
            'fallback': self.is_fallback,
            'num_qubits': self._hamiltonian.num_qubits if self._hamiltonian else None,
            'active_problem': self._problem_active is not None,
            'mapper': self._mapper.__class__.__name__ if self._mapper else None,
            'geometry_lines': len(self.geom),
            'active_space_used': self.reduction_applied
        }

    def reset(self):
        """Reset the plugin to allow new calculations"""
        self._hamiltonian = None
        self._problem_active = None
        self._mapper = None
        self._hamiltonian_system = None
        self.is_fallback = False
        self.reduction_applied = False


# Test the updated code
plugin = HamiltonianPlugin(auto_active=True, active_electrons=4, active_orbitals=3)



# Get the hamiltonian system dictionary (compatible with ansatz)
hamiltonian_system = plugin.get_hamiltonian()

print("\nPlugin Info:")
print(plugin.info())

print(f"\nHamiltonian System Keys: {list(hamiltonian_system.keys())}")
print(f"Number of qubits: {hamiltonian_system['num_qubits']}")
print(f"Geometry: {hamiltonian_system['geometry']}")
print(f"Is fallback: {hamiltonian_system['is_fallback']}")
print(f"VQE ready: {hamiltonian_system['problem_active'] is not None}")

print()
plugin.print_hamiltonian()

print("\n--- Ready for Ansatz Integration ---")
print("Usage: ansatz.build_from_hamiltonian(hamiltonian_system)")