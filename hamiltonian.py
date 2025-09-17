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

    def get_hamiltonian(self):
        if self._hamiltonian is not None:
            return self._hamiltonian

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

        return self._hamiltonian

    def print_hamiltonian(self):
        """Print the Hamiltonian in a formatted way"""
        if self._hamiltonian is None:
            print("Hamiltonian not computed yet. Call get_hamiltonian() first.")
            return
        
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

# Test the corrected code
plugin = HamiltonianPlugin(auto_active=True, active_electrons=4, active_orbitals=3)

# Get geometry input from user
print("Enter molecular geometry (one atom per line, format: 'Element x y z')")
print("Type 'done' when finished:")
geometry = []
while True:
    line = input().strip()
    if line.lower() == 'done':
        break
    geometry.append(line)

plugin.geom = geometry
H = plugin.get_hamiltonian()
print("\nPlugin Info:")
print(plugin.info())
print()
plugin.print_hamiltonian()


