from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.circuit.library import UCCSD
from qiskit_nature.second_q.circuit.library import HartreeFock
from qiskit_nature.second_q.drivers import PySCFDriver

# 1. Define geometry
geometry = [
    ["N", [0.0000, 0.0000, 0.0000]],
    ["H", [0.9377, 0.0000, -0.3816]],
    ["H", [-0.4688, 0.8119, -0.3816]],
    ["H", [-0.4688, -0.8119, -0.3816]]
]

# 2. Set up driver and problem
driver = PySCFDriver(atom=geometry, basis="sto3g")
es_problem = ElectronicStructureProblem(driver)
second_q_op = es_problem.second_q_ops()

# 3. Get number of orbitals and particles
num_spin_orbitals = es_problem.num_spin_orbitals
num_particles = es_problem.num_particles

# 4. Set up qubit mapper
mapper = JordanWignerMapper()

# 5. Prepare Hartree-Fock reference
hf_init_state = HartreeFock(num_spin_orbitals, num_particles, mapper)

# 6. Build UCCSD ansatz
uccsd_ansatz = UCCSD(
    num_spin_orbitals=num_spin_orbitals,
    num_particles=num_particles,
    qubit_mapper=mapper,
    initial_state=hf_init_state
)

# Now uccsd_ansatz is your ansatz circuit
print(uccsd_ansatz)