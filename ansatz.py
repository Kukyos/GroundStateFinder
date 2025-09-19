from qiskit_nature.second_q.hamiltonians import ElectronicStructureHamiltonian
from qiskit_nature.second_q.drivers import PySCFDriver

def build_hamiltonian(molecule="H2", basis="sto3g"):
    """Build electronic Hamiltonian for a given molecule."""
    driver = PySCFDriver(atom=molecule, basis=basis)
    problem = driver.run()
    hamiltonian = ElectronicStructureHamiltonian(problem)
    return hamiltonian
