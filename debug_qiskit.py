import traceback
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.problems import ElectronicStructureProblem

try:
    geom = (
        "N  0.0000  0.0000  0.0000;"
        " H  0.9377  0.0000 -0.3816;"
        " H -0.4688  0.8119 -0.3816;"
        " H -0.4688 -0.8119 -0.3816"
    )
    driver = PySCFDriver(atom=geom, basis="sto3g", charge=0, spin=0, unit=DistanceUnit.ANGSTROM)
    print("driver created:", type(driver))
    problem = ElectronicStructureProblem(driver)
    print("problem created:", type(problem))
    sops = problem.second_q_ops()
    print("second_q_ops keys:", list(sops.keys()))
except Exception:
    traceback.print_exc()
