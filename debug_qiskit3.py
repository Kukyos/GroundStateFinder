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
    result = driver.run()
    # result might already be an ElectronicStructureProblem in this build
    if isinstance(result, ElectronicStructureProblem):
        problem = result
    else:
        problem = ElectronicStructureProblem(result)
    sops = problem.second_q_ops()
    print("second_q_ops type:", type(sops))
    try:
        print("repr(second_q_ops):", repr(sops))
    except Exception as e:
        print("repr failure:", e)
    # If iterable, show element types
    try:
        for i, el in enumerate(sops):
            print(f"element[{i}] type=", type(el), "repr=", repr(el)[:200])
    except Exception as e:
        print("iter failure:", type(e), e)
    # If mapping, show keys and types
    try:
        for k in getattr(sops, 'keys', lambda: [])():
            v = sops[k]
            print(f"key {k!r} -> type=", type(v), "repr=", repr(v)[:200])
    except Exception as e:
        print("mapping access failure:", type(e), e)
except Exception:
    traceback.print_exc()
