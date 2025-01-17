import sys 
import os
# Add the parent directory to the PYTHONPATH 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.save_account import save_account, get_first_available_backend
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.primitives import StatevectorSampler, StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorOptions, EstimatorV2 as Estimator
from qiskit_ibm_runtime.options.resilience_options import ResilienceOptionsV2


def main():
    """
    Example from https://github.com/Qiskit/qiskit
    """
    

    # 1. Map the problem: A quantum circuit for preparing the quantum state |000> + i |111>
    num_qubits = 3
    qc_example = QuantumCircuit(num_qubits)
    qc_example.h(0)          # generate superpostion
    qc_example.p(np.pi/2,0)  # add quantum phase
    qc_example.cx(0,1)       # 0th-qubit-Controlled-NOT gate on 1st qubit
    qc_example.cx(0,2)       # 0th-qubit-Controlled-NOT gate on 2nd qubit

    # 2. Define the observable to be measured 
    operator = SparsePauliOp.from_list([("XXY", 1), ("XYX", 1), ("YXX", 1), ("YYY", -1)])
    print(f"The operators strings:\n{operator}")

    # 3. Optimize the problem
    token  = os.getenv('IBM_QUANTUM_TOKEN') # getting the custom env variable that stores my IBM token
    service = QiskitRuntimeService(channel="ibm_quantum", token=token)
    backend = service.least_busy(operational=True, simulator=False)
    pass_manager = generate_preset_pass_manager(backend=backend, optimization_level=1)
    qc_transpiled = pass_manager.run(qc_example)
    operator_transpiled = operator.apply_layout(layout=qc_transpiled.layout)

    # 4. Execute on the Backend
    estimator = Estimator(mode=backend)
    job  = estimator.run([(qc_transpiled, operator_transpiled)])
    job_id = job.job_id()
    job_status = job.status()
    print(job_id)
    print(job_status)
    pub_result = job.result()[0]
    print(f"Expectation values: {pub_result.data.evs}")









    # TODO: run on quantum computer following https://github.com/Qiskit/qiskit-ibm-runtime
if __name__ == "__main__":
    main()
