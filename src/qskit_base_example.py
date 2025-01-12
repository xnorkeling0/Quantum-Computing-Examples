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


def main():
    """
    Example from https://github.com/Qiskit/qiskit
    Here below steps for simulation only (not on Quantum Computer) are specified with (Simulation)
    while the ones with no specifications are meant for execution on Quantum Computer.
    """
    token  = os.getenv('IBM_QUANTUM_TOKEN') # getting the custom env variable that stores my IBM token
    save_account(token)
    backend = get_first_available_backend(token)

    # 1. Map the problem: A quantum circuit for preparing the quantum state |000> + i |111>
    qc_example = QuantumCircuit(3)
    qc_example.h(0)          # generate superpostion
    qc_example.p(np.pi/2,0)  # add quantum phase
    qc_example.cx(0,1)       # 0th-qubit-Controlled-NOT gate on 1st qubit
    qc_example.cx(0,2)       # 0th-qubit-Controlled-NOT gate on 2nd qubit

    # 2. (Simulation) Add the classical output in the form of measurement of all qubits
    qc_measured = qc_example.measure_all(inplace=False)

    # 3. (Simulation) Execute using the Sampler primitive
    sampler = StatevectorSampler()
    job = sampler.run([qc_measured], shots=1000)
    result = job.result()
    print(f" > Counts: {result[0].data["meas"].get_counts()}")

    # 2. Define the observable to be measured 
    operator = SparsePauliOp.from_list([("XXY", 1), ("XYX", 1), ("YXX", 1), ("YYY", -1)])
    print(f"The operators strings:\n{operator}")
    
    # 3. (Simulation) Execute using the Estimator primitive
    estimator = StatevectorEstimator()
    job = estimator.run([(qc_example, operator)], precision=1e-3)
    result = job.result()
    print(f" > Expectation values: {result[0].data.evs}")

    # 2. Optimize the problem
    # qc_transpiled = transpile(qc_example, basis_gates = ['cz', 'sx', 'rz'], coupling_map =[[0, 1], [1, 2]] , optimization_level=3)
    pass_manager = generate_preset_pass_manager(backend=backend, optimization_level=1)
    qc_transpiled = pass_manager.run(qc_example)
    operator_transpiled = operator.apply_layout(qc_transpiled.layout)

    # 3. Execute on the Backend
    # initialize an estimator that takes in the backend with some options
    #
    options = EstimatorOptions()
    options.resilience_level = 1 # we use measurements without mitigation (e.g., level 2 is to get zero noise extrapolation)
    options.optimization_level = 0 # because transpilation is already done on the local machine
    options.dynamical_decoupling.enable = True # to get rid of interferences like cross-talks
    options.dynamical_decoupling.sequence_type = "XY4" # error suppression technique

    """
    Dynamical Decoupling is a method used to reduce errors in quantum computations by applying a sequence 
    of pulses to qubits that are idle (not currently being used in operations). The goal is to cancel out
    the effects of unwanted interactions and noise that can cause errors.
    The "XY4" sequence is one type of pulse sequence used in dynamical decoupling. It consists of a series
    of X and Y gates applied in a specific order to help mitigate errors. The XY4 sequence is known for its
    simplicity and effectiveness in reducing certain types of noise.
    """







    # TODO: run on quantum computer following https://github.com/Qiskit/qiskit-ibm-runtime
if __name__ == "__main__":
    main()
