import sys 
import os
# Add the parent directory to the PYTHONPATH 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.save_account import save_account, get_first_available_backend
import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler


def main():
    token  = os.getenv('IBM_QUANTUM_TOKEN')
    save_account(token)
    backend = get_first_available_backend(token)

    # 1. A quantum circuit for preparing the quantum state |000> + i |111>
    qc_example = QuantumCircuit(3)
    qc_example.h(0)          # generate superpostion
    qc_example.p(np.pi/2,0)  # add quantum phase
    qc_example.cx(0,1)       # 0th-qubit-Controlled-NOT gate on 1st qubit
    qc_example.cx(0,2)       # 0th-qubit-Controlled-NOT gate on 2nd qubit

    # 2. Add the classical output in the form of measurement of all qubits
    qc_measured = qc_example.measure_all(inplace=False)

    # 3. Execute using the Sampler primitive
    sampler = StatevectorSampler()
    job = sampler.run([qc_measured], shots=1000)
    result = job.result()
    print(f" > Counts: {result[0].data["meas"].get_counts()}")


if __name__ == "__main__":
    main()
