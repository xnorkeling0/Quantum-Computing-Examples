import os
from qiskit_ibm_runtime import QiskitRuntimeService

token = os.getenv('IBM_QUANTUM_TOKEN')
service = QiskitRuntimeService(channel="ibm_quantum", token=token)
QiskitRuntimeService.save_account(channel='ibm_quantum', token=token)
backend = service.backend(name = "ibm_brisbane") # TODO: set name in local configuration
backend.num_qubits # to get number of qubits of selected Quantum computer ibm_brisbane
# TODO add logger
