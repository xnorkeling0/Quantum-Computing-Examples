import os
from qiskit_ibm_runtime import QiskitRuntimeService

class QuantumService:
    """
    It defines a quantum service.
    Usage:
    service = QuantumService()

    Application example:
    backend = service.backend(name = "ibm_brisbane")
    backend.num_qubits # to get number of qubits of selected Quantum computer ibm_brisbane
    """
    def __init__(self):
        self.token = os.getenv('IBM_QUANTUM_TOKEN')
        self.check_and_save_token()

    def check_and_save_token(self):
        saved_accounts = QiskitRuntimeService.saved_accounts()
        if not any(account['token'] == self.token for account in saved_accounts):
            QiskitRuntimeService.save_account(channel="ibm_quantum", token=self.token)
            print("Token has been saved.")
        else:
            print("Token is already saved.")

# TODO add logger
