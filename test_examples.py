import sys 
import os
from pprint import pformat
from qiskit_ibm_runtime import QiskitRuntimeService
from utils.save_account import save_account

# Add the parent directory to the PYTHONPATH 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def main():
    """
    With virtual environment active, to run this script in terminal CLI
    enter the following:
    python tests/test_examples.py
    """

    host_name = "ibm_brisbane"
    channel = "ibm_quantum"
    token  = os.getenv('IBM_QUANTUM_TOKEN')
    save_account(channel, token)
    service = QiskitRuntimeService(channel=channel, token=token)
    backend = service.backend(name=host_name)
    print(f"{backend.num_qubits} qubits in {host_name} quantum computer") # to get number of qubits of selected Quantum computer ibm_brisbane
    
if __name__ == "__main__":
    main()
