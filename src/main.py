import sys 
import os
# Add the parent directory to the PYTHONPATH 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pprint import pformat
from qiskit_ibm_runtime import QiskitRuntimeService
from utils.save_account import save_account, get_available_backend_name


def main():
    """
    With virtual environment active, to run this script in terminal CLI
    enter the following:
    python src/main.py
    """

    min_required_qubits = 100
    channel = "ibm_quantum"
    token  = os.getenv('IBM_QUANTUM_TOKEN')
    save_account(channel, token)
    get_available_backend_name(token, min_required_qubits)
    
if __name__ == "__main__":
    main()
