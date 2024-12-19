import sys 
import os
# Add the parent directory to the PYTHONPATH 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pprint import pformat
from utils.save_account import save_account, get_first_available_backend


def main():
    """
    Note:
    by default min_required_qubits = 100

    Usage:
    With virtual environment active, to run this script in terminal CLI
    enter the following:
    python src/main.py
    """

    token  = os.getenv('IBM_QUANTUM_TOKEN')
    save_account(token)
    backend = get_first_available_backend(token)
    print(backend)
    
if __name__ == "__main__":
    main()
