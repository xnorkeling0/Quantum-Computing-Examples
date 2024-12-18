from examples.set_backend import QuantumService

# run_script.py


def main():
    """
    With virtual environment active, to run this script in terminal CLI
    enter the following:
    python test_examples.py

    """
    service = QuantumService()
    backend = service.backend(name = "ibm_brisbane")
    print(backend.num_qubits) # to get number of qubits of selected Quantum computer ibm_brisbane
    
if __name__ == "__main__":
    main()
