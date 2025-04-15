import os
from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, ibm_backend
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

def save_account(token, channel: str="ibm_quantum"):
    """
    Ensures saved_accounts is a list of dictionaries
    """
    
    saved_accounts = QiskitRuntimeService.saved_accounts()
    for _, account in saved_accounts.items():
        if not account["token"] == token:
            QiskitRuntimeService.save_account(channel=channel, token=token)
            print("token has been saved")
        else:
            print("token is already saved")

def get_first_available_backend(token:str, channel: str="ibm_quantum") -> ibm_backend.IBMBackend:
    """
    Returns the first available quantum computing machine backend handler
    compatible with the user requirements including its IBM quantum account.

    :param token: the user's IBM QUantum account token
    :param channel: the communication channel to use for connecting with IBM Quantum services.
    the default value "ibm_quantum" specifies cloud services.
    :return backend: name of the first available quantum computing machine
    """

    service = QiskitRuntimeService(channel=channel, token=token)
    backends = service.backends(simulator=False, operational=True, min_num_qubits=127)
    for backend in backends:
        if backend:
            print({backend.name: f"available with {backend.num_qubits} qubits"})
            break
    return backend

def transpile_circuit(
        circuit: QuantumCircuit,
        channel: str = "ibm_quantum",
        operational: bool = True,
        simulator: bool = False,
        optimization_level: int = 1
        )->QuantumCircuit:
    """
    Performs all the necessary steps to generate a transpiled circuit to run on a real IBM quantum computer
    including:
        -   getting user account token
        -   instantiating the service, backend and pass manager
    """
    token  = os.getenv('IBM_QUANTUM_TOKEN') # getting the custom env variable that stores my IBM token
    service = QiskitRuntimeService(channel=channel, token=token)
    backend = service.least_busy(operational=operational, simulator=simulator)
    pass_manager = generate_preset_pass_manager(backend=backend, optimization_level=optimization_level)
    qc_transpiled = pass_manager.run(circuit)
    return backend, qc_transpiled