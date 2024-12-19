from qiskit_ibm_runtime import QiskitRuntimeService

def save_account(channel, token):
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

def get_available_backend_name(token:str , min_required_qubits: int, channel: str="ibm_quantum") -> str:
    """
    Returns the name of the first available quantum computing machine
    compatible with the user requirements including its IBM quantum account.

    :param token: the user's IBM QUantum account token
    :param min_required_qubits: the minimum number of qubits needed by the user to run a specific application
    :param channel: the communication channel to use for connecting with IBM Quantum services.
    the default value "ibm_quantum" specifies cloud services.
    :return backend.name: name of the first available quantum computing machine
    """

    service = QiskitRuntimeService(channel=channel, token=token)
    backends = service.backends()
    for backend in backends: 
        if backend.status().operational and backend.num_qubits>=min_required_qubits:
            print({backend.name: f"{backend.num_qubits} qubits"})
            break
    return backend.name