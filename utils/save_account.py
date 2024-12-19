from qiskit_ibm_runtime import QiskitRuntimeService, ibm_backend

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

def get_first_available_backend(token:str , min_required_qubits: int=100, channel: str="ibm_quantum") -> ibm_backend.IBMBackend:
    """
    Returns the first available quantum computing machine backend handler
    compatible with the user requirements including its IBM quantum account.

    :param token: the user's IBM QUantum account token
    :param min_required_qubits: the minimum number of qubits needed by the user to run a specific application
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