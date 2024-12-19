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