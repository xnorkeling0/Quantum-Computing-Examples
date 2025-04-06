"""Quantum Computing Machine Learning Example"""

from qiskit import QuantumCircuit
import pandas as pd

import os
import sys
# Add the parent directory to the PYTHONPATH 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.save_account import save_account, get_first_available_backend
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.primitives import StatevectorSampler, StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorOptions, EstimatorV2 as Estimator
from qiskit_ibm_runtime.options.resilience_options import ResilienceOptionsV2

db_path = "src/dataset.csv"

def get_dataset(db_path):
    df = pd.read_csv(db_path)
    dataset = df.values.tolist()
    print(f"The Dataset:\n{dataset}")
    return dataset
    # The dataset from src/dataset.csv:
    # (It needs an empty row above the first data row)
    # dataset = [
    #     [4.5, 3, 1],
    #     [1, 1.5, 2]
    # ]

"""
Simplified Study Case (dataset):
-----------------------------------
        option_1 option_2  choice(label)
case_1  4.5      3         1
case_2  1        1.5       2
test    3.5      2         ?
"""

dataset = get_dataset(db_path)
test = [3.5, 2]

initial_state = [
    0,
    dataset[0][0]/2,
    0,
    dataset[0][1]/2,
    dataset[1][0]/2,
    0,
    dataset[1][1]/2,
    0,
    0,
    test[0]/2,
    0,
    test[1]/2,
    test[0]/2,
    0,
    test[1]/2,
    0
]
circuit = QuantumCircuit(4,2) # 4 qubits, 2 classical
circuit.initialize(initial_state)
circuit.h(3) # add Hadamart gate on qubit 3; 
circuit.measure(3,0)
circuit.measure(0,1)

# 2. Define the observable to be measured 
operator = SparsePauliOp.from_list([("XXY", 1), ("XYX", 1), ("YXX", 1), ("YYY", -1)])
print(f"The operators strings:\n{operator}")
# 3. Optimize the problem
token  = os.getenv('IBM_QUANTUM_TOKEN') # getting the custom env variable that stores my IBM token
service = QiskitRuntimeService(channel="ibm_quantum", token=token)
backend = service.least_busy(operational=True, simulator=False)
pass_manager = generate_preset_pass_manager(backend=backend, optimization_level=1)
qc_transpiled = pass_manager.run(circuit)
operator_transpiled = operator.apply_layout(layout=qc_transpiled.layout)

