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
case_1  4.5      3         1 (option_1 is selected)
case_2  1        1.5       0 (option_2 is selected)
test    3.5      2         ?

in general:
-----------------------------------
        option_1 option_2  choice(label)
case_1  a        b         1
case_2  c        d         0
test    e        f         ?

Quantum Machine Learning steps:
    1.  a Specific quantum circuit is implemented using 4 qubits
    (q0, q1, q2, q3)
    2.  qubits are initialized into a state vector containing
    training data (case_1 and case_2) and query data (test)
    3.  Hadamart gate is applied on qubit 3 (hence qubits are entagled
    and they can't ne computed separately)
    4.  state vector size = 2^n (n: number of qubits) in this case
    5.  in the state vector states are increasing values starting from zero 
    6. to encode in a state vector the 6 numbers of our dataset (4.5, 3, 1, 1.5, 3.5, 2) of our 
    case study example above or in general (a,b,c,d,e,f) we need 3 qubits (Q3,Q2,Q1)
    (2^3 = 8 states = state vector size)
    7.  this is a  state vector with 8 states:
    [000, 001, 010, 011, 100, 101, 110, 111]
    8. this is the states to dataset "amplitude" encoding, and to fit the 6 data points
    to the 8 states, the query poings  (e,f) have been repeated
    
    Q3,Q2,Q1
    000 -> a
    001 -> b
    010 -> c
    011 -> d
    100 -> e
    101 -> f
    110 -> e <- repeated
    111 -> f <- repeated

    9.  Also the labels (or choice) from our study case table must be encoded using an
    additional qubit (Q0) which encodes labels 0 or 1 hence we need to expand the state vector
    by executing the tensorial product between the state vector of (Q3,Q2,Q1) and Q0=[0,1] obtaining
    a new state vector or 2^(4 qubits) = 16 states as below.

    form table remember that 
    (a,b) -> Q0=1
    (c,d) -> Q0=0

    Q3,Q2,Q1,Q0 -> |v>: state vector  values to initialize the 4 qubits
    0000 -> 0
    0001 -> a <- "a" is stored where Q0=1 (label 1)
    0010 -> 0
    0011 -> b
    0100 -> c <- "c" is stored where Q0=0 (label 0)
    0101 -> 0
    0110 -> d
    0111 -> 0
    1000 -> 0|<- 1000 and 1001 are for e (when query (e,f) is labelled 1)
    1001 -> e|
    1010 -> 0|<-1010 and 1011 are for f (when query (e,f) is labelled 1)
    1011 -> f|
    1100 -> e| <- repeated (when query (e,f) is labelled 0)
    1101 -> 0| 
    1110 -> f| <- repeated (when query (e,f) is labelled 0)
    1111 -> 0|

    10. all the data values in the 5th column must be normalized by dividing
    them by 2 to fit them into the state vector.

    The constant value 2 ensures that the sum of the squares of all those values
    adds up to 1 (because they are the result of applying an Hadamart gate to 4 qubits
    which originates a 16x16 matrix multiplied by a 1/2 constant)

    11. To the state vector obtained at point 10. it is applied the following Hadamard
    transformation T:
    T = H (x) I (x) I (x) I = (1/sqrt(2))[16 rows x 16 columns]
    where H Hdamard matrix; I identity matrix; (x):tensorial product;
    T has 16 rows x 16 columns

    12. Applying the transformation T to |v>: T|v> we obtain the following column matrix (vector):
    [0, a+e, 0, b+f, c+e, 0, d+f, 0, 0, a-e, 0, b-f, c-e, 0, d-f].
    This has to be multiplied by (1/sqrt(2)) but it will be done later for simplicity.

    13. Recalling point 9 Q3,Q2,Q1,Q0 -> |v>; we will collect measurements only when
    Q3 measurement is zero hence we set to zero all the components of the initialization
    state vector where Q3 is 1 [motivation TBD]. See below we set to zero the last eight element
    of |v>:

    Q3,Q2,Q1,Q0 -> |v>: state vector  values to initialize the 4 qubits
    0000 -> 0
    0001 -> a <- "a" is stored where Q0=1 (label 1)
    0010 -> 0
    0011 -> b
    0100 -> c <- "c" is stored where Q0=0 (label 0)
    0101 -> 0
    0110 -> d
    0111 -> 0
    1000 -> 0|<- 1000 and 1001 are for e (when query (e,f) is labelled 1)
    1001 -> 0|
    1010 -> 0|<-1010 and 1011 are for f (when query (e,f) is labelled 1)
    1011 -> 0|
    1100 -> 0| <- repeated (when query (e,f) is labelled 0)
    1101 -> 0| 
    1110 -> 0| <- repeated (when query (e,f) is labelled 0)
    1111 -> 0|

    Hence T|v> = [0, a+e, 0, b+f, c+e, 0, d+f, 0, 0, a-e, 0, b-f, c-e, 0, d-f]
    becomes 
    [0, a+e, 0, b+f, c+e, 0, d+f, 0, 0, 0, 0, 0, 0, 0, 0]
    each element of this vector corresponds to a probabiliy P() in Q0.

    14. Q0 is measured. It is of interest the probability P(1) of Q0 to be equal to 1.
    Specifically look at 
            option_1 option_2  choice(label)   P
    case_1  a        b         1               P(1) = 1 and P(0) = 0
    case_2  c        d         0
    test    e        f         ?

    For data point (a,b) case_1  the ML model choice has to be 1 hence its probability
    P(1) (the probability that this happens) has to be 1 hence the other choice is P(0)=0.

    Hence looking at the qubits to datapoint relations above we see:

    Q3,Q2,Q1,Q0 -> |v>:
    0001 -> a
    0011 -> b
    
    which in the the state vector T|v> = [0, a+e, 0, b+f, c+e, 0, d+f, 0, 0, 0, 0, 0, 0, 0, 0]
    corresponds to the elements a+e and b+f

    hence considering the probability definition:

                number of event favorable outcomes
    P(event) = -----------------------------------
                    number of all outcomes

                (a+e)^2 + (b+f)^2
   P(1) = -----------------------------------
        (a+e)^2 + (b+f)^2 + (c+e)^2 + (d+f)^2

    since all the datapoints are normalized hece they are on the same unit circle
    we have the following properties:

    a^2 + b^2 = 1
    c^2 + d^2 = 1
    e^2 + f^2 = 1

    by expanding the (a+e)^2 = a^2 + b^2 + 2ab = 2ab (do the same for the other elements
    in the P(1) formula) and we get

                1 + ae + bf
    P(1) = --------------------------
            2 + ae + bf + ce + df
     








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

