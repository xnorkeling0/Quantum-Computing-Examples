"""Quantum Computing Machine Learning Example"""


import os
import sys
import numpy as np
from qiskit import QuantumCircuit
from qiskit_ibm_runtime import SamplerV2 as Sampler
# Add the parent directory to the PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from .data_processing import get_dataset, normalize_dataset, normalize_test_set


"""
Simplified Study Case (dataset):
-----------------------------------
        option_1 option_2  choice(label)
case_1  4.5      3         1 (option_1 is selected)
case_2  1        1.5       0 (option_2 is selected)
test    3.5      2         ?

in general:
-----------------------------------
        option_1 option_2  choice(label)   Probability
case_1  a        b         1               P(1)=1
case_2  c        d         0               P(0)=0
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

To run the script, in CLI enter:
python src/quantum_machine_learning/qc_ml_knn.py
"""

class QuantumKnnModel:
    """
    A Machine Learning model trained with IBM quantum computer
    """

    def __init__(self):
        pass

    def compute_tensor_product(self, identity_size: int)-> np.ndarray:
        I = np.eye(identity_size)
        # Define the Hadamard gate
        H = (1 / np.sqrt(identity_size)) * (np.ones((identity_size, identity_size)) - 2 * np.eye(identity_size))
        # Tensor product H ⊗ I ⊗ I ⊗ I
        H_tensor_I = np.kron(H, np.kron(I, np.kron(I, I)))
        return H_tensor_I
    
    def count_data_points(self, dataset: list) -> int:
        """
        Counts the number of datapoints of a dataset.
        For example, the dataset below has 2 data points [4.5, 3, 1]
        and [1, 1.5, 2]  ([coordinate 1, coordinate 2, label])
        dataset = [
        [4.5, 3, 1],
        [1, 1.5, 2]
        ]
        """
        try:
            # Validate that the dataset is a list of lists
            if not isinstance(dataset, list) or not all(isinstance(item, list) for item in dataset):
                raise ValueError("Dataset must be a list of lists.")
            # Return the number of datapoints
            return len(dataset)
        except TypeError as e:
            raise ValueError("Datapoints not valid") from e


    def construct_state_vector(self, normalized_dataset: list) -> np.array:
        """
        Construct the initial state vector |v> dynamically from the dataset.
        Set Q3 = 1 components to zero.
        Construct the initial state vector |v> which components are function
        of the training set points ((a,b),(c,d)) and the query point (e,f).

        data points ---> label:
        (a,b) ---> 1
        (c,d) ---> 0
        (e,f) ---> ?
    
        Amplitude encoding:
        3 data points ---> sixe of |v> = 2^3 = 8
        combinations(Q3,Q2,Q1) --> points coordinates on the unit circle
        000 -> a
        001 -> b
        010 -> c
        011 -> d
        100 -> e
        101 -> f
        110 -> e <- repeated
        111 -> f <- repeated
    
        Add Q0 to store the label
        |v> components  <----> qubits combinatinos(Q3,Q2,Q1,Q0)
        state_vector = [
            0,        # 0000 Q3 = 0 states  
            a,        # 0001 Q3 = 0 states
            0,        # 0010 Q3 = 0 states
            b,        # 0011 Q3 = 0 states
            c,        # 0100 Q3 = 0 states
            0,        # 0101 Q3 = 0 states
            d,        # 0110 Q3 = 0 states
            0,        # 0111 Q3 = 0 states
            0,        # 1000 Q3 = 1 states
            0,        # 1001 Q3 = 1 states
            0,        # 1010 Q3 = 1 states
            0,        # 1011 Q3 = 1 states
            0,        # 1100 Q3 = 1 states
            0,        # 1101 Q3 = 1 states
            0,        # 1110 Q3 = 1 states
            0         # 1111 Q3 = 1 states
        ]
        """
        n_data = len(normalized_dataset)
        assert n_data % 2 == 0, "Dataset must have an even number of elements (pairs for labels)."
        # Map dataset to amplitudes
        a, b, c, d, e, f = normalized_dataset  # Automatically assign values
        state_vector = [ # column matrix
            0, a, 0, b,  # Q3=0, Labels 1 (a, b)
            c, 0, d, 0,  # Q3=0, Labels 0 (c, d)
            0, 0, 0, 0,  # Q3=1 (set to 0) (e, f)
            0, 0, 0, 0   # Q3=1 (set to 0) (e, f)
        ]
        return np.array(state_vector)


    def compute_initial_state(self, db_path, test_set) -> list:
        dataset = normalize_dataset(get_dataset(db_path))
        test = normalize_test_set(test_set)
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
        return initial_state

    def knn_quantum_circuit(self, initial_state):
        """
        Initialize the 4 quibits Q3Q2Q1Q0 state vector with amplitude encoding
        see point 13. in the docstring. Here the vector has also been multiplied
        by the factor 1/2 from the 4qubits Hadamard operator.
        """
        circuit = QuantumCircuit(4,2) # 4 qubits, 2 classical
        circuit.initialize(initial_state)
        circuit.h(3) # add Hadamart gate on qubit 3; 
        circuit.measure(3,0) # Qubit Q3 measured value is stored into classical bit 0 
        circuit.measure(0,1) # Qubit Q0 measured value is stored into classical bit 1
        return circuit

    def execute_knn_model_on_quantum_computer(self, backend, qc_transpiled):
        """
        Execute on a Quantum Computer using the Sampler primitive
        Getting counts for separate registers
        https://quantumcomputing.stackexchange.com/questions/40735/getting-combined-counts-when-using-qiskit-ibm-runtime-samplerv2/40736#40736

        counts contains the values for the two classical bit c0 and c1 in the form c1c0 --> Q0Q1
        In the circuit c0 measure Q0 and c1 measures Q3
        c0 --> Q3
        c1 --> Q0
        we need to compute probability of Q0 = 1 when Q3 is zero
        hence we need to count statistics components to obtain  a numerator 
        and a denominator for the probability formula
        denominator is counted only when Q3 is zero (see explanation point 13.)
        under this condition the numerator is counted 
        """
    
        shots = 1
        sampler = Sampler(mode=backend)
        sampler.options.default_shots = shots  # Options can be set using auto-complete.
        numerator = 0
        denominator = 0
        shots = 50
        job_cnt = 0
        for i in range(shots):
            job = sampler.run([qc_transpiled])
            job_cnt+=1
            print(f"Job ID: {job.job_id()} | number: {job_cnt} | status: {job.status()}")
            result = job.result()[0]
            counts = result.join_data().get_counts()
            if ("00" in counts or "10" in counts):
                denominator += 1  # Increment denominator if condition is met
                if "10" in counts :
                    numerator += 1  # Increment numerator

        # for bitstring, count in counts.items():
        #     print(f"{bitstring}: {count}")
        """
         Printed result: 
         strings and their occurencies. Basically per each string it is printed
         how many times it was detected when running the given circuit for the
         specified number of shots.
             01: 602
             00: 9465
             10: 9716
             11: 217
         """

        if denominator !=0:
            p1 = numerator/denominator
            p2 = (denominator-numerator)/denominator
            return p1, p2
        else:
            print("Division by zero detected in probability formula")

