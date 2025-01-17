"""This code is taken from Estimator Example from EstimatorV2 Class docstring"""

import os
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as Estimator

token  = os.getenv('IBM_QUANTUM_TOKEN') 
service = QiskitRuntimeService(channel="ibm_quantum", token=token)
backend = service.least_busy(operational=True, simulator=False)

psi = RealAmplitudes(num_qubits=2, reps=2)
hamiltonian = SparsePauliOp.from_list([("II", 1), ("IZ", 2), ("XI", 3)])
theta = [0, 1, 1, 2, 3, 5]

pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
isa_psi = pm.run(psi)
isa_observables = hamiltonian.apply_layout(isa_psi.layout)

estimator = Estimator(mode=backend)

# calculate [ <psi(theta1)|hamiltonian|psi(theta)> ]
job = estimator.run([(isa_psi, isa_observables, [theta])])
job_id = job.job_id()
job_status = job.status()
print(job_id)
print(job_status)
pub_result = job.result()[0]
print(f"Expectation values: {pub_result.data.evs}")