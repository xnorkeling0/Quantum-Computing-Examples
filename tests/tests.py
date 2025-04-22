import os
import sys
import matplotlib.pyplot as plt
from src.quantum_machine_learning.qc_ml_knn import QuantumKnnModel

# class TestQuantumMachineLearningModel:
# 
#     def test_compute_tensor_product(self):
#         qml_model = QuantumKnnModel()
#         H_tensor_I = qml_model.compute_tensor_product
#         plt.imshow(H_tensor_I, cmap='viridis', interpolation='nearest')
#         plt.colorbar(label="Matrix Values")
#         plt.title("Visualization of H ⊗ I ⊗ I ⊗ I")
#         plt.xlabel("Columns")
#         plt.ylabel("Rows")
#         plt.show()
    
qml_model = QuantumKnnModel()
H_tensor_I = qml_model.compute_tensor_product()
H_tensor_I.astype(float)
# print(H_tensor_I)
plt.imshow(H_tensor_I, cmap='viridis', interpolation='nearest')
plt.colorbar(label="Matrix Values")
plt.title("Visualization of H ⊗ I ⊗ I ⊗ I")
plt.xlabel("Columns")
plt.ylabel("Rows")
plt.show()
if __name__ == "__main__":
    pass