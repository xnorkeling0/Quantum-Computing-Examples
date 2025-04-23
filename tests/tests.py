import matplotlib.pyplot as plt
from src.quantum_machine_learning.qc_ml_knn import QuantumKnnModel

class TestQuantumMachineLearningModel:

    def test_plot_tensor_product(self):
        """
        To visualize on screen the tensor product matrix
        H ⊗ I ⊗ I ⊗ I
        """
        qml_model = QuantumKnnModel()
        H_tensor_I = qml_model.compute_tensor_product(2)
        plt.imshow(H_tensor_I, cmap='viridis', interpolation='nearest')
        cbar = plt.colorbar(label="Matrix Values")  
        cbar.set_ticks([H_tensor_I.min(), H_tensor_I.max()])  # Set ticks to the extreme values
        cbar.set_ticklabels([f"Min: {H_tensor_I.min():.2f}", f"Max: {H_tensor_I.max():.2f}"])  # Label ticks
        plt.title("Visualization of H ⊗ I ⊗ I ⊗ I")
        plt.xlabel("Columns")
        plt.ylabel("Rows")
        plt.show()
    
    def test_dimensions(self):
        """
        Test if the matrix dimensions are correct (16x16 for 4 qubits)
        """
        qml_model = QuantumKnnModel()
        H_tensor_I = qml_model.compute_tensor_product(2)
        assert H_tensor_I.shape == (16, 16)