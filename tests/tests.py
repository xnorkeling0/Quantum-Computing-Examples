import matplotlib.pyplot as plt
import pytest
from src.quantum_machine_learning.qc_ml_knn import QuantumKnnModel
from unittest.mock import MagicMock, patch


class TestQuantumMachineLearningModel:

    def test_plot_tensor_product(self):
        """
        To visualize on screen the tensor product matrix
        H ⊗ I ⊗ I ⊗ I
        """
        qml_model = QuantumKnnModel()
        identity_size = 2 # to match Hadamard gate size
        H_tensor_I = qml_model.compute_tensor_product(identity_size)
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
        identity_size = 2 # to match Hadamard gate size
        H_tensor_I = qml_model.compute_tensor_product(identity_size)
        assert H_tensor_I.shape == (16, 16)
    
    def test_count_data_points(self):
        """
        Tests if the count_data_points method can effectively
        count the datapoint in a dataset.
        """
        training_set = [
        [4.5, 3, 1],
        [1, 1.5, 2]
        ]
        query_set = [[3.5,2]]
        qml_model = QuantumKnnModel()
        number_of_datapoints = qml_model.count_data_points(training_set, query_set)

        assert number_of_datapoints == 3
    
    def test_generate_qubit_combinations(self):
        num_points = 3
        qml_model = QuantumKnnModel()
        qubit_combinations = qml_model.generate_qubit_combinations(num_points)
        expected_list = [
                "000",
                "001",
                "010",
                "011",
                "100",
                "101",
                "110",
                "111"
        ]
        assert qubit_combinations == expected_list
    
@patch("src.quantum_machine_learning.qc_ml_knn.Sampler")  
def test_execute_knn_model_on_quantum_computer(mock_sampler_class):
    """Test quantum KNN model execution."""
    mock_sampler = MagicMock()  # Create the mock instance
    mock_sampler_class.return_value = mock_sampler  # Replace Sampler with our mock
    mock_job = MagicMock()
    
    # Generate counts dynamically for each call
    def mock_get_counts():
        """Simulate different results per iteration."""
        return {"10": 1} if mock_sampler.run.call_count <= 30 else {"00": 1}
    
    mock_result = MagicMock()
    mock_result.join_data.return_value.get_counts.side_effect = mock_get_counts
    mock_job.result.return_value = [mock_result]

    mock_sampler.run.return_value = mock_job  # Ensure run() returns the mock job

    backend = "mock_backend"
    qc_transpiled = "mock_circuit"

    model = QuantumKnnModel()
    p1, p2, numerator, denominator = model.execute_knn_model_on_quantum_computer(backend, qc_transpiled)

    # Verify that the mock's run method was called
    mock_sampler.run.assert_called_with([qc_transpiled])
    assert p1 >= 0.5
    assert p2 < 0.5
    assert mock_sampler.run.call_count == 50, f"Expected 50 calls, got {mock_sampler.run.call_count}"
    assert numerator == 30, f"Expected numerator to be 30, got {numerator}"
    assert denominator == 50, f"Expected denominator to be 50, got {denominator}"