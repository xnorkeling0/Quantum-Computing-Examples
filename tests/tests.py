from quantum_machine_learning.qc_ml_knn import QuantumKnnModel

class TestQuantumMachineLearningModel:

    def test_compute_tensor_product(self):
        qml_model = QuantumKnnModel()
        H_tensor_I = qml_model.compute_tensor_product
        print(H_tensor_I)