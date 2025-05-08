from pathlib import Path
from ml_knn import KnnModel
from qc_ml_knn import QuantumKnnModel
from utils.save_account import transpile_circuit

if __name__ == "__main__":
    """
    The dataset from src/dataset.csv:
    (It needs an empty row above the first data row)
    dataset = [
        [4.5, 3, 1],
        [1, 1.5, 2]
    ]
    """
    current_dir = Path(__file__).parent
    db_path = current_dir / "dataset.csv"
    test_set = [3.5, 2]

    print("Running classic Knn Model")
    ml_model_inst = KnnModel(db_path, test_set)
    ml_model_inst.run()

    print("Running quantum Knn Model")
    qc_knn_model = QuantumKnnModel()
    initial_state = qc_knn_model.compute_initial_state(db_path, test_set)
    circuit = qc_knn_model.knn_quantum_circuit(initial_state)
    backend, qc_transpiled = transpile_circuit(circuit)
    p1, p2, _, _, = qc_knn_model.execute_knn_model_on_quantum_computer(backend, qc_transpiled)
    print(f"P(1) = {p1}, P(0)={p2}")
    if p1 >= p2:
        print("Option 1 is better")
    else:
        print("Option 2 is better")
