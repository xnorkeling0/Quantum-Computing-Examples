from pathlib import Path
import pandas as pd
from math import sqrt


def get_dataset(db_path):  # TODO:move to common module
    df = pd.read_csv(db_path)
    dataset = df.values.tolist()
    print(f"The Dataset:\n{dataset}")
    return dataset


def normalize_dataset(dataset: list):
    for i in range(len(dataset)):
        base = sqrt(dataset[i][0] ** 2 + dataset[i][1] ** 2)
        dataset[i][0] = dataset[i][0] / base
        dataset[i][1] = dataset[i][1] / base
        vector_length = sqrt((dataset[i][0]) ** 2 + (dataset[i][1]) ** 2)
        print(f"Vector {i + 1} length after normalization: {vector_length}")
    return dataset


def normalize_test_set(test_set: list):
    base = sqrt(test_set[0] ** 2 + test_set[1] ** 2)
    test_set[0] = test_set[0] / base
    test_set[1] = test_set[1] / base
    vector_length = sqrt((test_set[0]) ** 2 + (test_set[1]) ** 2)
    print(f"Normalized test points:\n{test_set[0]}\n{test_set[1]}")
    print(f"test set euclidian vector length: {vector_length}")
    return test_set
