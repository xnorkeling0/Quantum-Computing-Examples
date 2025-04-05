"""
Machine Learning Example with the KNN Method

Study Case:
-----------
Train a model that can recognize the best option (e.g., best restaurants
, hotel) given their rating from online users reviews.

Reviews possible values from 1 to 4 stars:
1, 1.5, 2, 2.5, 3, 3.5, 4
The choice is 1 if option_1 > option_2 else it is 2 (including the
case where option_1 = option_2)

Step #1: grab the following data from a spreadsheet

        option_1 option_2  choice
case_2  4.5      3         1
case_1  1        1.5       2
case_3  4        4         1
query   3.5      2         ?


Step #2: Normalize Options (no_<option_number>)

no_1 = option_1/(option_1^2 + option_2^2)
no_2 = option_2/(option_1^2 + option_2^2)

Step #3: Compute Squared Euclidean Distances
Step #4: Compute Weights
Step #5: Normalize Weights
Step #6: Print Results

Simplified Study Case (as starter):
-----------------------------------
        option_1 option_2  choice
case_1  4.5      3         1
case_2  1        1.5       2
test    3.5      2         ?
"""

# Simplified Study Case:
from math import sqrt


dataset = [[4.5, 3, 1],[1, 1.5, 2]]
test = [3.5, 2]

# Training set normalization:
for i in range(len(dataset)):
    base = sqrt(dataset[i][0]**2 + dataset[i][1]**2)
    dataset[i][0] = dataset[i][0]/base
    dataset[i][1] = dataset[i][1]/base
    vector_length = sqrt((dataset[i][0])**2 + (dataset[i][1])**2)
    print(f"Vector {i + 1} length after normalization: {vector_length}")

# Test set normalization
base = sqrt(test[0]**2 + test[1]**2)
test[0] = test[0]/base
test[1] = test[1]/base
vector_length = sqrt((test[0])**2 + (test[1])**2)
print(f"Normalized test points:\n{test[0]}\n{test[1]}")
print(f"Vector {i + 1} length after normalization: {vector_length}")

# Squared Euclidean Distances and weights
weight = []
print(f"Distances:")
for i in range(len(test)):
    squared_euclidean_distance = (test[0]-dataset[i][0])**2 + (test[1]-dataset[i][1])**2
    weight.append(1 - 0.25*squared_euclidean_distance)
    print(f"from point {i}: {squared_euclidean_distance} with weight of {weight[i]}")

# Weighs normalization:
base = 0
for i in range(len(weight)):
    base = base + weight[i]
sum = 0
for i in range(len(weight)):
    weight[i] = weight[i] / base
    sum = sum + weight[i]
    print(f"Normalized weight for point{i}: {weight[i]}")
print(f"Sum of normalized weights is {sum}")

    



if __name__ == "__main__":
    pass


