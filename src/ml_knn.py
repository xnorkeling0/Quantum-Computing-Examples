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

#base = sqrt(dataset[0][0]**2 + dataset[0][1]**2)
#dataset[0][0] = dataset[0][0]/base
#dataset[0][1] = dataset[0][1]/base
#vector_length = print(sqrt((dataset[0][0])**2 + (dataset[0][1])**2))
#
#base = sqrt(dataset[1][0]**2 + dataset[1][1]**2)
#dataset[1][0] = dataset[1][0]/base
#dataset[1][1] = dataset[1][1]/base
#vector_length = print(sqrt((dataset[1][0])**2 + (dataset[1][1])**2))

for i in range(len(dataset)):
    base = sqrt(dataset[i][0]**2 + dataset[i][1]**2)
    dataset[i][0] = dataset[i][0]/base
    dataset[i][1] = dataset[i][1]/base
    vector_length = sqrt((dataset[i][0])**2 + (dataset[i][1])**2)
    print(f"Vector {i + 1} length after normalization: {vector_length}")





if __name__ == "__main__":
    pass


