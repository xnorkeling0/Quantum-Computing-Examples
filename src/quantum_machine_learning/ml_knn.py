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
case_1  4.5      3         1 (option_1 is selected)
case_2  1        1.5       0 (option_2 is selected)
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
case_1  4.5      3         1 (option_1 is selected)
case_2  1        1.5       0 (option_2 is selected)
test    3.5      2         ?
"""

# Simplified Study Case:
from pathlib import Path
from data_processing import get_dataset, normalize_dataset, normalize_test_set


class KnnModel:

    def __init__(self, db_path, test):
        self.db_path: str = db_path
        self.test: list = test

    def decision(self, weight):
        # ML model taking a decision
        largest = max(weight)
        index_of_largest = weight.index(largest)
        decision = f"Option {index_of_largest+1} is better"
        return decision
    
    def compute_weights(self, dataset, test):
        """
        sqd = squared_euclidean_distance: it measures the distance between
        two points on the unit circle
        weight: it measures the closeness between the same two points
        weight = 1 - sqd/4
        
        Quantities are all normalized in [0,1] the unit circle radius
        since the problem relates to probability which is always in [0,1].
        sqd in [0,4] hence to bring it in [0,1] it is divided by 4.

        In the following dataset, point_1 = case_1 and point_2 = case_2
        For each point the sqd from the query point (aka test) is computed.
        Simplified Study Case (as starter):
        -----------------------------------
                option_1 option_2  choice  weights
        case_1  4.5      3         1       w1
        case_2  1        1.5       2       w2
        test    3.5      2         ?       -

        Then the probabilities of each data point (aka case) are:
        P(1) = w1/(w1+w2)
        P(0) = w2/(w1+w2)
        
        """
        weight = []
        print(f"Distances:")
        for i in range(len(test)):
            squared_euclidean_distance = (test[0]-dataset[i][0])**2 + (test[1]-dataset[i][1])**2
            weight.append(1 - 0.25*squared_euclidean_distance)
            print(f"from point {i}: {squared_euclidean_distance} with weight of {weight[i]}")
        return weight
    
    def weights_normalization(self, weight):
        base = 0
        for i in range(len(weight)):
            base = base + weight[i]
        sum = 0
        for i in range(len(weight)):
            weight[i] = weight[i] / base
            sum = sum + weight[i]
            print(f"Normalized weight for point{i}: {weight[i]}")
        print(f"Sum of normalized weights is {sum}")
        return weight

    def run(self):
        dataset = get_dataset(self.db_path)
        dataset = normalize_dataset(dataset)
        test = normalize_test_set(self.test)
        weight = self.compute_weights(dataset, test)
        weight = self.weights_normalization(weight)
        print(self.decision(weight))



