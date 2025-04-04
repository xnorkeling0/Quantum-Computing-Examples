"""
Machine Learning Example with the KNN Method

Study Case:
Train a model that can recognize the best option (e.g., best restaurants
, hotel) given their rating from online users reviews.

Reviews possible values from 1 to 4 stars:
1, 1.5, 2, 2.5, 3, 3.5, 4
The choice is 1 if Option_1 > Option_2 else it is 2 (including the
case where Option_1 = Option_2)

        Option_1 Option_2  Choice
Case_1  1        1.5       2
Case_2  4.5      3         1
Case_3  4        4         1
Query   3.5      2         ?


Step #1: Normalized Options (NOs)

NO1 = Option_1/(Option_1^2 + Option_2^2)
NO1 = Option_1/(Option_1^2 + Option_2^2)
"""



