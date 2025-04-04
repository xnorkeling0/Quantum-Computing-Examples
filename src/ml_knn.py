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
"""



