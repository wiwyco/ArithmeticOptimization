ARITHMETIC OPTIMIZER README
====================================
By Winslow Conneen
7.14.2021


DESCRIPTION OF FUNCTION
------------------------------------
This module is for use in the preprocessing stage of the machine learning model building process. Reductions in input
dimensionality and increases in variable meaningfulness both result in an increase in accuracy for a given model. To
accomplish both these goals, this module uses basic arithmetic combinations of features to produce features that
better partition frequently occurring data structures for the purposes of predicting a target variable more effectively.

For example, the variables X and Y would create the subsequent features X, Y, X+X, X+Y, Y+Y, X*X, X*Y, Y*Y, X-Y, Y-X, X/Y, and Y/X.

This frame is then run through a multiple regression and the variables with the lowest p-values are selected and
assembled into a final dataframe with the dependant variable.

HOW TO USE
------------------------------------
Input the full dataframe, including the dependant variable(s), the list of dependant feature names, the quantity of
repetitions through the algorithm, and the number of independant variables that you would like to derive. For example:

Num_repetitions = 2
Num_variables = 3

ArithmeticOptimization(dataframe, ["DependantVar"], Num_repetitons, Num_variables)
