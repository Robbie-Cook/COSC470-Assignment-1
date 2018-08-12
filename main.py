from decisionTree import DecisionTree, Tree, Data
from forest import Forest

from sklearn.datasets import load_iris, load_digits, load_wine, load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn import tree
from enum import Enum
from sklearn import metrics
import random
import copy
import math




"""
Class defining which algoriths are avaliable
"""
class Algorithm(Enum):
    RANDOM_FOREST = 1
    ADABOOST = 2
    DECISION_TREE = 3

"""
Settings
"""

# Main settings
algorithm = Algorithm.RANDOM_FOREST
max_depth = 20
skset = load_breast_cancer() # Set to use

# Random Forest settings
num_trees = 10   # How many trees to make in the forest 
num_samples = 150   # How much data to train the trees on (with replacement)
forest = None # The forest

"""
Check the accuracy of the output using cross-validation

Test percentage -- the percent of how many samples to have in the text data
set e.g. 20% test data, 80% training data
dt -- the decision tree
"""
def check_accuracy(dt, dataset, test_percentage=20, num_repeats=3):
    training_set = copy.deepcopy(dataset)
    accuracies = []

    num_test_samples = int(len(training_set)/(100/test_percentage))

    for i in range(num_repeats):
        random.shuffle(training_set)
        test_set = [training_set.pop(random.randrange(len(training_set))) for i in range(num_test_samples)]

        # Fit the dt
        tree = dt.fit(training_set)
        values = [tree.predict(a) for a in test_set]
        expected = [a.classification for a in test_set]

        same = 0
        for i in range(len(values)):
            if values[i] == expected[i]:
                same += 1

        accuracies.append( (same/len(values)) *100)
        [training_set.append(i) for i in test_set] # test set overwritten later,
                                                  # so no need to pop
    average_accuracy = sum(accuracies)/len(accuracies)
    std_dev = 0
    for a in accuracies:
        std_dev += ((a - average_accuracy)**2) 
    std_dev = math.sqrt(1/len(accuracies) * std_dev)
    
    print("Accuracy: {}, Std dev: {}".format(average_accuracy, std_dev))
    return (average_accuracy, std_dev)

"""
Main program. Uses decisionTree.py as a support program to classify data
"""

values = [list(a) for a in skset.data]
targets = [int(a) for a in skset.target]
dataset = [Data(values[i], targets[i]) for i in range(len(values))]
# print([str(a) for a in dataset])
# dataset = d.processFile('data/digitsModified.txt')

# Basic decision tree algorithm
if algorithm == Algorithm.DECISION_TREE:
    d = DecisionTree(max_depth=max_depth)
    # tree = d.fit(datase)
    check_accuracy(dt=d, dataset=dataset, num_repeats=10)

# Random forest algorithm
elif algorithm == Algorithm.RANDOM_FOREST:
    forest = Forest(max_depth=max_depth, num_trees=num_trees, num_samples=num_samples)
    # f = forest.fit(dataset)
    check_accuracy(dt=forest, dataset=dataset, num_repeats=10)

# Adaboost algorithm
elif algorithm = Algorithm.ad
    
