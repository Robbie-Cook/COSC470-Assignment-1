from decisionTree import Data,DecisionTree
import math
import random

"""
Class for the Adaboost algorithm
"""
class Adaboost:
    def __init__(self):
        pass
    
    """
    Fit method -- builds a classifier using boosting
    """
    def fit(self, dataset, T=10, bag_size=50):
        # Set initial weights for all data items
        for item in dataset:
            item.weight = 1/len(dataset)    

        for t in range(T):
            

            # Make and train a new classifier
            stump = DecisionTree(max_depth=1).fit(dataset)

            # Calculate e
            numerator = 0
            for d in dataset:
                numerator += d.weight * indicator(d,stump)

            denominator = sum([d.weight for d in dataset])     
            e = float(numerator)/denominator

            # Calculate alpha (slightly different to in lecture)
            alpha = math.log((1-e)/e)

            # Calculate the weight change for each data item
            for d in dataset:
                d.weight = d.weight * math.exp(alpha*indicator(d,stump)) 

"""
Returns 1 if dataset's classification is *not* equal to the predicted value
"""
def indicator(d, tree):
    return (1 if tree.predict(d) != d.classification else 0)




        
        