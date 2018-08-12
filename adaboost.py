from decisionTree import Data,DecisionTree
import math
import random
import numpy as np

"""
Class for the Adaboost algorithm
"""
class Adaboost:
    def __init__(self):
        self.alphas = []
        self.classifiers = []
    
    """
    Fit method -- builds a classifier using boosting

    One difference I did was with the weights -- weights influence the
    probability that the item will be chosen for the classifier rather than
    influencing the imparity of the split.
    i.e. bagging is used with boosting
    """
    def fit(self, dataset, T=100, bag_perc=60, bag_size=None):
        if bag_size == None:
            bag_size = int(len(dataset)/(100/bag_perc)) # e.g. 20% of dataset
        
        # Set initial weights for all data items
        for item in dataset:
            item.weight = 1/len(dataset)    

        for t in range(T):
            weights = [a.weight for a in dataset] # List of weights

            # Normalise the weights vector
            for item in dataset:
                item.weight = item.weight/sum(weights)  
            weights = [a.weight for a in dataset] # List of weights have changed

            # Pick a weighted subset
            indices = np.random.choice(len(weights), bag_size, replace=False, p=weights)
            subset = [dataset[i] for i in indices]

            # Make and train a new classifier
            stump = DecisionTree(max_depth=1).fit(subset)
            self.classifiers.append(stump)

            # Calculate e
            numerator = 0
            for d in subset:
                numerator += d.weight * indicator(d,stump)
            denominator = sum([d.weight for d in subset])     
            e = float(numerator)/denominator

            # Calculate alpha (slightly different to in lecture)
            alpha = math.log((1-e)/e)
            self.alphas.append(alpha)

            # Calculate the weight change for each data item
            for d in subset:
                d.weight = d.weight * math.exp(alpha*indicator(d,stump)) 

        return self

    """
    Prediction method for the adaboost algorithm
    Takes a single Data class as a parameter
    """
    def predict(self, data):
        assert len(self.alphas) > 0
        assert len(self.classifiers) == len(self.alphas)

        value = 0
        for i in range(len(self.alphas)):
            value += self.alphas[i] * (2*self.classifiers[i].predict(data)-1)

        myclass = (0 if value < 0 else 1)
        return myclass
        
"""
Returns 1 if dataset's classification is *not* equal to the predicted value
"""
def indicator(d, tree):
    return (1 if tree.predict(d) != d.classification else 0)




        
        