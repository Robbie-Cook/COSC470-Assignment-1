from decisionTree import Tree, DecisionTree, Data
import random 
"""
A class defining a forest (a selection of decision trees),
and methods to fit and predict them
"""

class Forest:
    """
    Initialise a new forest
    """
    def __init__(self, max_depth, num_trees, num_samples):
        # Make a forest (a set of decision trees)
        self.forest = None
        self.max_depth = max_depth
        self.num_trees = num_trees
        self.num_samples = num_samples

    """
    Fit a forest to a set of data
    """
    def fit(self, dataset):
        self.forest = []
        for i in range(self.num_trees):
            # Generate a random subset of the dataset to train the tree on
            subset = [dataset[random.randrange(0,len(dataset))] for a in range(self.num_samples)]
            self.forest.append(DecisionTree(self.max_depth).fit(subset))

        return self

    """
    Predict a forest i.e. classify a dataset
    """
    def predict(self, data):
        votes = [tree.predict(data) for tree in self.forest]
        mr_president = votes[0]
        # print("The votes are in: ", votes)
        for vote in votes:
            if votes.count(vote) > mr_president:
                mr_president = vote
        return mr_president