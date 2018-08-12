import numpy as np

"""
Class which defines a value,classification pair
"""
class Data:
    def __init__(self, values, classification=None):
        assert type(values) is list

        self.values = values
        self.classification = classification

    def __str__(self):
        return "Values: {}, Class: {}".format(self.values, self.classification)


"""
Defines a value point -- contains the attribute (the dimension to use) e.g.

                                                Class
-----------------------------------         ---------------
|    0.3    |   2.1    |     0     |        |      0      |
-----------------------------------         ---------------
                ^
                |
                1

and the value to value on e.g. 2.1
"""
class Split:
    def __init__(self, attribute, value, score, left=None, right=None):
        assert type(attribute) == int
        assert type(value) == float
        assert type(score) == float

        self.attribute = attribute
        self.value = value
        self.score = score
        self.left = left
        self.right = right
    
    def __str__(self):
        if self.left == None or self.right == None:
            return "Column {} < {} (Score: {})".format(self.attribute, self.value, self.score)
        else:
            return "Column {} < {} (Score: {}), Left: {}\n\n, Right: {}".format(
                self.attribute, self.value, self.score, [str(i) for i in self.left], [str(i) for i in self.right])


"""
Make a new tree, which contains a split (criteria to split by), and two leaf branches 
"""
class Tree:
    def __init__(self, dataset):
        self.left = None # the left tree 
        self.right = None # the right tree
        self.dataset = dataset
        self.classification = mostCommonClass(dataset) # Which class to assign
                                    # the branch
        self.split = None
    
    def __str__(self):
        attribute = ""
        value = ""
            
        if self.left != None or self.right != None:
            attribute, value = self.split.attribute, self.split.value
            return "attr({}) < {}:\n (Left:{}\n Right: {})\n\n".format(str(attribute),
                                                  str(value),
                                                  str(self.left),
                                                  str(self.right))
        
        return ""

    """
    Predict the class of a given set
    data -- must be a Data object
    """
    def predict(self, data):
        if self.split == None:
            return self.classification

        if data.values[self.split.attribute] < self.split.value:
            return self.left.predict(data)
        else:
            return self.right.predict(data)



"""
Main decision tree class to be used by outside methods
"""
class DecisionTree:
    def __init__(self, max_depth):
        self.max_depth = max_depth

    """
    Fit the data using my implementation of Data instead of [X,y]
    """
    def fit(self, dataset):
        self.classifications = list(set([d.classification for d in dataset]))
        global classifications
        classifications = self.classifications
        return self.growTree(dataset)
    
    # """
    # Fit the data using the standard (X,y) way by transferring the information
    # to a list of Data objects

    # X -- the inputs
    # y -- the expected outputs
    # """
    # def fit(self, x, y):
    #     assert len(x) == len(y)

    #     dataset = [Data(list(x[i]), int(y[i])) for i in range(len(x))]
    #     return self.data_fit(dataset)
    """
    Main recursive splitting method used with data_fit()
    """
    def growTree(self, dataset, depth=0):
        assert type(dataset) == list

        m = Tree(dataset)

        # Stopping conditions 
        #   -- only one item or all items are of the same class
        #   -- max depth reached
        if depth >= self.max_depth:
            m.classification = mostCommonClass(dataset)
            return m

        if len(dataset) == 1:
            m.classification = dataset[0].classification
            return m

        allSame = True
        for item in dataset:
            if item.classification != dataset[0].classification:
                allSame = False     
        if allSame:
            m.classification = dataset[0].classification
            return m



        # Assign the best split to the tree
        m.split = self.find_split(dataset)

        if len(m.split.left) == 0 or len(m.split.right) == 0:
            print("Caution -- false split")

        # Run the algorithm recursively
        m.left = self.growTree(dataset = m.split.left, depth=depth+1)
        m.right = self.growTree(dataset = m.split.right, depth=depth+1)
        
        return m

    """
    Get the proportion of a given class and dataset
    (i.e. proportion(X,Y))
    """
    def proportion(self, dataset, classification):
        assert type(dataset) == list
        assert type(classification) == int
        if( len(dataset) == 0 ):
            return 0

        classes = [a.classification for a in dataset]
        proportion = float(classes.count(classification)) / len(classes)
        return proportion


    """
    Get imparity of a dataset using the gini attribute
    """
    def imparity(self, dataset):
        assert type(dataset) == list

        gini = 0
        for c in self.classifications:
            p_k = self.proportion(dataset, c)
            gini += p_k * (1 - p_k)

        return gini


    """
    Finds the best value of data (currently from one attribute)
    """
    def find_split(self, dataset):

        best_split = Split(
                        attribute = 0,
                        value = float(np.inf),
                        score = float(np.inf))

        attributes = range(len(dataset[0].values))
        for attribute in attributes:
            split_values = [data.values[attribute] for data in dataset]
            for value in split_values: # The value to value the groups by
                left, right = [],[]  # Make the two groups of the current value
                
                # Split dataset
                for row in dataset:
                    if row.values[attribute] < value:
                        left.append(row)
                    else:
                        right.append(row)

                # Get imparity of the two datasets and add them
                score = self.imparity(left) + self.imparity(right)

                # If neither left nor right is empty -- i.e. stops false splits
                if len(left) > 0 and len(right) > 0:
                    # Update best score if its better
                    if score < best_split.score:
                        best_split = Split(
                                        left = left,
                                        right = right,
                                        attribute=attribute, 
                                        value=float(value),
                                        score=float(score))

        # print("Best split found", str(best_split))
        return best_split

    """
    Process a file and return an appropriate data set
    e.g. file1.txt (
        1 1 1    0
        2 1 0    0
        ...
    ) 

    -->

    [
        Data([1,1,1],0),
        Data([2,1,0],0),
        ...
    ]
    """
    def processFile(self, myfile):
        myfile = open(myfile, 'r')
        mylist = []
        line = myfile.readline().split()
        while len(line) != 0:
            mylist.append(Data([float(a) for a in line[:-1]], int(line[-1])))
            line = myfile.readline().split()

        return mylist

"""
Find the most common classification in a dataset
"""
def mostCommonClass(dataset):
    assert len(dataset) != 0

    classInstances = [a.classification for a in dataset]
    biggest = classifications[0]
    score = 1
    for c in classifications:
        num = classInstances.count(c)
        if num > score:
            biggest = c
            score = num

    return biggest