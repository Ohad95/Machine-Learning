#!/usr/bin/env python
# coding: utf-8

# # Exercise 2: Decision Trees
# 
# In this assignment you will implement a Decision Tree algorithm as learned in class.
# 
# ## Read the following instructions carefully:
# 
# 1. This jupyter notebook contains all the step by step instructions needed for this exercise.
# 1. Submission includes this notebook only with the exercise number and your ID as the filename. For example: `hw2_123456789_987654321.ipynb` if you submitted in pairs and `hw2_123456789.ipynb` if you submitted the exercise alone.
# 1. Write **efficient vectorized** code whenever possible. Some calculations in this exercise take several minutes when implemented efficiently, and might take much longer otherwise. Unnecessary loops will result in point deduction.
# 1. You are responsible for the correctness of your code and should add as many tests as you see fit. Tests will not be graded nor checked.
# 1. Write your functions in this notebook only. **Do not create Python modules and import them**.
# 1. You are allowed to use functions and methods from the [Python Standard Library](https://docs.python.org/3/library/) and [numpy](https://www.numpy.org/devdocs/reference/) only. **Do not import anything else.**
# 1. Your code must run without errors. Make sure your `numpy` version is at least 1.15.4 and that you are using at least python 3.6. Changes of the configuration we provided are at your own risk. Any code that cannot run will not be graded.
# 1. Write your own code. Cheating will not be tolerated.
# 1. Answers to qualitative questions should be written in **markdown** cells (with $\LaTeX$ support). Answers that will be written in commented code blocks will not be checked.
# 
# ## In this exercise you will perform the following:
# 1. Practice OOP in python.
# 2. Implement two impurity measures: Gini and Entropy.
# 3. Construct a decision tree algorithm.
# 4. Prune the tree to achieve better results.
# 5. Visualize your results.

# # I have read and understood the instructions: *** YOUR ID HERE ***

# In[47]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# make matplotlib figures appear inline in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')


# ## Warmup - OOP in python
# 
# Our desicion tree will be implemented using a dedicated python class. Python classes are very similar to classes in Java.
# 
# 
# You can use the following [site](https://jeffknupp.com/blog/2014/06/18/improve-your-python-python-classes-and-object-oriented-programming/) to learn about classes in python.

# In[48]:


class Node(object):  # class DecisionNode:
    def __init__(self, data=None):
        self.data = data
        self.children = []

    def add_child(self, node):
        self.children.append(node)
        
    def set_data(self, data):
        self.data = data


# In[49]:


n = Node(5)
p = Node(6)
q = Node(7)
n.add_child(p)
n.add_child(q)
n.children


# ## Data preprocessing
# 
# For the following exercise, we will use a dataset containing mushroom data `agaricus-lepiota.csv`. 
# 
# This data set includes descriptions of hypothetical samples corresponding to 23 species of gilled mushrooms in the Agaricus and Lepiota Family. Each species is identified as definitely edible, definitely poisonous, or of unknown edibility and not recommended. This latter class was combined with the poisonous
# one (=there are only two classes **edible** and **poisonous**). 
#     
# The dataset contains 8124 observations with 22 features:
# 1. cap-shape: bell=b,conical=c,convex=x,flat=f,knobbed=k,sunken=s
# 2. cap-surface: fibrous=f,grooves=g,scaly=y,smooth=s
# 3. cap-color: brown=n,buff=b,cinnamon=c,gray=g,green=r,pink=p,purple=u,red=e,white=w,yellow=y
# 4. bruises: bruises=t,no=f
# 5. odor: almond=a,anise=l,creosote=c,fishy=y,foul=f, musty=m,none=n,pungent=p,spicy=s
# 6. gill-attachment: attached=a,descending=d,free=f,notched=n
# 7. gill-spacing: close=c,crowded=w,distant=d
# 8. gill-size: broad=b,narrow=n
# 9. gill-color: black=k,brown=n,buff=b,chocolate=h,gray=g,green=r,orange=o,pink=p,purple=u,red=e,white=w,yellow=y
# 10. stalk-shape: enlarging=e,tapering=t
# 11. stalk-root: bulbous=b,club=c,cup=u,equal=e,rhizomorphs=z,rooted=r
# 12. stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s
# 13. stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s
# 14. stalk-color-above-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
# 15. stalk-color-below-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
# 16. veil-type: partial=p,universal=u
# 17. veil-color: brown=n,orange=o,white=w,yellow=y
# 18. ring-number: none=n,one=o,two=t
# 19. ring-type: cobwebby=c,evanescent=e,flaring=f,large=l,none=n,pendant=p,sheathing=s,zone=z
# 20. spore-print-color: black=k,brown=n,buff=b,chocolate=h,green=r,orange=o,purple=u,white=w,yellow=y
# 21. population: abundant=a,clustered=c,numerous=n,scattered=s,several=v,solitary=y
# 22. habitat: grasses=g,leaves=l,meadows=m,paths=p,urban=u,waste=w,woods=d
# 
# First, we will read and explore the data using pandas and the `.read_csv` method. Pandas is an open source library providing high-performance, easy-to-use data structures and data analysis tools for the Python programming language.

# In[50]:


# load dataset
data = pd.read_csv('agaricus-lepiota.csv')


# One of the advantages of the Decision Tree algorithm is that almost no preprocessing is required. However, finding missing values is always required.

# In[51]:


display(data)


# In[52]:


#############################################################################
# TODO: Find the columns with missing values and remove them from the data.#
#############################################################################

cols_with_missing_values = [col for col in data.columns if data[col].isnull().any()]
data = data.drop(cols_with_missing_values, axis=1)
display(data)

#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################


# We will split the dataset to `Training` and `Testing` datasets.

# In[53]:


from sklearn.model_selection import train_test_split
# Making sure the last column will hold the labels
X, y = data.drop('class', axis=1), data['class']
X = np.column_stack([X,y])
# split dataset using random_state to get the same split each time
X_train, X_test = train_test_split(X, random_state=99)

print("Training dataset shape: ", X_train.shape)
print("Testing dataset shape: ", X_test.shape)


# In[54]:


y.shape


# ## Impurity Measures
# 
# Impurity is a measure of how often a randomly chosen element from the set would be incorrectly labeled if it was randomly labeled according to the distribution of labels in the subset. Implement the functions `calc_gini` and `calc_entropy`. You are encouraged to test your implementation (10 points).

# In[58]:


def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.
 
    Input:
    - data: any dataset where the last column holds the labels.
 
    Returns the gini impurity.    
    """
    
    gini = 0.0
    impurity = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    
    y = data[:,-1]
    gini = 1 - ((np.unique(y, return_counts=True)[1] / y.size) ** 2).sum()
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return gini


# In[59]:


def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns the entropy of the dataset.    
    """

    entropy = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    
    y = data[:,-1]
    P_col = np.unique(y, return_counts=True)[1] / y.size
    entropy = -1 * (P_col * np.log2(P_col)).sum()
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return entropy


# In[73]:


##### Your Tests Here #####
calc_gini(X), calc_entropy(X)


# ## Goodness of Split
# 
# Given a feature the Goodnees of Split measures the reduction in the impurity if we split the data according to the feature.
# $$
# \Delta\varphi(S, A) = \varphi(S) - \sum_{v\in Values(A)} \frac{|S_v|}{|S|}\varphi(S_v)
# $$
# 
# In our implementation the goodness_of_split function will return either the Goodness of Split or the Gain Ratio as learned in class. You'll control the return value with the `gain_ratio` parameter. If this parameter will set to False (the default value) it will return the regular Goodness of Split. If it will set to True it will return the Gain Ratio.
# $$
# GainRatio(S,A)=\frac{InformationGain(S,A)}{SplitInformation(S,A)}
# $$
# Where:
# $$
# InformationGain(S,A)=Goodness\ of\ Split\ calculated\ with\ Entropy\ as\ the\ Impurity\ function \\
# SplitInformation(S,A)=- \sum_{a\in A} \frac{|S_a|}{|S|}\log\frac{|S_a|}{|S|}
# $$
# NOTE: you can add more parameters to the function and you can also add more returning variables (The given parameters and the given returning variable should not be touch). (10 Points)

# In[74]:


def goodness_of_split(data, feature, impurity_func, gain_ratio=False):
    """
    Calculate the goodness of split of a dataset given a feature and impurity function.

    Input:
    - data: any dataset where the last column holds the labels.
    - feature: the feature index.
    - impurity func: a function that calculates the impurity.
    - gain_ratio: goodness of split or gain ratio flag.

    Returns the goodness of split (or the Gain Ration).  
    """

    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    split = 0.0
    impurity_after_split = 0
    uniques = np.unique(data[:,feature], return_counts=True)
    impurity_before_split = impurity_func(data)
    
    for i in range(len(uniques[0])):
        count = uniques[1][i]
        total = data.shape[0]
        unique_value = uniques[0][i]
        current_arr = data[data[:, feature] == unique_value] # filtered array
        impurity_after_split += (count/total) * impurity_func(current_arr)
        
        split += (count/total) * np.log2((count/total))

    goodness = impurity_before_split - impurity_after_split
    split = round((split * -1), 5)
    if gain_ratio == True:
        return goodness / split
        

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return goodness


# In[75]:


goodness_of_split(X, 21, calc_entropy)


# ## Building a Decision Tree
# 
# Use a Python class to construct the decision tree. Your class should support the following functionality:
# 
# 1. Initiating a node for a decision tree. You will need to use several class methods and class attributes and you are free to use them as you see fit. We recommend that every node will hold the feature and value used for the split and its children.
# 2. Your code should support both Gini and Entropy as impurity measures. 
# 3. The provided data includes categorical data. In this exercise, when splitting a node create the number of children needed according to the attribute unique values.
# 
# Complete the class `DecisionNode`. The structure of this class is entirely up to you. 
# 
# Complete the function `build_tree`. This function should get the training dataset and the impurity as inputs, initiate a root for the decision tree and construct the tree according to the procedure you learned in class. (30 points)

# In[76]:


class DecisionNode:
    """
    This class will hold everything you require to construct a decision tree.
    The structure of this class is up to you. However, you need to support basic 
    functionality as described above. It is highly recommended that you 
    first read and understand the entire exercise before diving into this class.
    """
    
    def __init__(self, data, feature = None, value = None , feature_val = None, label = None):
        self.feature = feature # Index of the feature beung tested
        self.value = value # Value of the node in regards to the parent feature
        self.data = data 
        self.children = []
        self.depth = 0
        self.flag = 1 # Will be in use several functions
            
    def add_child(self, node):
        self.children.append(node)
        
    def is_leaf(self):
        return (len(self.children) == 0)


# In[77]:


def find_best_feature(data, impurity, gain_ratio=False):
        '''
        Finds the best feature using the given impurity measure that has the highest information gain
        Input:
        - data: the training dataset
        - impurity: the chosen impurity measure. Notice that you can send a function
                as an argument in python.
        Output: the best feature and his values ---> (best_feature , [value1,value2])
        '''
        
        highest_info = 0
        best_feature = None
        for i in range(data.shape[1]-1):
            info_gain = goodness_of_split(data, i, calc_gini, gain_ratio)
            if (info_gain > highest_info):
                highest_info = info_gain
                best_feature = i
            
        if highest_info == 0:
            return (best_feature, None) # returns (None,None)
        
        return (best_feature, np.unique(data[:,best_feature]))


# In[78]:


def build_tree(data, impurity, gain_ratio=False, chi=1, max_depth=1000):
    """
    Build a tree using the given impurity measure and training dataset. 
    You are required to fully grow the tree until all leaves are pure. 

    Input:
    - data: the training dataset.
    - impurity: the chosen impurity measure. Notice that you can send a function
                as an argument in python.
    - gain_ratio: goodness of split or gain ratio flag
    - chi: chi square p-value cut off (1 means no pruning)
    - max_depth: the allowable depth of the tree

    Output: the root node of the tree.
    """
    import queue
    root = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    
    root = DecisionNode(np.copy(data))
    root.tree_depth = 0
    nodes_queue = queue.SimpleQueue()
    nodes_queue.put(root)
    
    #Adding label to the root
    unique_values_root = np.unique(root.data[:,-1], return_counts=True)
    max_index = unique_values_root[1].argmax()
    common_value = unique_values_root[0][max_index]
    root.label = common_value
   
    # keep the tree growing until all leaves are pure
    while(not nodes_queue.empty()):
        curr_node = nodes_queue.get()
       
            
        # stop if number of features is 1
        if curr_node.data.shape[1] == 1:
            continue
        
        # stop if pure
        elif impurity(curr_node.data) == 0: 
            continue
            
        elif curr_node.depth == max_depth:
            continue
      
        # use the best attribute function to assign best feature for the node
        best_feature_ind, best_values = find_best_feature(curr_node.data, impurity, gain_ratio)
        curr_node.feature = best_feature_ind
        
        # stop if the best feature doesn't reduce impurity
        if(best_feature_ind == None):
            continue
            
        # The values of the children    
        feature_values = np.unique(curr_node.data[:, best_feature_ind])
        
        if(chi != 1):
            chi_square = chi_square_test(curr_node)
            if chi_square < chi_table[len(feature_values) - 1][chi]:
                continue
        
        for val in feature_values:
            # initiating a child
            child = DecisionNode(curr_node.data[np.where(curr_node.data[:, best_feature_ind] == val)], value = val)
            
            # Adding label to the child
            unique_values = np.unique(child.data[:,-1], return_counts=True)
            max_index = unique_values[1].argmax()
            common_value = unique_values[0][max_index]
            child.label = common_value
            
            child.depth = curr_node.depth + 1
            root.tree_depth = max(root.tree_depth, child.depth)
            curr_node.add_child(child)
            nodes_queue.put(child)

    return root


        
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return root


# In[79]:


get_ipython().run_cell_magic('time', '', '# python supports passing a function as an argument to another function.\ntree_gini = build_tree(data=X_train, impurity=calc_gini) # gini and goodness of split\ntree_entropy = build_tree(data=X_train, impurity=calc_entropy) # entropy and goodness of split\ntree_entropy_gain_ratio = build_tree(data=X_train, impurity=calc_entropy, gain_ratio=True) # entropy and gain ratio')


# ## Tree evaluation
# 
# Complete the functions `predict` and `calc_accuracy`. (10 points)

# In[80]:


def predict(node, instance):
    """
    Predict a given instance using the decision tree
 
    Input:
    - root: the root of the decision tree.
    - instance: an row vector from the dataset. Note that the last element 
                of this vector is the label of the instance.
 
    Output: the prediction of the instance.
    """
    pred = None
    
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    
    while (node.children != None):
        found_child = False
        curr_value = instance[node.feature]
        
        if node.flag == 0: # Relevant for other valuation functions
            break
            
        for child in node.children: # Find the child with the same value
            if curr_value == child.value: # Instance value is the same as the child
                found_child = True
                new_node = child        
                node = new_node
                
        if found_child == False:
            break
                
    pred = node.label
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################    
        
    return pred


# In[81]:


def calc_accuracy(node, dataset):
    """
    Predict a given dataset using the decision tree
 
    Input:
    - node: a node in the decision tree.
    - dataset: the dataset on which the accuracy is evaluated
 
    Output: the accuracy of the decision tree on the given dataset (%).
    """
    accuracy = 0
    hits = 0
    miss = 0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################    

    accuracy = 0
    num_of_successes = 0
    for row in dataset:
        pred = predict(node, row) 
        if pred == row[-1]: # Prediction and actual label are the same
            num_of_successes += 1
    accuracy = num_of_successes / dataset.shape[0] * 100
        
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return accuracy 


# In[82]:


get_ipython().run_cell_magic('time', '', 'calc_accuracy(tree_gini ,X_train)')


# After building the three trees using the training set, you should calculate the accuracy on the test set. For each tree print the training and test accuracy. Select the tree that gave you the best test accuracy. For the rest of the exercise, use that tree (when you asked to build another tree use the same impurity function and same gain_ratio flag). 

# In[83]:


#### Your code here ####
print(calc_accuracy(tree_gini ,X_test), calc_accuracy(tree_gini ,X_train))
print(calc_accuracy(tree_entropy ,X_test), calc_accuracy(tree_entropy ,X_train))
print(calc_accuracy(tree_entropy_gain_ratio ,X_test), calc_accuracy(tree_entropy_gain_ratio ,X_train))


# ## Post pruning
# 
# Iterate over all nodes in the tree that have at least a single child which is a leaf. For each such node, replace it with its most popular class. Calculate the accuracy on the testing dataset, pick the node that results in the highest testing accuracy and permanently change it in the tree. Repeat this process until you are left with a single node in the tree (the root). Finally, create a plot of the training and testing accuracies as a function of the number of nodes in the tree. (15 points)

# In[84]:


def count_nodes(node):
    """
    Count the number of node in a given tree
 
    Input:
    - node: a node in the decision tree.
 
    Output: the number of node in the tree.
    """
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
        
    num_nodes = 1
    for child in node.children:
        num_nodes += count_nodes(child)
    return num_nodes

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    


# In[85]:


def find_nodes_to_check(curr_node, nodes_to_delete):
    
    parent_of_leaf = False
    for child in curr_node.children:
        if child.is_leaf() is False and child.flag == 1:
            nodes_to_delete = find_nodes_to_check(child, nodes_to_delete)
            continue
        elif parent_of_leaf == False:
            parent_of_leaf = True
            nodes_to_delete.put(curr_node)
        continue
    return nodes_to_delete
        
            


# In[86]:


def find_node_to_delete_Q(root, nodes_to_delete):
    node_to_delete = None
    accuracy_after_delete = 0
    
    for i in range(nodes_to_delete.qsize()):
        node = nodes_to_delete.get()
        node.flag = 0
        node_accuracy = calc_accuracy(root ,X_test)
        node.flag = 1
        if node_accuracy > accuracy_after_delete:
            accuracy_after_delete = node_accuracy
            node_to_delete = node
    return (accuracy_after_delete, node_to_delete)


# In[87]:


import queue
def post_pruning(root):
    num_of_nodes = []
    train_data_acc = []
    test_data_acc = []
    nodes_to_check = queue.SimpleQueue()
    while (root.is_leaf() is False):
        nodes_to_check = find_nodes_to_check(root, nodes_to_check)
        accr, node_to_delete = find_node_to_delete_Q(root, nodes_to_check)
        node_to_delete.children = []
        num_of_nodes.append(count_nodes(root))
        test_data_acc.append(calc_accuracy(root, X_test))
        train_data_acc.append(calc_accuracy(root, X_train))
    
    return (num_of_nodes, train_data_acc, test_data_acc)
        
    
    
        
        


# In[88]:


num_of_nodes, train_data_acc, test_data_acc = post_pruning(tree_entropy_gain_ratio)

plt.figure(figsize=(12, 10))
plt.plot(num_of_nodes, train_data_acc)
plt.plot(num_of_nodes, test_data_acc)
plt.xlabel('Nodes Count')
plt.ylabel('Accuracy')
plt.title('Accuracy as a function of the number of nodes - Train vs. Test')
plt.legend(['Train', 'Test ']);
plt.show()


# ## Chi square pre-pruning
# 
# Consider the following p-value cut-off values: [1 (no pruning), 0.5, 0.25, 0.1, 0.05, 0.0001 (max pruning)]. For each value, construct a tree and prune it according to the cut-off value. Next, calculate the training and testing accuracy. On a single plot, draw the training and testing accuracy as a function of the tuple (p-value, tree depth). Mark the best result on the graph with red circle. (15 points)

# In[93]:


### Chi square table values ###
# The first key is the degree of freedom 
# The second key is the p-value cut-off
# The values are the chi-statistic that you need to use in the pruning
chi_table = {1: {0.5 : 0.45,
                 0.25 : 1.32,
                 0.1 : 2.71,
                 0.05 : 3.84,
                 0.0001 : 100000},
             2: {0.5 : 1.39,
                 0.25 : 2.77,
                 0.1 : 4.60,
                 0.05 : 5.99,
                 0.0001 : 100000},
             3: {0.5 : 2.37,
                 0.25 : 4.11,
                 0.1 : 6.25,
                 0.05 : 7.82,
                 0.0001 : 100000},
             4: {0.5 : 3.36,
                 0.25 : 5.38,
                 0.1 : 7.78,
                 0.05 : 9.49,
                 0.0001 : 100000},
             5: {0.5 : 4.35,
                 0.25 : 6.63,
                 0.1 : 9.24,
                 0.05 : 11.07,
                 0.0001 : 100000},
             6: {0.5 : 5.35,
                 0.25 : 7.84,
                 0.1 : 10.64,
                 0.05 : 12.59,
                 0.0001 : 100000},
             7: {0.5 : 6.35,
                 0.25 : 9.04,
                 0.1 : 12.01,
                 0.05 : 14.07,
                 0.0001 : 100000},
             8: {0.5 : 7.34,
                 0.25 : 10.22,
                 0.1 : 13.36,
                 0.05 : 15.51,
                 0.0001 : 100000},
             9: {0.5 : 8.34,
                 0.25 : 11.39,
                 0.1 : 14.68,
                 0.05 : 16.92,
                 0.0001 : 100000},
             10: {0.5 : 9.34,
                  0.25 : 12.55,
                  0.1 : 15.99,
                  0.05 : 18.31,
                  0.0001 : 100000},
             11: {0.5 : 10.34,
                  0.25 : 13.7,
                  0.1 : 17.27,
                  0.05 : 19.68,
                  0.0001 : 100000}}


# In[94]:


def chi_square_test(node):
    
    chi_square = 0
    m, n = node.data.shape
    labels, count_per_label = np.unique(node.data[:, -1], return_counts = True)
    values, count_per_value = np.unique(node.data[:, node.feature], return_counts = True)
    p_0 = count_per_label[0] / m
    p_1 = count_per_label[1] / m
    
    for i, value in enumerate(values):
        d_f = count_per_value[i]
        p_f = np.count_nonzero((node.data[:, node.feature] == value) & (node.data[:, -1] == labels[0]))
        n_f = np.count_nonzero((node.data[:, node.feature] == value) & (node.data[:, -1] == labels[1]))
        E_0 = d_f * p_0
        E_1 = d_f * p_1
        
        chi_square += ((p_f - E_0)**2 / E_0) + ((n_f - E_1)**2 / E_1)
        
    return chi_square


# In[95]:


p_values = [1, 0.5, 0.25, 0.1, 0.05, 0.0001]
training_set = []
test_set = []
chi_and_depth = []
best_accuracy = 0.0
ideal_p = -1
for val in p_values:
    decision_tree = build_tree(X_train, calc_entropy, gain_ratio=False, chi = val)
    test_accuracy = calc_accuracy(decision_tree, X_test)
    test_set.append(test_accuracy)
    training_set.append(calc_accuracy(decision_tree, X_train))
    chi_and_depth.append((val, decision_tree.tree_depth))
    
    if test_accuracy >= best_accuracy:
        best_accuracy = test_accuracy
        ideal_p = val
        
plt.figure(figsize=(12, 10))
plt.xscale('log')
plt.xticks(p_values, chi_and_depth)
plt.plot(p_values, training_set)
plt.plot(p_values, test_set)
plt.scatter(ideal_p, best_accuracy, s=1000, color="white" , edgecolors="red")
plt.xlabel('Chi, Tree Depth')
plt.ylabel('Accuracy')
plt.title('Accuracy as a function of Chi value and Tree depth')
plt.legend(['Train accuracy', 'Test accuracy']);
plt.show()


# Build the best 2 trees:
# 1. tree_max_depth - the best tree according to max_depth pruning
# 1. tree_chi - the best tree according to chi square pruning

# In[96]:


tree_chi = build_tree(data=X_train, impurity=calc_entropy, chi=0.05, gain_ratio=False)


# ## Number of Nodes
# 
# Of the two trees above we will choose the one with fewer nodes. Complete the function counts_nodes and print the number of nodes in each tree. (5 points) 

# In[97]:


def count_nodes(node):
    """
    Count the number of node in a given tree
 
    Input:
    - node: a node in the decision tree.
 
    Output: the number of node in the tree.
    """
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
        
    num_nodes = 1
    for child in node.children:
        num_nodes += count_nodes(child)
    return num_nodes

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    


# In[98]:


count_nodes(tree_gini)


# ## Print the tree
# 
# Complete the function `print_tree` and execute it on your chosen tree. Your tree should be visualized clearly. You can use the following example as a reference:
# ```
# [ROOT, feature=X0],
#   [X0=a, feature=X2]
#     [X2=c, leaf]: [{1.0: 10}]
#     [X2=d, leaf]: [{0.0: 10}]
#   [X0=y, feature=X5], 
#     [X5=a, leaf]: [{1.0: 5}]
#     [X5=s, leaf]: [{0.0: 10}]
#   [X0=e, leaf]: [{0.0: 25, 1.0: 50}]
# ```
# In each brackets:
# * The first argument is the parent feature with the value that led to current node
# * The second argument is the selected feature of the current node
# * If the current node is a leaf, you need to print also the labels and their counts
# 
# (5 points)

# In[99]:


# you can change the function signeture
def print_tree(node, labels, parent_feature='ROOT'):
    '''
    prints the tree according to the example above

    Input:
    - node: a node in the decision tree

    This function has no return value
    '''
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    
    num_of_spaces = node.depth * "   "
    
    if node.value == None:
        print("[ROOT, feature={0}],".format(node.feature))
    
    # leaf     
    elif node.is_leaf():     
        count_class_0 = np.count_nonzero(node.data[:, -1] == labels[0])
        count_class_1 = np.count_nonzero(node.data[:, -1] == labels[1])
        class_0 = "{0}: {1}".format(labels[0], count_class_0) if count_class_0 != 0 else ""
        class_1 = "{0}: {1}".format(labels[1], count_class_1) if count_class_1 != 0 else ""
        comma = "" if (count_class_0 == 0 or count_class_1 == 0) else ", "
        
        print(num_of_spaces, "[X{0}={1}, leaf]: [{2}{3}{4}]".format(parent_feature, node.value, class_0, comma, class_1))
    
    # others
    else:
        print(num_of_spaces, "[X{0}={1}, feature=X{2}],".format(parent_feature, node.value, node.feature))
    
    for i in node.children:
        print_tree(i, labels, node.feature)
    
    ###########################################################################
    #                                                                         #
    ###########################################################################


# In[100]:


print_tree(tree_gini, ['p', 'e'])


# In[ ]:




