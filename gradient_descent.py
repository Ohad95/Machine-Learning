#!/usr/bin/env python
# coding: utf-8

# # Exercise 1: Linear Regression
# 
# ### This notebook is executed automatically. Failing to meet any of the submission requirements will results in a 25 point fine or your submission not being graded at all. Kindly reminder: the homework assignments grade is 50% of the final grade. 
# 
# ### Do not start the exercise until you fully understand the submission guidelines.
# 
# ## Read the following instructions carefully:
# 
# 1. This jupyter notebook contains all the step by step instructions needed for this exercise.
# 1. Submission includes this notebook only with the exercise number and your ID as the filename. For example: `hw1_123456789_987654321.ipynb` if you submitted in pairs and `hw1_123456789.ipynb` if you submitted the exercise alone.
# 1. Write **efficient vectorized** code whenever possible. Some calculations in this exercise take several minutes when implemented efficiently, and might take much longer otherwise. Unnecessary loops will result in point deduction.
# 1. You are responsible for the correctness of your code and should add as many tests as you see fit. Tests will not be graded nor checked.
# 1. Write your functions in this notebook only. **Do not create Python modules and import them**.
# 1. You are allowed to use functions and methods from the [Python Standard Library](https://docs.python.org/3/library/) and [numpy](https://www.numpy.org/devdocs/reference/) only. **Do not import anything else.**
# 1. Your code must run without errors. Make sure your `numpy` version is at least 1.15.4 and that you are using at least python 3.6. Changes of the configuration we provided are at your own risk. Any code that cannot run will not be graded.
# 1. Write your own code. Cheating will not be tolerated.
# 1. Answers to qualitative questions should be written in **markdown** cells (with $\LaTeX$ support). Answers that will be written in commented code blocks will not be checked.
# 
# ## In this exercise you will perform the following:
# 1. Load a dataset and perform basic data exploration using a powerful data science library called [pandas](https://pandas.pydata.org/pandas-docs/stable/).
# 1. Preprocess the data for linear regression.
# 1. Compute the cost and perform gradient descent in pure numpy in vectorized form.
# 1. Fit a linear regression model using a single feature.
# 1. Visualize your results using matplotlib.
# 1. Perform multivariate linear regression.
# 1. Pick the best features in the dataset.
# 1. Experiment with adaptive learning rates.

# # I have read and understood the instructions: *** 316298256 & 336319538 ***

# In[42]:


import numpy as np # used for scientific computing
import pandas as pd # used for data analysis and manipulation
import matplotlib.pyplot as plt # used for visualization and plotting

np.random.seed(42) 

# make matplotlib figures appear inline in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (14.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


# ## Part 1: Data Preprocessing (10 Points)
# 
# For the following exercise, we will use a dataset containing housing prices in King County, USA. The dataset contains 5,000 observations with 18 features and a single target value - the house price. 
# 
# First, we will read and explore the data using pandas and the `.read_csv` method. Pandas is an open source library providing high-performance, easy-to-use data structures and data analysis tools for the Python programming language.

# In[43]:


# Read comma separated data
df = pd.read_csv('data.csv') # Make sure this cell runs regardless of your absolute path.
# df stands for dataframe, which is the default format for datasets in pandas


# ### Data Exploration
# A good practice in any data-oriented project is to first try and understand the data. Fortunately, pandas is built for that purpose. Start by looking at the top of the dataset using the `df.head()` command. This will be the first indication that you read your data properly, and that the headers are correct. Next, you can use `df.describe()` to show statistics on the data and check for trends and irregularities.

# In[4]:


df.head(5)


# In[5]:


df.describe()


# We will start with one variable linear regression by extracting the target column and the `sqft_living` variable from the dataset. We use pandas and select both columns as separate variables and transform them into a numpy array.

# In[79]:


X = df['sqft_living'].values
y = df['price'].values


# ## Preprocessing
# 
# As the number of features grows, calculating gradients gets computationally expensive. We can speed this up by normalizing the input data to ensure all values are within the same range. This is especially important for datasets with high standard deviations or differences in the ranges of the attributes. Use [mean normalization](https://en.wikipedia.org/wiki/Feature_scaling) for the fearures (`X`) and the true labels (`y`).
# 
# Implement the cost function `preprocess`.

# In[45]:


def preprocess(X, y):
    """
    Perform mean normalization on the features and true labels.

    Input:
    - X: Inputs (n features over m instances).
    - y: True labels.

    Returns a two vales:
    - X: The mean normalized inputs.
    - y: The mean normalized labels.
    """    
    
    ###########################################################################
    # TODO: Implement the normalization function.                             #
    ###########################################################################
    
    X = X.astype('float64')
    y = y.astype('float64')
    
    mean_y = np.mean(y, axis=0) 
    max_y = np.max(y, axis=0)
    min_y = np.min(y, axis=0)
    y = (y-mean_y)/(max_y-min_y)
    
    if X.ndim == 1:
        mean_X = np.mean(X, axis=0) 
        max_X = np.max(X, axis=0)
        min_X = np.min(X, axis=0)
        X = (X-mean_X)/(max_X-min_X)
        
    else:
        for i in range(X.shape[1]):
            mean_X = np.mean(X[:,i], axis=0) 
            max_X = np.max(X[:,i], axis=0)
            min_X = np.min(X[:,i], axis=0)
            X[:,i] = (X[:,i]-mean_X)/(max_X-min_X)
            
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return X, y


# In[80]:


X, y= preprocess(X, y)


# We will split the data into two datasets: 
# 1. The training dataset will contain 80% of the data and will always be used for model training.
# 2. The validation dataset will contain the remaining 20% of the data and will be used for model evaluation. For example, we will pick the best alpha and the best features using the validation dataset, while still training the model using the training dataset.

# In[81]:


# training and validation split
np.random.seed(42)
indices = np.random.permutation(X.shape[0])
idx_train, idx_val = indices[:int(0.8*X.shape[0])], indices[int(0.8*X.shape[0]):]
X_train, X_val = X[idx_train], X[idx_val]
y_train, y_val = y[idx_train], y[idx_val]


# In[82]:


X_train


# ## Data Visualization
# Another useful tool is data visualization. Since this problem has only two parameters, it is possible to create a two-dimensional scatter plot to visualize the data. Note that many real-world datasets are highly dimensional and cannot be visualized naively. We will be using `matplotlib` for all data visualization purposes since it offers a wide range of visualization tools and is easy to use.

# In[48]:


plt.plot(X_train, y_train, 'ro', ms=1, mec='k') # the parameters control the size, shape and color of the scatter plot
plt.ylabel('Price in USD')
plt.xlabel('sq.ft')
plt.show()


# ## Bias Trick
# 
# Make sure that `X` takes into consideration the bias $\theta_0$ in the linear model. Hint, recall that the predications of our linear model are of the form:
# 
# $$
# \hat{y} = h_\theta(x) = \theta^T x = \theta_0 + \theta_1 x_1
# $$
# 
# Add columns of ones as the zeroth column of the features (do this for both the training and validation sets).

# In[83]:


###########################################################################
#                            START OF YOUR CODE                           #
###########################################################################

def add_bias(X):
    num_of_rows = X.shape[0]
    if X.ndim == 1: # In case X is a vector
        X = X.reshape((num_of_rows, 1))
    
    new_X = np.ones((num_of_rows, X.shape[1]+1))
    new_X[:,1:] = X
    
#     bias_column = np.ones(num_of_rows).reshape((num_of_rows, 1))
#     X = np.append(bias_column, X, axis=1)
    return new_X


###########################################################################
#                             END OF YOUR CODE                            #
###########################################################################


# In[84]:


X_train = add_bias(X_train)
X_val = add_bias(X_val)


# ## Part 2: Single Variable Linear Regression (40 Points)
# Simple linear regression is a linear regression model with a single explanatory varaible and a single target value. 
# 
# $$
# \hat{y} = h_\theta(x) = \theta^T x = \theta_0 + \theta_1 x_1
# $$
# 
# ## Gradient Descent 
# 
# Our task is to find the best possible linear line that explains all the points in our dataset. We start by guessing initial values for the linear regression parameters $\theta$ and updating the values using gradient descent. 
# 
# The objective of linear regression is to minimize the cost function $J$:
# 
# $$
# J(\theta) = \frac{1}{2m} \sum_{i=1}^{n}(h_\theta(x^{(i)})-y^{(i)})^2
# $$
# 
# where the hypothesis (model) $h_\theta(x)$ is given by a **linear** model:
# 
# $$
# h_\theta(x) = \theta^T x = \theta_0 + \theta_1 x_1
# $$
# 
# $\theta_j$ are parameters of your model. and by changing those values accordingly you will be able to lower the cost function $J(\theta)$. One way to accopmlish this is to use gradient descent:
# 
# $$
# \theta_j = \theta_j - \alpha \frac{1}{m} \sum_{i=1}^m (h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}
# $$
# 
# In linear regresion, we know that with each step of gradient descent, the parameters $\theta_j$ get closer to the optimal values that will achieve the lowest cost $J(\theta)$.

# Implement the cost function `compute_cost`. (10 points)

# In[87]:


def compute_cost(X, y, theta):
    """
    Computes the average squared difference between an obserbation's actual and
    predicted values for linear regression.  

    Input:
    - X: inputs  (n features over m instances).
    - y: true labels (1 value over m instances).
    - theta: the parameters (weights) of the model being learned.

    Returns a single value:
    - J: the cost associated with the current set of parameters (single number).
    """
    ###########################################################################
    # TODO: Implement the MSE cost function.                                  #
    ###########################################################################
    
    h_theta = np.matmul(X, theta) # 𝜃_0 + 𝜃_1*𝑥_1 ...
    sum_of_error = sum((h_theta - y) ** 2)
    
    J = sum_of_error/(2 * X.shape[0])  # Use J for the cost.

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return J


# In[88]:


theta = np.array([-1, 2])
J = compute_cost(X_train, y_train, theta)


# In[89]:


print(J)


# Implement the gradient descent function `gradient_descent`. (10 points)

# In[90]:


def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of the model using gradient descent using 
    the *training set*. Gradient descent is an optimization algorithm 
    used to minimize some (loss) function by iteratively moving in 
    the direction of steepest descent as defined by the negative of 
    the gradient. We use gradient descent to update the parameters
    (weights) of our model.

    Input:
    - X: Inputs  (n features over m instances).
    - y: True labels (1 value over m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns two values:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    
    J_history = [] # Use a python list to save cost in every iteration
    theta = theta.copy() # avoid changing the original thetas
    
    ###########################################################################
    # TODO: Implement the gradient descent optimization algorithm.            #
    ###########################################################################
        
    for j in range(num_iters):
        predictions = np.dot(X, theta)
        theta = theta - alpha * (1/X.shape[0]) * np.dot(np.transpose(X), predictions - y) # Updating theta
        J_current = compute_cost(X, y, theta)
        J_history.append(J_current)
        
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return theta, J_history


# In[91]:


np.random.seed(42)
theta = np.random.random(size=2)
iterations = 40000
alpha = 0.1
theta, J_history = gradient_descent(X_train ,y_train, theta, alpha, iterations)


# In[92]:


print(J_history[-1])


# You can evaluate the learning process by monitoring the loss as training progress. In the following graph, we visualize the loss as a function of the iterations. This is possible since we are saving the loss value at every iteration in the `J_history` array. This visualization might help you find problems with your code. Notice that since the network converges quickly, we are using logarithmic scale for the number of iterations. 

# In[19]:


plt.plot(np.arange(iterations), J_history)
plt.xscale('log')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss as a function of iterations')
plt.show()


# Implement the pseudo-inverse function `pinv`. **Do not use `np.linalg.pinv`**, instead use only direct matrix multiplication as you saw in class (you can calculate the inverse of a matrix using `np.linalg.inv`). (10 points)

# In[93]:


def pinv(X, y):
    """
    Calculate the optimal values of the parameters using the pseudoinverse
    approach as you saw in class using the *training set*.

    Input:
    - X: Inputs  (n features over m instances).
    - y: True labels (1 value over m instances).

    Returns two values:
    - theta: The optimal parameters of your model.

    ########## DO NOT USE np.linalg.pinv ##############
    """
    
    pinv_theta = []

    ###########################################################################
    # TODO: Implement the pseudoinverse algorithm.                            #
    ###########################################################################
    
    newX = np.linalg.inv(np.dot(np.transpose(X),X))
    newX = np.dot(newX ,np.transpose(X))
    pinv_theta = np.matmul(newX,y)
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pinv_theta


# In[21]:


theta_pinv = pinv(X_train ,y_train)
J_pinv = compute_cost(X_train, y_train, theta_pinv)


# In[22]:


print(J_pinv)


# We can add the loss value for the theta calculated using the psuedo-inverse to our graph. This is another sanity check as the loss of our model should converge to the psuedo-inverse loss.

# In[23]:


plt.plot(np.arange(iterations), J_history)
plt.xscale('log')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss as a function of iterations')
plt.hlines(y = J_pinv, xmin = 0, xmax = len(J_history), color='r',
           linewidth = 1, linestyle = 'dashed')
plt.show()


# We can use a better approach for the implementation of `gradient_descent`. Instead of performing 40,000 iterations, we wish to stop when the improvement of the loss value is smaller than `1e-8` from one iteration to the next. Implement the function `efficient_gradient_descent`. (5 points)

# In[94]:


def efficient_gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of your model using the *training set*, but stop 
    the learning process once the improvement of the loss value is smaller 
    than 1e-8. This function is very similar to the gradient descent 
    function you already implemented.

    Input:
    - X: Inputs  (n features over m instances).
    - y: True labels (1 value over m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns two values:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    
    J_history = [] # Use a python list to save cost in every iteration
    theta = theta.copy() # avoid changing the original thetas
    
            
    ###########################################################################
    # TODO: Implement the gradient descent optimization algorithm.            #
    ###########################################################################
    
    for j in range(num_iters):
        predictions = np.dot(X, theta)
        theta = theta - alpha * (1/X.shape[0]) * np.dot(np.transpose(X), predictions - y) # Updating theta
        J_current = compute_cost(X, y, theta)
        J_history.append(J_current)
        if (j > 1 and (J_history[j-1] - J_history[j] < 1e-8)):    # Check if the change in J is smaller then the threshold
            return theta, J_history
                    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return theta, J_history


# In[25]:


np.random.seed(42)
theta = np.random.random(size=2)
iterations = 40000
alpha = 0.1
theta, J_history = efficient_gradient_descent(X_train ,y_train, theta, alpha, iterations)


# In[26]:


print(theta)
print(J_history[-1])


# The learning rate is another factor that determines the performance of our model in terms of speed and accuracy. Complete the function `find_best_alpha`. Make sure you use the training dataset to learn the parameters (thetas) and use those parameters with the validation dataset to compute the cost.

# In[95]:


def find_best_alpha(X_train, y_train, X_val, y_val, iterations):
    """
    Iterate over provided values of alpha and train a model using the 
    *training* dataset. maintain a python dictionary with alpha as the 
    key and the loss on the *validation* set as the value.

    Input:
    - X_train, y_train, X_val, y_val: the training and validation data
    - iterations: maximum number of iterations

    Returns:
    - alpha_dict: A python dictionary - {key (alpha) : value (validation loss)}
    """

    alphas = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2]
    alpha_dict = {}
    
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    
    np.random.seed(42)
    theta0 = np.random.random(size=2)
    for alpha in alphas:
        updated_theta = efficient_gradient_descent(X_train, y_train, theta0, alpha, iterations)[0]
        alpha_dict[alpha] = compute_cost(X_val, y_val, updated_theta)
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return alpha_dict


# In[29]:


alpha_dict = find_best_alpha(X_train, y_train, X_val, y_val, 40000)


# In[30]:


alpha_dict


# Obtain the best learning rate from the dictionary `alpha_dict`. This can be done in a single line using built-in functions.

# In[31]:


best_alpha = min(alpha_dict, key=alpha_dict.get)
best_alpha


# Pick the best three alpha values you just calculated and provide **one** graph with three lines indicating the training loss as a function of iterations (Use 10,000 iterations). Note you are required to provide general code for this purpose (no hard-coding). Make sure the visualization is clear and informative. (5 points)

# In[32]:


np.random.seed(42)
theta0 = np.random.random(size=2)
iterations = 40000
top_3_alphas = sorted(alpha_dict, key=alpha_dict.get)[:3] # Getting 3 best alphas
for alpha in top_3_alphas:
    theta, J_history = gradient_descent(X_train ,y_train, theta0, alpha, iterations)
    plt.plot(np.arange(iterations), J_history, label = alpha)
plt.xscale('log')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss as a function of iterations')
legend = plt.legend(shadow=True, fontsize='x-large')
plt.show()


# This is yet another sanity check. This function plots the regression lines of your model and the model based on the pseudoinverse calculation. Both models should exhibit the same trend through the data. 

# In[33]:


plt.figure(figsize=(7, 7))
plt.plot(X_train[:,1], y_train, 'ro', ms=1, mec='k')
plt.ylabel('Price in USD')
plt.xlabel('sq.ft')
plt.plot(X_train[:, 1], np.dot(X_train, theta), 'o')
plt.plot(X_train[:, 1], np.dot(X_train, theta_pinv), '-')

plt.legend(['Training data', 'Linear regression', 'Best theta']);


# ## Part 2: Multivariate Linear Regression (30 points)
# 
# In most cases, you will deal with databases that have more than one feature. It can be as little as two features and up to thousands of features. In those cases, we use a multiple linear regression model. The regression equation is almost the same as the simple linear regression equation:
# 
# $$
# \hat{y} = h_\theta(\vec{x}) = \theta^T \vec{x} = \theta_0 + \theta_1 x_1 + ... + \theta_n x_n
# $$
# 
# 
# If you wrote vectorized code, this part should be straightforward. If your code is not vectorized, you should go back and edit your functions such that they support both multivariate and single variable regression. **Your code should not check the dimensionality of the input before running**.

# In[96]:


# Read comma separated data
df = pd.read_csv('data.csv')
df.head()


# ## Preprocessing
# 
# Like in the single variable case, we need to create a numpy array from the dataframe. Before doing so, we should notice that some of the features are clearly irrelevant.

# In[97]:


X = df.drop(columns=['price', 'id', 'date']).values
y = df['price'].values


# Use the **same** `preprocess` function you implemented previously. Notice that proper vectorized implementation should work regardless of the dimensionality of the input. You might want to check that your code in the previous parts still works.

# In[98]:


# preprocessing
X, y = preprocess(X, y)


# In[99]:


# training and validation split
np.random.seed(42)
indices = np.random.permutation(X.shape[0])
idx_train, idx_val = indices[:int(0.8*X.shape[0])], indices[int(0.8*X.shape[0]):]
X_train, X_val = X[idx_train,:], X[idx_val,:]
y_train, y_val = y[idx_train], y[idx_val]


# Using 3D visualization, we can still observe trends in the data. Visualizing additional dimensions requires advanced techniques we will learn later in the course.

# In[100]:


get_ipython().run_line_magic('matplotlib', 'inline')
import mpl_toolkits.mplot3d.axes3d as p3
fig = plt.figure(figsize=(5,5))
ax = p3.Axes3D(fig)
xx = X_train[:, 1][:1000]
yy = X_train[:, 2][:1000]
zz = y_train[:1000]
ax.scatter(xx, yy, zz, marker='o')
ax.set_xlabel('bathrooms')
ax.set_ylabel('sqft_living')
ax.set_zlabel('price')
plt.show()


# Use the bias trick again (add a column of ones as the zeroth column in the both the training and validation datasets).

# In[101]:


###########################################################################
#                            START OF YOUR CODE                           #
###########################################################################

X_train = add_bias(X_train)
X_val = add_bias(X_val)

###########################################################################
#                             END OF YOUR CODE                            #
###########################################################################


# Make sure the functions `compute_cost` (10 points), `gradient_descent` (15 points), and `pinv` (5 points) work on the multi-dimensional dataset. If you make any changes, make sure your code still works on the single variable regression model. 

# In[102]:


shape = X_train.shape[1]
theta = np.ones(shape)
J = compute_cost(X_train, y_train, theta)
print(J)


# In[104]:


np.random.seed(42)
shape = X_train.shape[1]
theta = np.random.random(shape)
iterations = 40000
theta, J_history = gradient_descent(X_train ,y_train, theta, best_alpha, iterations)


# In[105]:


print(J_history[-1])


# In[106]:


theta_pinv = pinv(X_train ,y_train)
J_pinv = compute_cost(X_train, y_train, theta_pinv)


# In[107]:


print(J_pinv)


# We can use visualization to make sure the code works well. Notice we use logarithmic scale for the number of iterations, since gradient descent converges after ~500 iterations.

# In[108]:


plt.plot(np.arange(iterations), J_history)
plt.xscale('log')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss as a function of iterations - multivariate linear regression')
plt.hlines(y = J_pinv, xmin = 0, xmax = len(J_history), color='r',
           linewidth = 1, linestyle = 'dashed')
plt.show()


# ## Part 3: Find best features for regression (10 points)
# 
# Adding additional features to our regression model makes it more complicated but does not necessarily improves performance.
# Use forward and backward selection and find 4 features that best minimizes the loss. First, we will reload the dataset as a dataframe in order to access the feature names.

# In[19]:


columns_to_drop = ['price', 'id', 'date']
all_features = df.drop(columns=columns_to_drop)
all_features.head(5)


# In[20]:


X = all_features.values
y = df['price'].values
X, y = preprocess(X, y)
X = add_bias(X)


# In[21]:


# training and validation split
np.random.seed(42)
indices = np.random.permutation(X.shape[0])
idx_train, idx_val = indices[:int(0.8*X.shape[0])], indices[int(0.8*X.shape[0]):]
X_train, X_val = X[idx_train], X[idx_val]
y_train, y_val = y[idx_train], y[idx_val]


# ### Forward Feature Selection
# 
# Complete the function `forward_selection`. Train the model using a single feature at a time, and choose the best feature using the validation dataset. Next, check which feature performs best when added to the feature you previously chose. Repeat this process until you reach 4 features + bias. You are free to use any arguments you need.

# In[37]:


def forward_selection(X_train, y_train, X_val, y_val, num_iters):
    """
    Train the model using the training set using a single feature. 
    Choose the best feature according to the validation set. Next, 
    check which feature performs best when added to the feature
    you previously chose. Repeat this process until you reach 4 
    features and the bias. Don't forget the bias trick.

    Returns:
    - The names of the best features using forward selection.
    """
    #gradient_descent(X, y, theta, alpha, num_iters) --> (theta, J_history)
    #compute_cost(X, y, theta) --> J
    
    np.random.seed(42)
    best_features = []
    
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    
    feature_indexes = [0]
    feature_names = all_features.columns
    
    for j in range(4):
        initial_theta = np.random.random(size=2+j)
        lowest_cost = np.inf
        
        for i in range(1, X_train.shape[1]):
            if i in feature_indexes: continue
            feature_name = feature_names[i-1] #The current feature
            current_indexs = feature_indexes + [i] #Add the feature index to the current features we test
            
            theta = efficient_gradient_descent(X_train[:,current_indexs], y_train, initial_theta, 1, num_iters)[0]
            current_J = compute_cost(X_val[:,current_indexs], y_val, theta)
            
            if current_J < lowest_cost: # Then the feature is currently the best
                lowest_cost = current_J
                current_best_feature_details = [feature_name, i]

                
        best_features.append(current_best_feature_details[0])
        feature_indexes.append(current_best_feature_details[1])       

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return best_features


# In[38]:


forward_selection(X_train, y_train, X_val, y_val, 1000)


# ### Backward Feature Selection
# 
# Complete the function `backward_selection`. Train the model with all but one of the features at a time and remove the worst feature (the feature that its absence yields the best loss value using the validation dataset). Next, remove an additional feature along with the feature you previously removed. Repeat this process until you reach 4 features + bias. You are free to use any arguments you need.

# In[39]:


def backward_selection(X_train, y_train, X_val, y_val, best_alpha, num_iters):
    """
    Train the model using the training set using all but one of the 
    features at a time. Remove the worst feature according to the 
    validation set. Next, remove an additional feature along with the 
    feature you previously removed. Repeat this process until you 
    reach 4 features and the bias. Don't forget the bias trick.

    Returns:
    - The names of the best features using backward selection.
    """
    np.random.seed(42)
    best_features = None
    
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    
    worst_features = []
    all_feature_names = all_features.columns
    feature_indexes = [i for i in range(1,len(all_feature_names)+1)] #List of indexes starting from one to the number of the features
    
    for j in range(len(all_feature_names)-4): #loop of the number of the features - 4 because we delete a feature in each round and want to remain with 4
        highest_cost = -np.inf
        for i in range(1, X_train.shape[1]):
            feature_name = all_feature_names[i-1]
            if feature_name in worst_features: continue
                
            current_indexs = [index for index in feature_indexes if index != i] # List of the indexes except i
            initial_theta = np.random.random(size=len(current_indexs)) # random theta in the relevant size
            theta = efficient_gradient_descent(X_train[:,current_indexs], y_train, initial_theta, 1, num_iters)[0] # theta which minimzes the cost function
            current_J = compute_cost(X_val[:,current_indexs], y_val, theta) # the minimized cost function
            
            if current_J > highest_cost: # Then the feature is currently the worse
                current_worst_feature_details = [feature_name, i]
                max_cost = current_J
                
        worst_features.append(current_worst_feature_details[0])
        feature_indexes.remove(current_worst_feature_details[1])   
    
    best_features = set(all_feature_names) - set(worst_features) # The best feature = All the features - the worst features
        
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return list(best_features)


# In[41]:


backward_selection(X_train, y_train, X_val, y_val, best_alpha, 200)


# Give an explanations to the results. Do they make sense?

# ### Use this Markdown cell for your answer
# 
# We got different features from the backward and forward selection. 
# It make sense to some extent, for example:
# If there are 3 variables and we want to select 2, and if 2 of the features share high correlation and also have a big effect on the target label - in the forward selection they won't get chosen together but in the backwards they might.
# 
# We are not sure if that is the case in our dataset.
# However, since we have only one shared feature in the forward and backward selection, it makes us wonder about the correctness of our implementaion and the efficiency of the forward/backwards selections.

# ## Part 4: Adaptive Learning Rate (10 points)
# 
# So far, we kept the learning rate alpha constant during training. However, changing alpha during training might improve convergence in terms of the global minimum found and running time. Implement the adaptive learning rate method based on the gradient descent algorithm above. 
# 
# **Your task is to find proper hyper-parameter values for the adaptive technique and compare this technique to the constant learning rate. Use clear visualizations of the validation loss and the learning rate as a function of the iteration**. 
# 
# Time based decay: this method reduces the learning rate every iteration according to the following formula:
# 
# $$\alpha = \frac{\alpha_0}{1 + D \cdot t}$$
# 
# Where $\alpha_0$ is the original learning rate, $D$ is a decay factor and $t$ is the current iteration.

# In[54]:


def adaptive(X_train, y_train, X_val, y_val, theta, alpha, d, num_iters):
    """
    Performs gradient descent with an adaptive alpha which depends on a decay factor 'd'
    """
    J_history = [] # Use a python list to save cost in every iteration
    theta = theta.copy() # avoid changing the original thetas
        
    for j in range(num_iters):
        new_alpha = alpha / (1 + (d * j))
        predictions = np.dot(X, theta)
        theta = theta - new_alpha * (1/X.shape[0]) * np.dot(np.transpose(X), predictions - y) # Updating theta
        J_current = compute_cost(X_val, y_val, theta)
        J_history.append(J_current)
    
    return theta, J_history


# In[55]:


np.random.seed(42)
iterations = 4000
array_of_d = [0, 0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3]
d_dict = {}


# In[58]:


initial_theta = np.random.random(size=X_train.shape[1])
for d in array_of_d:
    theta, J_history = adaptive(X_train, y_train, X_val, y_val, initial_theta, best_alpha, d, iterations)
    d_dict[d] = (theta, J_history)


# In[59]:


for d in d_dict:
    plt.plot(np.arange(iterations), d_dict[d][1], label = d)
plt.xscale('log')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss as a function of iterations')
legend = plt.legend(shadow=True, loc = 'upper right')
plt.show()

