# Random Forest Classifier

Implemented a simple Random Forest Classifier to predict the likelihood of a patient falling into a category, whether it has low or high risk of having a heart attack. First implementation works with its default values for each hyperparameter, since we want to differenciate the impact some hyperparameters have over the prediction. The second implementation works with some modified hyperparameters to obtain a better performance in the model. In both cases, the proportion for the train and test data is about 80% for training and 20% for validation, even though the train data for the second model was scaled to obtain better results. After making some predictions, an accuracy score is calculated for each model, we can see differences in the performance of each model, since we're working with a Random Forest Classifier, we'll not always have the same accuracy, and the performance is evaluated with many runs.

## Hyperparameters used for the Second Model
1. n_estimators: The number of trees generated for the model, the defaul, 100 trees, were used.
2. max_features: How the features are splitted for the trees, the logarithm base 2 is used for this model.
3. max_depth: The longest distance from the root node and the last leaf node, 4 is being used.
4. random_state: Used to reduce overfitting and control randomness among runs of the algorithm.
5. max_leaf_nodes: Controls the number of leafs in each tree, it's set to 10.
6. n_jobs: Set to -1 means it's using all the processors to run jobs in parallel.

# SMA0401A
## Uses a framework to implement a technique or algorithm for Machine Learning, like regressions, trees, clusters, etc...
The Random Forest Classifier Algorithm is implemented with a framework from Scikit-Learn, the classifier works correctly and predicts a classification for certain input variables. This performance can be observed with the plots for the true values and the predicted values on each model, the accuracy scores and the Mean Squared Error when executing the algorithms.

# Files
## Dataset
The Dataset used for this implementation is 'heart.csv', which can be found in this GitHub Repository. In order to change the Dataset there are some modifications that need to be done in the script before executing it; such as changing the loading file for the DataFrame and changing the columns name from the DataFrame that are going to be used.
## random_forest.py
This is the file that needs to be executed to train the models and validate them.
