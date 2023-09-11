'''
    Random Forest Classifier using Scikit-Learn framework

    Hilda Beltrán Acosta
    A01251916
'''

# Import necessary libraries to implement the Random Forest Classifier
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
import matplotlib.pyplot as plt

# Load data into DataFrame
columns = ['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh', 'exng', 'oldpeak', 'slp', 'caa', 'thall', 'output']
df = pd.read_csv('heart.csv', names = columns)

# Random order for the DataFrame rows, since most of the biggest values of y are at the end
df1 = df.sample(frac = 1)

# Divide between inputs and outputs
x = df1.drop(columns = 'output').to_numpy()
y = df1[['output']].to_numpy()


'''
    Function forest splits the inputs and outputs into train and test.
    This splitting occurs as 80% training and 20% testing.
    The Random Forest Classifier is generated with the default hyperparameters, 
    the 100 trees is the default value for the estimators.

    There's calculations of Accuracy and Mean Squared Error, in order to evaluate
    the bias and variance of the model, real and predicted output are plotted.

    @params x, y
    @return fitted model
'''
def forest(x, y):
    #split train and test data and targets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20)

    forest = RandomForestClassifier(n_estimators=100)
    forest.fit(x_train, y_train.ravel())

    y_pred = forest.predict(x_test)

    print('\nAccuracy Score for Random Forest with default hyperparameters:')
    print(accuracy_score(y_test, y_pred))

    print('\nMean Squared Error for Random Forest with default hyperparameters:')
    print(mean_squared_error(y_test, y_pred))

    plt.scatter(range(len(y_test)), y_test, color='red')
    plt.scatter(range(len(y_pred)), y_pred, color='blue')

    plt.xlabel('Output Position')
    plt.ylabel('Output')
    plt.title('Real vs Predicted Values for Output Default')

    plt.show()

    return forest


'''
    Function forest2 scales the input data, then splits inputs and
    outputs into train and test. This splitting occurs as 80% training 
    and 20% testing.

    The Random Forest Classifier is generated with 100 trees, the features 
    are splitted into the logarithm base 2 of the total features. The max
    depth parameter is set to 4, since the longest distance from the root 
    and the last leaf node we want is 4. We use a random state value, which
    is used to reduce overfitting and controls randomness among runs of the
    algorithm. We set the max leaf nodes parameter to 10, which controls
    the number of leafs in every individual tree. And the n_jobs as -1 means
    it's using all the processors to run many jobs in parallel.

    There's calculations of Accuracy and Mean Squared Error, in order to evaluate
    the bias and variance of the model, real and predicted output are plotted.

    @params x, y
    @return fitted model
'''
def forest2(x, y):
    #scale features
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    #split train and test data and targets
    x_train1, x_test1, y_train1, y_test1 = train_test_split(x_scaled, y, test_size = 0.20)

    forest1 = RandomForestClassifier(n_estimators=100, max_features = 'log2', max_depth = 4, random_state = 42, max_leaf_nodes=10, n_jobs=-1)
    forest1.fit(x_train1, y_train1.ravel())

    y_pred1 = forest1.predict(x_test1)

    print('\nAccuracy Score for Random Forest with modified hyperparameters:')
    print(accuracy_score(y_test1, y_pred1))

    print('\nMean Squared Error for Random Forest with modified hyperparameters:')
    print(mean_squared_error(y_test1, y_pred1))

    plt.scatter(range(len(y_test1)), y_test1, color='red')
    plt.scatter(range(len(y_pred1)), y_pred1, color='blue')

    plt.xlabel('Output Position')
    plt.ylabel('Output')
    plt.title('Real vs Predicted Values for Output Modified')

    plt.show()

    return forest1

# Run the algorithm using the default parameters
forest1_pred = forest(x, y)

# Run the algorithm using the modified parameters
forest2_pred = forest2(x, y)

x_pred1 = [[62, 1, 0, 120, 267, 0, 1, 99, 1, 1.8, 1, 2, 3]]
#x_pred1 = [[-0.81242462, 0.68100522, -0.93851463, -1.12077005, -0.81677269, -0.41763453, 0.89896224, -0.29067075, -0.69663055, -0.81059216, 0.97635214, -0.71442887, -0.51292188]]

#Make predictions
print('\nRandom Forest Classifier with default hyperparameters')
print('\n Input:')
print(x_pred1)
probs1 = forest1_pred.predict_proba(x_pred1)
print('\nProbability for each class for this input:')
print('\n')
print(probs1)

pred1 =  forest1_pred.predict(x_pred1)
print('Prediction:')
print(pred1)

print('\nRandom Forest Classifier with modified hyperparameters')

probs2 = forest2_pred.predict_proba(x_pred1)
print('\nProbability for each class for this input:')
print(probs2)

pred2 =  forest2_pred.predict(x_pred1)
print('Prediction:')
print(pred2)