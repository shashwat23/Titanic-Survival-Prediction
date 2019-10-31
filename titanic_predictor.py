#***********************************************#

#Author - Shashwat Gaur (. Email - shashwatgaur23@gmail.com
#Written in Python v3.4

#***********************************************#

#importing all relavent packages
import pandas as pd
import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt
import math
#import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from pandas.tools.plotting import scatter_matrix
from sklearn import preprocessing
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingClassifier

train = pd.read_csv('train.csv')	#location needs to be given from the folder the code is presently saved
#taking a quick glance of data
print(train.head())


#converting text containing columns into numbers, for easy processing and visualisation
#also imputing all empty values with mode (of that field) for discrete variables and with median for continuous values
train["Embarked"][train["Embarked"] == "S"] = 0
train["Embarked"][train["Embarked"] == "C"] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2
train["Embarked"][train["Survived"] == 0] = train["Embarked"][train["Survived"] == 0].fillna(train["Embarked"][train["Survived"] == 0].mode())
train["Embarked"][train["Survived"] == 1] = train["Embarked"][train["Survived"] == 1].fillna(train["Embarked"][train["Survived"] == 1].mode())


train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] = 1
train["Sex"][train["Survived"] == 0] = train["Sex"][train["Survived"] == 0].fillna(train["Sex"][train["Survived"] == 0].mode())
train["Sex"][train["Survived"] == 1] = train["Sex"][train["Survived"] == 1].fillna(train["Sex"][train["Survived"] == 1].mode())


train["Age"][train["Survived"] == 0] = train["Age"][train["Survived"] == 0].fillna(train["Age"][train["Survived"] == 0].median())
train["Age"][train["Survived"] == 1] = train["Age"][train["Survived"] == 1].fillna(train["Age"][train["Survived"] == 1].median())


train["Pclass"][train["Survived"] == 0] = train["Pclass"][train["Survived"] == 0].fillna(train["Pclass"][train["Survived"] == 0].mode())
train["Pclass"][train["Survived"] == 1] = train["Pclass"][train["Survived"] == 1].fillna(train["Pclass"][train["Survived"] == 1].mode())


train["Fare"][train["Survived"] == 0] = train["Fare"][train["Survived"] == 0].fillna(train["Fare"][train["Survived"] == 0].median())
train["Fare"][train["Survived"] == 1] = train["Fare"][train["Survived"] == 1].fillna(train["Fare"][train["Survived"] == 1].median())


train["SibSp"][train["Survived"] == 1] = train["SibSp"][train["Survived"] == 1].fillna(train["SibSp"][train["Survived"] == 1].mode())
train["SibSp"][train["Survived"] == 0] = train["SibSp"][train["Survived"] == 0].fillna(train["SibSp"][train["Survived"] == 0].mode())
train["Parch"][train["Survived"] == 1] = train["Parch"][train["Survived"] == 1].fillna(train["Parch"][train["Survived"] == 1].mode())
train["Parch"][train["Survived"] == 0] = train["Parch"][train["Survived"] == 0].fillna(train["Parch"][train["Survived"] == 0].mode())

#creating new coulumn "Relatives" containing sum of nymber of Siblings/Spouse/Parent or Children a person has onboard
train["Relatives"] = train["SibSp"] + train["Parch"]
train["Relatives"][train["Survived"] == 1] = train["Relatives"][train["Survived"] == 1].fillna(train["Relatives"][train["Survived"] == 1].mode())
train["Relatives"][train["Survived"] == 0] = train["Relatives"][train["Survived"] == 0].fillna(train["Relatives"][train["Survived"] == 0].mode())

#plotting scatter plot to get an idea of correlation within varaibles
numeric_cols = train[["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked","Relatives"]]
_ = scatter_matrix(numeric_cols, c = train["Survived"] ,alpha = 0.2, figsize=(8,8), diagonal = 'hist')
plt.show()


#converting "Age" into a discrete variable for decision tree functioning. Age of 16 was taken as all people below age 16 (children) had
#higher chances of getting saved as observed from training data.
train["Age"][train["Age"] < 16] = 0
train["Age"][train["Age"] >= 16][ train["Age"] < 60] = 1
train["Age"][train["Age"] >= 60] = 2

#taking log of Fare column as it has a very long range. This discretizes into only 3 values - 0,1 and 2.
train["Fare"] = np.log10(train["Fare"]+1).astype(int)


#plotting scatter plot again to see effect of above discretizations
numeric_cols = train[["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked","Relatives"]]
_ = scatter_matrix(numeric_cols, c = train["Survived"] ,alpha = 0.2, figsize=(8,8), diagonal = 'hist')
plt.show()

#list of all columns which will be a part of training. Some columns have been dropped due to lots of missing values,
# and others due to lack of correlation with survival.
#feature_table = train[["Pclass","Sex","Age","Fare","Relatives"]]
feature_table = train[["Pclass","Sex","Age","Fare","SibSp","Parch"]]
#standardisation for algorithms which need to calculate distance (so that all features are given equal weights)
ft = preprocessing.StandardScaler().fit(feature_table)
#training data is stored in 'feature_table' and output in 'target_values'
feature_table = ft.transform(feature_table)
target_values = train["Survived"].values


#spot checking of machine learning algorithms begins

"""
#all models are stored in list 'models'
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('DTC', DecisionTreeClassifier(max_depth = 6, min_samples_split = 4)))
models.append(('SVC', SVC()))
models.append(('RFC', RandomForestClassifier(n_estimators=100)))
models.append(('KNC', KNeighborsClassifier(n_neighbors = 3)))
models.append(('MLP', MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
       beta_1=0.9, beta_2=0.999, early_stopping=False,
       epsilon=1e-08, hidden_layer_sizes=(10, 5), learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False)))
# evaluate each model in turn

results = []
names = []
scoring = 'accuracy'

#all models are tested for accuracy and their performace is printed
for name, model in models:
	kfold = KFold(n_splits=10, random_state=7)
	cv_results = cross_val_score(model, feature_table, target_values, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
"""
#spot checking ends

#now all good performing algorithms will be tuned and tested one by one

#mlp begins
"""
h_layer_sizes = []
for i in range(4,16):
       for j in range(4,16):
              for k in range(4,16):
                    h_layer_sizes.append((i,j,k))

#hyperparameter space to be tuned is specified
params_nn = {"activation":["relu","logistic","tanh"], "hidden_layer_sizes": h_layer_sizes, "solver" : ["lbfgs", "sgd", "adam"], "alpha" : 
             [0.0001,0.00005,0.00001], "tol": [0.0001]}
obj = MLPClassifier(max_iter = 1000)
#as there are lots of hyperparameters to be searched from, a RandomizedSearch is performed to get one of the possible local minimas
grid = RandomizedSearchCV( obj, params_nn, n_iter = 1000)
grid.fit(feature_table, target_values)
print(grid.best_params_)

#Best parameters after search results are as follows-
# best_params_ = {'tol':0.0001, 'hidden_layer_sizes':(14,7), 'activation':'tanh'
#'alpha' :5e-5, 'solver':'adam'}

# for 3 hidden layer configuration
# best_params = {'tol':0.0001, 'hidden_layer_sizes :(7,13,7), 'activation':'relu', 'alpha' :0.0001, 'solver' : 'adam'}
"""

mlp = MLPClassifier(activation='tanh',alpha=2e-05, hidden_layer_sizes=(7, 13, 7), max_iter=1000, solver='adam', tol=0.0001)
#even though the above mlp classifier is not giving that good results, it gives best output on kaggle servers, so we will go with this only

print("Format:\n\nAlgorithm_Name : mean_accuracy (std_of_accuracy)")

# 10 fold cross validation is performed to identify the performance of tuned classifier
scoring = 'accuracy'
kfold = KFold(n_splits=10, random_state=7)
cv_results = cross_val_score(mlp, feature_table, target_values, cv=kfold, scoring=scoring)
msg = "%s: %f (%f)" % ("mlp ", cv_results.mean(), cv_results.std())
print(msg)
mlp_result = mlp.fit(feature_table,target_values)

print('MLP Training Score: ' + str(mlp.score(feature_table,target_values)))

# using instance bagging to improve performance
bag_mlp = BaggingClassifier(base_estimator = mlp, n_estimators = 10)
kfold = KFold(n_splits=10, random_state=7)
cv_results = cross_val_score(bag_mlp, feature_table, target_values, cv=kfold, scoring=scoring)
msg = "%s: %f (%f)" % ("mlp bagged:", cv_results.mean(), cv_results.std())
print(msg + '\n')
bag_mlp_result = bag_mlp.fit(feature_table, target_values)

#mlp ends


#svm begins
"""
#hyperparameter space to be tuned is specified
params_svm = {"C" : [0.5,1,1.5,2], "kernel" : ["linear","poly","rbf","sigmoid"], "degree" : [2,3], "tol" : [0.001,0.0001,0.00001]}

obj = SVC()
grid = GridSearchCV( obj, params_svm)
grid.fit(feature_table, target_values)
print(grid.best_params_)
# Best parameters after search results are as follows-
# best_params_ = {C = 2, kernel = 'rbf', tol = 0.001, degree = 2}
"""

#cross validation suggests that some parameters (such as C) obtained from grid search are settled in a manner which causes overfitting.
#These parameters are adjusted to avoid overfitting, by cross validation check 
svm = SVC(C = 0.5, kernel = 'rbf', tol = 0.00001, degree = 2)
scoring = 'accuracy'
kfold = KFold(n_splits=10, random_state=7)
cv_results = cross_val_score(svm, feature_table, target_values, cv=kfold, scoring=scoring)
msg = "%s: %f (%f)" % ("svm", cv_results.mean(), cv_results.std())
print(msg)
svm_result = svm.fit(feature_table,target_values)

print('SVM Training Score: ' + str(svm.score(feature_table,target_values)))

# using instance bagging to improve performance
bag_svm = BaggingClassifier(base_estimator = svm, n_estimators = 10)
kfold = KFold(n_splits=10, random_state=7)
cv_results = cross_val_score(bag_svm, feature_table, target_values, cv=kfold, scoring=scoring)
msg = "%s: %f (%f)" % ("svm bagged:", cv_results.mean(), cv_results.std())
print(msg + '\n')
bag_svm_result = bag_svm.fit(feature_table, target_values)

#svm ends


#Decision Tree starts
"""
depth = []
for i in range(4,16):
       depth.append(i)

#hyperparameter space to be tuned is specified
params_dt = {"criterion" : ['gini','entropy'], "max_depth" : depth, "min_samples_split" : [2,3,4,5], "min_samples_leaf" : [1,2,3],
 "min_impurity_split" : [1e-07,1e-06,1e-08]}

obj = DecisionTreeClassifier()
grid = GridSearchCV( obj, params_dt)
grid.fit(feature_table, target_values)
print(grid.best_params_)
# Best parameters after search results are as follows-
# best_params_ = {criterion = 'entropy', min_samples_leaf = 1, min_samples_split = 4, min_impurity_split = 1e-07, max_depth = 5}
"""

dt = DecisionTreeClassifier(criterion = 'entropy', min_samples_leaf = 1, min_samples_split = 4, min_impurity_split = 1e-07, max_depth = 5)
scoring = 'accuracy'
#10 fold cross validation begins
kfold = KFold(n_splits=10, random_state=7)
cv_results = cross_val_score(dt, feature_table, target_values, cv=kfold, scoring=scoring)
msg = "%s: %f (%f)" % ("dt", cv_results.mean(), cv_results.std())
print(msg)
dt_result = dt.fit(feature_table,target_values)

print('Decision Tree Training Score: ' + str(dt.score(feature_table,target_values)))

# using instance bagging to improve performance
bag_dt = BaggingClassifier(base_estimator = dt, n_estimators = 10)
kfold = KFold(n_splits=10, random_state=7)
cv_results = cross_val_score(bag_dt, feature_table, target_values, cv=kfold, scoring=scoring)
msg = "%s: %f (%f)" % ("dt bagged:", cv_results.mean(), cv_results.std())
print(msg + '\n')
bag_dt_result = bag_dt.fit(feature_table, target_values)


#Decision tree Ends
#now as the classifiers have been tuned, they are ready for predicting 

#prediction part begins
test = pd.read_csv('test.csv')

#Same data processing and value imputing is being performed on test data, for processing by the classifiers
#converting text containing columns into numbers, for easy processing and visualisation
#also converting all empty spaces with most common column value for discrete variables and with median for continuous values
test["Embarked"][test["Embarked"] == "S"] = 0
test["Embarked"][test["Embarked"] == "C"] = 1
test["Embarked"][test["Embarked"] == "Q"] = 2
test["Embarked"] = test["Embarked"].fillna(test["Embarked"].mode())

#test["Sex"] = test["Sex"].fillna("male")
test["Sex"][test["Sex"] == "male"] = 0
test["Sex"][test["Sex"] == "female"] = 1
test["Sex"] = test["Sex"].fillna(test["Sex"].mode())

#try fillna of all columns using survival... will be better it think
test["Age"] = test["Age"].fillna(test["Age"].median())
test["Age"][test["Age"] < 16] = 0
test["Age"][test["Age"] >= 16][ test["Age"] < 60] = 1
test["Age"][test["Age"] >= 60] = 2
##

test["Pclass"] = test["Pclass"].fillna(test["Pclass"].mode())

test["Fare"] = test["Fare"].fillna(test["Fare"].median())
test["Fare"] = np.log10(test["Fare"]+1).astype(int)
##

#creating new coulumn "Relatives" containing sum of nymber of Siblings/Spouse/Parent or Children a person has onboard
test["Parch"] = test["Parch"].fillna(test["Parch"].mode())
test["SibSp"] = test["SibSp"].fillna(test["SibSp"].mode())

test["Relatives"] = test["SibSp"] + test["Parch"]
test["Relatives"] = test["Relatives"].fillna(test["Relatives"].mode())
##"""
#test_features = test[["Pclass","Sex","Age","Fare","Relatives"]]
test_features = test[["Pclass","Sex","Age","Fare","SibSp","Parch"]]

test_features = ft.transform(test_features)

#All 3 classifiers give similar performance. So a majority voting is performed between them for optimum results.
voting_result = VotingClassifier(estimators=[
         ('mlp', mlp),  ('bag_svm', bag_svm), ('bag_dt', bag_dt)], voting='hard')
#voting classifier performace is obtained using 10 fold cross validation
cv_results = cross_val_score(voting_result, feature_table, target_values, cv=kfold, scoring=scoring)
msg = "%s: %f (%f)" % ("voting result ", cv_results.mean(), cv_results.std())
print(msg)

voting_result = voting_result.fit(feature_table,target_values)

test_survive = bag_mlp.predict(test_features)
#test_survive = bag_svm.predict(test_features)
#test_survive = bag_dt.predict(test_features)

#predictions are stored in dataframe test_survive
#test_survive = voting_result.predict(test_features)
passenger_id = test["PassengerId"].values

#solution is saved on local system.
my_solution = pd.DataFrame(test_survive, passenger_id, columns = ["Survived"])
my_solution.to_csv("my_solution.csv", index_label = ["PassengerId"])


# Some conclusions - 
# bagged dt doesnt perform good
# bagged svm performs same as normal svm
# bagged voting classifier also gives similar performance
# bagged mlp isn't good either
# our best short is to use an SVM algo. as it gives best performance
