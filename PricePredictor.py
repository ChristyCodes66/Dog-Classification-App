import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import statsmodels as sm

housing_data = pd.read_csv('/Users/user/UniIDE/USWS/Datasets/housing.csv')

dummy_data = pd.get_dummies(housing_data.ocean_proximity)
housing_dummy = pd.concat([housing_data, dummy_data])
housing_clean = housing_dummy.drop(housing_dummy['ocean_proximity', 'ISLAND'])

# isolate target variable/column from table
X = housing_clean.drop(columns='median_house_value')

y = housing_clean['median_house_value']

print(y.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# build model

logistic = LogisticRegression()
X_train = sm.add_constant(X_train)
logistic.fit(X_train, y_train)

logisticAccuracy = logistic.score(X_test, y_test)
logisticAccuracy = "{:.0%}".format(logisticAccuracy)
print("Accuracy: " + str(logisticAccuracy))

cleanLogistic = sm.Logit
#
# print("clean Log")
#
# test = logistic.predict(X2_test)
#
# print(test)
#
# from sklearn.metrics import accuracy_score, confusion_matrix
#
# accuracy_score = accuracy_score(y2_test, test)
# print(accuracy_score)
