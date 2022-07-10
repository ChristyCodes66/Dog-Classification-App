import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm

housing_data = pd.read_csv('/Users/user/UniIDE/USWS/Datasets/housing.csv')

housing_data_na = housing_data.dropna(subset=["total_bedrooms"])
dummy_data = pd.get_dummies(housing_data_na.ocean_proximity)
housing_dummy = pd.concat([housing_data_na, dummy_data], axis='columns')
housing_clean = housing_dummy.drop(['ocean_proximity', 'ISLAND'], axis='columns')


# remove outliers
housing_clean['expensive?'] = np.where(housing_clean['median_house_value'] > 250000, 1, 0)
X = housing_clean.drop(columns=['median_house_value', 'expensive?'])
y = housing_clean['expensive?']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1984)

# build model

logistic = LogisticRegression(solver='lbfgs', max_iter=2050)
logistic.fit(X_train, y_train)

logisticAccuracy = logistic.score(X_test, y_test)
logisticAccuracy = "{:.0%}".format(logisticAccuracy)
print("Accuracy: " + str(logisticAccuracy))

cleanLogistic = sm.Logit(y_train, X_train).fit()

print(cleanLogistic.summary())

test_prediction = logistic.predict(X_test)

print("Test prediction: " + str(test_prediction))

