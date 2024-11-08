from xgboost import XGBClassifier
# read data
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
print('load data')
data = load_iris()
print('split data')
X_train, X_test, y_train, y_test = train_test_split(data['data'], data['target'], test_size=.2)
# create model instance
print('create classifier model instance')
bst = XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='binary:logistic')
# fit model
print('fit the model')
bst.fit(X_train, y_train)
# make predictions
print('make predictions')
preds = bst.predict(X_test)
print(preds)