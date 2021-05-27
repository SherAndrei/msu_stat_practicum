import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from statsmodels.tools.tools import add_constant
from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score

df = pd.read_csv("BostonData.txt")
X = add_constant(df.drop(columns=['medv']))
y = df[['medv']]

def adjusted_r2_squared(y_true, y_pred):
    n = len(y_true)
    k = X.shape[1]
    r2_coef = 1 - (1 - r2_score(y_true, y_pred)) * (n - 1) / (n - k)
    return r2_coef

adjusted_r2_scorer = make_scorer(adjusted_r2_squared, greater_is_better=True)

clf = LinearRegression()
cv = cross_val_score(clf, X, y, cv=5, scoring=adjusted_r2_scorer)

print("{:0.7f}".format(np.mean(cv)))
