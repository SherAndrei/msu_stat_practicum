import pandas as pd
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error as MAE, mean_squared_error as MSE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

def RMSE(y_actual, y_predicted):
    return np.sqrt(MSE(y_actual, y_predicted))

df = pd.read_csv("BostonData.txt")
X = df.drop(columns=['medv'])
y = df[['medv']]

rmse_scorer = make_scorer(RMSE, greater_is_better=False)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10, train_size=0.7)
scaler = StandardScaler().fit(X_train, y_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled  = scaler.transform(X_test)

alphas = np.logspace(-2, 3, 20)
searcher = GridSearchCV(Ridge(), [{"alpha": alphas}], scoring=rmse_scorer, cv=10)
searcher.fit(X_train_scaled, y_train)

print("{0:.5f}".format(searcher.best_params_["alpha"]))
