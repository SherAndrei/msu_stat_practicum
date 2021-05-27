import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error as MAE, mean_squared_error as MSE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler


def RMSE(y_actual, y_predicted):
    return np.sqrt(MSE(y_actual, y_predicted))


def main():
    df = pd.read_csv("BostonData.txt")
    X = df.drop(columns=['medv'])
    y = df['medv']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    rr_orig = Ridge(alpha=1, fit_intercept=True).fit(X_train_scaled, y_train)
    y_pred_orig = rr_orig.predict(X_test_scaled)

    better_x_train = X_train_scaled[:, np.abs(rr_orig.coef_).argsort()[5:]]
    better_x_test  = X_test_scaled[:, np.abs(rr_orig.coef_).argsort()[5:]]

    rr1 = Ridge(alpha=1, fit_intercept=True).fit(better_x_train, y_train)
    y_pred1 = rr1.predict(better_x_test)

    result = abs(RMSE(y_pred1, y_test) - RMSE(y_pred_orig, y_test))
    print("{0:.6f}".format(result))


if __name__ == '__main__':
    main()
