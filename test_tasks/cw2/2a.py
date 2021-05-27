import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error as MAE, mean_squared_error as MSE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

def calc_kfold_validation(k, x, y):
    x_k = x[['factor{}'.format(i) for i in range(1, k+1)]]
    lr = LinearRegression()
    cv_scores = cross_val_score(lr, x_k, y, cv=5, scoring="neg_mean_absolute_error")
    return np.mean(-cv_scores)

def get_polynomial(x_1, degree):
    x = x_1.rename('factor1').to_frame()
    for k in range(2, degree+1): 
        x['factor{}'.format(k)] = np.power(x['factor1'], k)
    return x

def best_degree (max_degree, x_1, y):
    x = get_polynomial(x_1, max_degree)
    res = pd.Series(index=range(1, max_degree + 1))
    for k in res.index.values:
        res.loc[k] = calc_kfold_validation(k, x, y)
    return res.idxmin()

def main():
    df = pd.read_csv("CarseatsData.txt")
    X = df.drop(columns=['Sales'])
    y = df['Sales']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

    x_1 = X_train['Price']
    degree = best_degree(10, x_1, y_train)
    x_degree = get_polynomial(x_1, degree)

    x_1_train = get_polynomial(X_train['Price'], degree)
    x_1_test  = get_polynomial(X_test['Price'], degree)

    poly_r = LinearRegression().fit(x_1_train, y_train)
    poly_score = poly_r.predict(x_1_test)

    lr = LinearRegression().fit(X_train, y_train)
    score = lr.predict(X_test)

    result = abs(MSE(score, y_test) - MSE(poly_score, y_test))

    print("{0:.6f}".format(result))


if __name__ == '__main__':
    main()
