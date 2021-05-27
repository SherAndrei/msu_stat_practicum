import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error as MAE, mean_squared_error as MSE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression


def RMSE(y_actual, y_predicted):
    return np.sqrt(MSE(y_actual, y_predicted))


def calc_kfold_validation(x, y):
    lr = LinearRegression()
    cv_scores = cross_val_score(lr, x, y, cv=10, scoring="neg_mean_absolute_error")
    return np.mean(-cv_scores)


def select_best_combination(x, y):
    current_factors = x.columns.to_list()
    metric_base = calc_kfold_validation(x[current_factors], y)

    while 1 == 1:
        res = pd.Series(index=current_factors)
        for factor in current_factors:
            res.loc[factor] = calc_kfold_validation(x[list(set(current_factors)-{factor})], y)

        res = res.sort_values(ascending=True)
        if res.iloc[0] < metric_base:
            current_factors.remove(res.index.values[0])
            metric_base = res.iloc[0]
        else:
            break
            
    return current_factors


def main():
    df = pd.read_csv("BostonData.txt")
    X = df.drop(columns=['medv'])
    y = df['medv']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
    
    best_factors = select_best_combination(X_train, y_train)

    rg1 = LinearRegression().fit(X_train, y_train)
    all_factors_score = rg1.predict(X_test)
    
    rg2 = LinearRegression().fit(X_train[best_factors], y_train)
    best_factors_score = rg2.predict(X_test[best_factors])
    
    result = abs(RMSE(all_factors_score, y_test) - RMSE(best_factors_score, y_test))
    print("{0:.4f}".format(result))


if __name__ == '__main__':
    main()
