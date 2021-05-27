import pandas as pd
import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.stats.diagnostic import het_white
from scipy.stats import shapiro

def main():
    df = pd.read_csv('Boston.txt')
    X = add_constant(df[['rm', 'age', 'crim']])
    y = df[['medv']]
    
    ols = OLS(y, X)
    results = ols.fit()

    influence = results.get_influence()
    
    sigma2_hat = results.ssr / (len(y) - X.shape[1] - y.shape[1] - 1)
    stand_residuals = influence.resid / np.sqrt(sigma2_hat)
    
    if shapiro(stand_residuals)[1] < 0.05 or \
       het_white(influence.resid, X)[1] < 0.05:
        print("{0:.7f}".format(ols.predict(results.params, [1, 6.5, 70.0, 0.02])))
        return

    pred_res = results.get_prediction(np.array([1, 6.5, 70.0, 0.02]))
    print("{0:.7f}".format(pred_res.conf_int(alpha=0.1)[0][1]))


if __name__ == '__main__':
    main()
