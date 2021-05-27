import pandas as pd
import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

def select_best_combination(y, x, metric, greater_is_better=False):
    current_factors = x.columns.to_list()
    ols = OLS(y, x[current_factors])
    results = ols.fit()
    metric_base = getattr(results, metric)

    while 1 == 1:
        res = pd.Series(index=current_factors)
        for factor in current_factors:
            ols = OLS(y, x[list(set(current_factors)-{factor})])
            results = ols.fit()
            res.loc[factor] = getattr(results, metric)

        if (greater_is_better):
            res = res.sort_values(ascending=False)
            if res.iloc[0] > metric_base:
                current_factors.remove(res.index.values[0])
                metric_base = res.iloc[0]
            else:
                break
        else:
            res = res.sort_values(ascending=True)
            if res.iloc[0] < metric_base:
                current_factors.remove(res.index.values[0])
                metric_base = res.iloc[0]
            else:
                break
    return current_factors


df = pd.read_csv('Life_Data.txt')

fill = df.median(axis=0)
df   = df.fillna(value=fill)

X = add_constant(df.drop(columns=['Life expectancy']))
y = df['Life expectancy']

aic_factors = select_best_combination(y, X, 'aic')
ar2_factors = select_best_combination(y, X, 'rsquared_adj', True)

print(np.abs(len(ar2_factors) - len(aic_factors)))
