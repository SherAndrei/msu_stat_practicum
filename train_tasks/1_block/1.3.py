import pandas as pd
from scipy.stats import norm
import numpy as np

data = pd.read_csv('Museum.txt')['museum']

def cws(X, assumed_mean):
    X = np.array(X)
    n = len(X)
    return (X.sum() - n * assumed_mean) / np.sqrt(n * X.var(ddof=1)) 

p_value = norm.cdf(cws(data, 100))
print("{0:.6f}".format(p_value))
