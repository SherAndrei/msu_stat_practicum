import pandas as pd
import numpy as np
from scipy.stats import shapiro
from scipy.stats import ttest_rel
from scipy.stats import norm


df = pd.read_csv("Insulin.txt")
x = df['x']
y = df['y']

is_norm = shapiro(x)[1] > 0.05 and shapiro(y)[1] > 0.05

if is_norm:
    p_value = ttest_rel(x, y)[1]
else:
    z = y - x
    z = z[z!=0]
    b = sum(z > 0)
    n = len(z)
    b_star = (b - n*0.5) / np.sqrt(n*0.25)
    p_value = norm.cdf(b_star, loc=0, scale=1)

print("{0:.7f}".format(p_value))
