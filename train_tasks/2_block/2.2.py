from pandas import read_csv
from scipy.stats import kstest, gamma

def MyGamma(X):
    return gamma.cdf(X, a=600, scale=0.3)

X = read_csv("Insurance.txt")['insurance']
print('{0:.7f}'.format(kstest(X, MyGamma)[1]))
