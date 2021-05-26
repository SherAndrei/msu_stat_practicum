from pandas import read_csv
from scipy.stats import shapiro
from statsmodels.stats.diagnostic import lilliefors

X = read_csv('Lake.txt')['lake']
print(int(not (shapiro(X)[1] > 0.03 and lilliefors(X)[1] > 0.03)))
