import pandas as pd
from scipy.stats import ttest_1samp

ncity = pd.read_csv('Ncity.txt')['ncity']
print(int(ttest_1samp(ncity, popmean=40000)[1] / 2 < 0.03))
