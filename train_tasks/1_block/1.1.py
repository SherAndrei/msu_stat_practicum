import pandas as pd
from scipy.stats import ttest_1samp

iq = pd.read_csv("IQ.txt")['iq']
print("{0:.7f}".format(ttest_1samp(iq, popmean=110)[1] / 2))
