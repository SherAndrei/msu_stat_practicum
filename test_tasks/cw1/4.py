import pandas as pd
import numpy as np
from scipy.stats import f_oneway
import scikit_posthocs as sp
import scipy.stats as st

def LSD_Fisher(i, j, samples):
    n1, n2 = len(samples[i]), len(samples[j])
    k = len(samples)
    N = np.sum([len(samples[l]) for l in range(k)])
    SSe = np.sum([np.var(samples[l], ddof=0) * len(samples[l]) for l in range(k)])
    stat = (np.mean(samples[i]) - np.mean(samples[j]))/np.sqrt(SSe / (N - k) * (1.0/n1 + 1.0/n2))
    return 2*np.min([ st.t.cdf(stat, N - k), 1 - st.t.cdf(stat, N - k)])

df = pd.read_csv("ExamMath.txt")
x, y, z, w = df.x, df.y, df.z, df.w

if f_oneway(x, y, z, w)[1] < 0.1:
    print("{0:.9f}".format(min(LSD_Fisher(0, 2, [x, y, z, w]), sp.posthoc_scheffe([x, y, z, w])[1][3])))
else:
    print(0)
