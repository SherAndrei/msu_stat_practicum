import pandas as pd
from scipy.stats import f_oneway
import scikit_posthocs as sp

def LSD_Fisher(i, j, samples):
    # Вычисляем объемы выборок
    n1, n2 = len(samples[i]), len(samples[j])
    k = len(samples)
    # Объем всех выборок    
    N = np.sum([len(samples[l]) for l in range(k)])
    # сумма по всем выборкам    
    SSe = np.sum([np.var(samples[l], ddof=0) * len(samples[l]) for l in range(k)])
    stat = (np.mean(samples[i]) - np.mean(samples[j]))/np.sqrt(SSe / (N - k) * (1.0/n1 + 1.0/n2))
    return 2*np.min([ st.t.cdf(stat, N - k), 1 - st.t.cdf(stat, N - k)])


df = pd.read_csv("Salary.txt")
x, y, z = df['x'], df['y'], df['z']
alpha = 0.02

if f_oneway(x, y, z)[1] < alpha:
    print('{0:.8f}'.format(min(LSD_Fisher(0, 1, [x, y, z]), sp.posthoc_scheffe([x, y, z]))))
else:
    print(0)

