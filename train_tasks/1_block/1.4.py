from scipy.stats import norm, binom_test, wilcoxon
import pandas as pd
import numpy as np

z = pd.read_csv('Insurance.txt')['insurance']
z -= 450
z = z[z!=0]

print('{0:.8f}'.format(
    min(wilcoxon(z)[1], binom_test(sum(z > 0), len(z), 0.5))))
