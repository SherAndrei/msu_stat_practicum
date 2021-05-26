import numpy as np
from scipy.stats import chi2_contingency

table = np.array([[150, 90], [170, 70]])

print('{0:.9f}'.format(chi2_contingency(table)[1]))
