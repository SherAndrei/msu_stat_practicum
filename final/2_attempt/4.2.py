import pandas as pd
from scipy.stats import kruskal, f_oneway, shapiro, bartlett

df = pd.read_csv('Exam.txt')
x, y, z, w = df['x'], df['y'], df['z'], df['w']

if bartlett(x, y, z, w)[1] < 0.05 or \
   shapiro(x)[1] < 0.05 or shapiro(y)[1] < 0.05 or \
   shapiro(z)[1] < 0.05 or shapiro(w)[1] < 0.05:
    p_value = kruskal(x, y, z, w)[1]
else:
    p_value = f_oneway(x, y, z, w)[1]

print('{0:.7f}'.format(p_value) if p_value < 0.03 else 0)


