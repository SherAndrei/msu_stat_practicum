import pandas as pd
from scipy.stats import kruskal, f_oneway, shapiro

df = pd.read_csv("SalaryData.txt")
x, y, z, w = df['x'], df['y'], df['z'], df['w']
alpha = 0.01

if shapiro(x)[1] >= 0.05 and shapiro(y)[1] >= 0.05 and \
   shapiro(z)[1] >= 0.05 and shapiro(w)[1] >= 0.05:
    p_value = f_oneway(x, y, z, w)[1]
else:
    p_value = kruskal(x, y, z, w)[1]    

print(1 if p_value < alpha else "{0:.7f}".format(p_value))
