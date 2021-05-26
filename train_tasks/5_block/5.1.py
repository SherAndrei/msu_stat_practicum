from pandas import read_csv
from scipy.stats import shapiro, f_oneway, kruskal 

data = read_csv("Pharmacy.txt")
x, y, z = data['x'], data['y'], data['z']

is_all_normal = ((shapiro(x)[1] > 0.03) and
                (shapiro(y)[1] > 0.03) and
                (shapiro(z)[1] > 0.03))

p_value = 0
if is_all_normal:
    p_value = f_oneway(x, y, z).pvalue
else:
    p_value = 3 * kruskal(x, y, z).pvalue

print(0 if p_value > 0.03 else '{0:.7f}'.format(p_value))
