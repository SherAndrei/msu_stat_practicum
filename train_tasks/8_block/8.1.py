import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("BostonData.txt")
X = df.drop(columns=["medv"])
y = df["medv"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

scaler = StandardScaler().fit(X_train, y_train)
X_train = scaler.transform(X_train)

reg = Ridge(alpha=1, fit_intercept=True).fit(X_train, y_train)

print(reg.coef_[abs(reg.coef_) > 0.1].shape[0])
