import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Ipython.display import display
from sklearn.linear_model import LinearRegression

X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
# y = 1 * x_0 + 2 * x_1 + 3
y = np.dot(X, np.array([1, 2])) + 3
reg = LinearRegression().fit(X, y)
print(reg.score(X, y))
print(reg.coef_)

X = np.linspace(-10, 10, 100)
y = np.sin(X)
print(plt.plot(X, y, marker="X"))
plt.plot(X, y, marker="X")
plt.show()

data ={'name':["rui","feifei"],
       'work':["programming","play"]}
data_panda=pd.DataFrame(data)
display(data_panda)