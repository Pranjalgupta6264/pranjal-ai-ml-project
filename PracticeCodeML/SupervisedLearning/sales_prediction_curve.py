import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Submitted by Pranjal Gupta
# Polynomial Regression - Advertisement Budget vs Sales

X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
y = np.array([10, 18, 28, 40, 55, 72, 90, 110])

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)

y_pred = model.predict(X_poly)

print("Predicted Sales:")
print(y_pred)

plt.scatter(X, y, label="Actual Data")
plt.plot(X, y_pred, label="Polynomial Curve")
plt.xlabel("Advertisement Budget")
plt.ylabel("Sales")
plt.title("Polynomial Regression: Advertisement Budget vs Sales")
plt.legend()
plt.show(block=True)
plt.close()