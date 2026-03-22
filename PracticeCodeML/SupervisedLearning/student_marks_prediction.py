import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Submitted by Pranjal Gupta
# Linear Regression - Study Hours vs Marks

X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
y = np.array([35, 40, 50, 55, 65, 70, 80, 85])

model = LinearRegression()
model.fit(X, y)

predicted_marks = model.predict(X)

print("Predicted values:")
print(predicted_marks)

plt.scatter(X, y, label="Actual Data")
plt.plot(X, predicted_marks, label="Regression Line")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Linear Regression: Study Hours vs Marks")
plt.legend()
plt.show()