import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# 1. Generate synthetic data with some noise
np.random.seed(42)
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.5, X.shape[0]) # A slightly curved base with noise

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Sort test data for consistent plotting
sort_idx = np.argsort(X_test.flatten())
X_test_sorted = X_test[sort_idx]
y_test_sorted = y_test[sort_idx]

# 2. Train a simple linear regression model (Occam's Razor: simpler explanation)
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear_train = linear_model.predict(X_train)
y_pred_linear_test = linear_model.predict(X_test_sorted)

# 3. Train a complex polynomial regression model (high degree, prone to overfitting)
poly_degree = 15 # A high degree to demonstrate overfitting
poly_features = PolynomialFeatures(degree=poly_degree, include_bias=False)
X_poly_train = poly_features.fit_transform(X_train)
X_poly_test = poly_features.transform(X_test_sorted)

poly_model = LinearRegression()
poly_model.fit(X_poly_train, y_train)
y_pred_poly_train = poly_model.predict(X_poly_train)
y_pred_poly_test = poly_model.predict(X_poly_test)

# 4. Evaluate models
mse_linear_train = mean_squared_error(y_train, y_pred_linear_train)
mse_linear_test = mean_squared_error(y_test_sorted, y_pred_linear_test)

mse_poly_train = mean_squared_error(y_train, y_pred_poly_train)
mse_poly_test = mean_squared_error(y_test_sorted, y_pred_poly_test)

print(f"--- Model Evaluation (Occam's Razor in action) ---")
print(f"Linear Model (Simple):")
print(f"  Training MSE: {mse_linear_train:.4f}")
print(f"  Test MSE:     {mse_linear_test:.4f}") # Often better generalization despite higher training error

print(f"\nPolynomial Model (Degree {poly_degree}, Complex):")
print(f"  Training MSE: {mse_poly_train:.4f}") # Often very low, indicating it fits training data well
print(f"  Test MSE:     {mse_poly_test:.4f}")   # Often much higher, indicating overfitting and poor generalization

# Occam's Razor Principle:
# The complex model might have a lower training error, but if its test error is significantly higher,
# it indicates overfitting. The simpler linear model, even with a slightly higher training error,
# often generalizes better to unseen data, making it the preferred choice according to Occam's Razor
# for its simplicity and robustness.

# 5. Visualize the results
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, s=20, edgecolor="k", label="Training data")
plt.scatter(X_test, y_test, s=20, edgecolor="red", facecolor="none", label="Test data")

# Plot simple linear model
plt.plot(X_test_sorted, y_pred_linear_test, color="blue", linestyle='--', label="Linear Model (Simple)")

# Plot complex polynomial model
plt.plot(X_test_sorted, y_pred_poly_test, color="red", label=f"Polynomial Model (Degree {poly_degree}, Complex)")

plt.xlabel("X")
plt.ylabel("y")
plt.title("Occam's Razor: Simple vs. Complex Models in AI")
plt.legend()
plt.ylim(-2, 2) # Set y-axis limits for better visualization
plt.grid(True)
plt.show()
