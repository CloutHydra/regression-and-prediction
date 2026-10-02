from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

# Load California housing dataset
data = fetch_california_housing(as_frame=True)
df = data.frame
X = df[data.feature_names]
y = df["MedHouseVal"]

# Fit a linear regression model using the original features
model = LinearRegression()
model.fit(X, y)
raw_coefficients = pd.Series(model.coef_, index=X.columns)

# Standardize the inputs and target, then refit the model
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()

model_std = LinearRegression()
model_std.fit(X_scaled, y_scaled)
std_coefs = pd.Series(model_std.coef_, index=X.columns)

# Compare raw and standardized coefficients
summary = pd.DataFrame(
    {
        "Raw Coefficients": raw_coefficients,
        "Standardized Coefficients": std_coefs,
        "Absolute Std Coef": np.abs(std_coefs),
    }
).sort_values(by="Absolute Std Coef", ascending=False)

print(summary)
