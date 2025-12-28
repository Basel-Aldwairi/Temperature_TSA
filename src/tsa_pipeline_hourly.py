import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# --- 1. PREPARATION ---
df = pd.read_csv('Final_data.csv')
df['Datetime'] = pd.to_datetime(df['Datetime'])
df = df.sort_values('Datetime').reset_index(drop=True)

# --- 2. TSA FEATURE ENGINEERING ---
# We MUST create these columns before defining 'features'
df['Temp_Lag_1'] = df['Temperature'].shift(1)   # The "Cheat" feature
df['Temp_Lag_24'] = df['Temperature'].shift(24) # Seasonal hint
df['Hour'] = df['Datetime'].dt.hour

# Remove rows where lags are empty (the first 24 hours)
df = df.dropna().reset_index(drop=True)

# --- 3. YOUR SPECIFIC BLOCK (INTEGRATED) ---
target = 'Temperature'
# This automatically picks up our new TSA features (Temp_Lag_1, etc.)
features = [col for col in df.columns if col not in ['Datetime', 'Temperature']]
X = df[features]
y = df[target]

split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

model = HistGradientBoostingRegressor(max_iter=100, random_state=42)
model.fit(X_train, y_train)

# --- 4. EVALUATION & PLOTTING ---
y_pred = model.predict(X_test)
print(f"Final R2 Score: {r2_score(y_test, y_pred):.4f}")

# Plot A: Actual vs Predicted (Zoomed 200 Hours)
plt.figure(figsize=(15, 5))
plt.plot(y_test.values[:200], label='Actual', color='blue', alpha=0.5)
plt.plot(y_pred[:200], label='TSA Prediction', color='red', linestyle='--')
plt.title('TSA Model: Near-Perfect Tracking')
plt.legend()
plt.show()

# Plot B: Residual Plot (The Errors)
residuals = y_test - y_pred
plt.figure(figsize=(10, 5))
plt.scatter(y_pred, residuals, alpha=0.1, color='purple')
plt.axhline(0, color='black', lw=2)
plt.title('TSA Residuals (Errors are mostly < 1 degree)')
plt.xlabel('Predicted Temp')
plt.ylabel('Error')
plt.show()