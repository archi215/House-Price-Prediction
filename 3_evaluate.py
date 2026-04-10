import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import joblib
import matplotlib.pyplot as plt
import os

# ── Load cleaned data ─────────────────────────────────────────────────────────
df = pd.read_csv("data/cleaned_data.csv")

X = df.drop(columns=['price'])
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Load saved model ──────────────────────────────────────────────────────────
model = joblib.load("models/property_model.pkl")
print("✅ Model loaded")

# ── Make predictions ──────────────────────────────────────────────────────────
pred_log = model.predict(X_test)

# ── Reverse log transform to get actual prices ────────────────────────────────
pred_actual  = np.expm1(pred_log)
y_test_actual = np.expm1(y_test)

# ── Calculate metrics on actual prices ───────────────────────────────────────
mae  = mean_absolute_error(y_test_actual, pred_actual)
rmse = np.sqrt(mean_squared_error(y_test_actual, pred_actual))
r2   = r2_score(y_test_actual, pred_actual)
mask = y_test_actual > 0
mape = np.mean(np.abs((y_test_actual[mask] - pred_actual[mask]) / y_test_actual[mask])) * 100

print("\n📊 Model Performance:")
print(f"   MAE   : ${mae:,.0f}")
print(f"   RMSE  : ${rmse:,.0f}")
print(f"   R²    : {r2:.4f}")
print(f"   MAPE  : {mape:.2f}%")

if r2 >= 0.80:
    print("\n✅ Great model! R² is above 0.80")
elif r2 >= 0.60:
    print("\n⚠️  Decent model. R² could be improved")
else:
    print("\n❌ Model needs improvement")

# ── Plot 1: Actual vs Predicted ───────────────────────────────────────────────
os.makedirs("outputs", exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(y_test_actual, pred_actual, alpha=0.3, color='steelblue')
axes[0].plot(
    [y_test_actual.min(), y_test_actual.max()],
    [y_test_actual.min(), y_test_actual.max()],
    'r--', lw=2
)
axes[0].set_xlabel("Actual Price ($)")
axes[0].set_ylabel("Predicted Price ($)")
axes[0].set_title("Actual vs Predicted")

# ── Plot 2: Residuals ─────────────────────────────────────────────────────────
residuals = y_test_actual - pred_actual
axes[1].hist(residuals, bins=50, color='salmon', edgecolor='black')
axes[1].axvline(0, color='red', linestyle='--')
axes[1].set_title("Residuals Distribution")
axes[1].set_xlabel("Residual (Actual - Predicted)")

plt.tight_layout()
plt.savefig("outputs/evaluation.png", dpi=150)
plt.show()
print("\n✅ Saved: outputs/evaluation.png")

# ── Feature Importance ────────────────────────────────────────────────────────
imp = pd.Series(model.feature_importances_, index=X.columns)
top15 = imp.sort_values(ascending=False).head(15)

plt.figure(figsize=(10, 6))
top15.sort_values().plot(kind='barh', color='steelblue')
plt.title("Top 15 Important Features")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("outputs/feature_importance.png", dpi=150)
plt.show()
print("✅ Saved: outputs/feature_importance.png")