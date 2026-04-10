import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import joblib

# ── Load cleaned data ─────────────────────────────────────────────────────────
df = pd.read_csv("data/cleaned_data.csv")
print("Loaded shape:", df.shape)

# ── Split features and target ─────────────────────────────────────────────────
X = df.drop(columns=['price'])
y = df['price']

print("Total Features:", X.shape[1])

# ── Train / test split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training rows: {len(X_train)} | Test rows: {len(X_test)}")

# ── Train model ───────────────────────────────────────────────────────────────
model = XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=10,
    reg_lambda=10,
    objective='reg:squarederror',
    random_state=42,
    early_stopping_rounds=30,
    eval_metric='rmse'
)

print("\n⏳ Training started...")
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50
)
print("✅ Model trained successfully!")

# ── Save model and features ───────────────────────────────────────────────────
import os
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/property_model.pkl")
joblib.dump(X.columns.tolist(), "models/features.pkl")

print("✅ Saved: models/property_model.pkl")
print("✅ Saved: models/features.pkl")

# Reverse log transform
from sklearn.metrics import r2_score
import numpy as np
pred = np.expm1(model.predict(X_test))
y_test_actual = np.expm1(y_test)
r2 = r2_score(y_test_actual, pred)
print(f"R² after reverse transform: {r2:.4f}")