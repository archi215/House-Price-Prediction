import pandas as pd
import numpy as np

# ── Load dataset ──────────────────────────────────────────────────────────────
df = pd.read_csv("data/data.csv")
print("Original shape:", df.shape)
print(df.head())

# ── 1. Parse date and extract time features ───────────────────────────────────
df['date'] = pd.to_datetime(df['date'])
df['sale_year']    = df['date'].dt.year
df['sale_month']   = df['date'].dt.month
df['sale_quarter'] = df['date'].dt.quarter
df.drop(columns=['date'], inplace=True)
print("✅ Date features extracted")

# ── 2. Drop columns that are noisy or have too many unique values ─────────────
df.drop(columns=['street', 'country'], inplace=True)
print("✅ Dropped street and country columns")

# ── 3. Extract zip code from statezip (e.g. "WA 98133" → 98133) ──────────────
df['zipcode'] = df['statezip'].str.extract(r'(\d+)').astype(int)
df.drop(columns=['statezip'], inplace=True)
print("✅ Zipcode extracted")

# ── 4. Target encode city (replace city name with avg price of that city) ─────
city_mean = df.groupby('city')['price'].mean()
df['city_encoded'] = df['city'].map(city_mean)
df.drop(columns=['city'], inplace=True)
print("✅ City target encoded")

# ── 5. Engineer new features ──────────────────────────────────────────────────
df['house_age']         = df['sale_year'] - df['yr_built']
df['was_renovated']     = (df['yr_renovated'] > 0).astype(int)
df['years_since_reno']  = df['sale_year'] - df['yr_renovated'].replace(0, np.nan)
df['years_since_reno']  = df['years_since_reno'].fillna(df['house_age'])
df['total_rooms']       = df['bedrooms'] + df['bathrooms']
df['basement_flag']     = (df['sqft_basement'] > 0).astype(int)
df['living_lot_ratio']  = df['sqft_living'] / (df['sqft_lot'] + 1)
df['price_per_sqft_est']= df['sqft_living'] / (df['sqft_lot'] + 1)
print("✅ New features engineered")

# ── 6. Remove top 1% price outliers ──────────────────────────────────────────
upper = df['price'].quantile(0.99)
df = df[df['price'] <= upper]
print(f"✅ Outliers removed. Rows remaining: {len(df)}")

# ── 7. Check final result ─────────────────────────────────────────────────────
print("\nFinal shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nSample data:")
print(df.head())

# ── 8. Save cleaned data ──────────────────────────────────────────────────────
df.to_csv("data/cleaned_data.csv", index=False)
print("\n✅ Saved: data/cleaned_data.csv")

# Remove houses with 0 price
df = df[df['price'] > 0]

# Remove unrealistic bedroom values
df = df[df['bedrooms'] < 20]

# Log transform price (helps XGBoost a lot)
df['price'] = np.log1p(df['price'])

df.to_csv("data/cleaned_data.csv", index=False)