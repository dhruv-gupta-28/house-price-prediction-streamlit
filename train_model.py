import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# =========================
# LOAD DATA
# =========================
df_train = pd.read_csv("train.csv")

# =========================
# REMOVE HIGH MISSING COLUMNS
# =========================
missing_percent = df_train.isnull().mean() * 100
good_cols = missing_percent[missing_percent < 50].index
df_train = df_train[good_cols]

# =========================
# HANDLE MISSING VALUES
# =========================
num_cols = df_train.select_dtypes(include=['int64','float64']).columns
cat_cols = df_train.select_dtypes(include=['object','string']).columns

for col in num_cols:
    if col != "SalePrice":
        df_train[col] = df_train[col].fillna(df_train[col].median())

for col in cat_cols:
    df_train[col] = df_train[col].fillna("Missing")

# =========================
# ENCODING
# =========================
df_train = pd.get_dummies(df_train)

# =========================
# FEATURE ENGINEERING
# =========================
df_train["TotalSF"] = df_train["TotalBsmtSF"] + df_train["1stFlrSF"] + df_train["2ndFlrSF"]
df_train["TotalBathrooms"] = df_train["FullBath"] + (0.5 * df_train["HalfBath"])

# =========================
# REMOVE OUTLIERS
# =========================
df_train = df_train[df_train["GrLivArea"] < 4000]

# =========================
# SPLIT
# =========================
X = df_train.drop("SalePrice", axis=1)
y = np.log(df_train["SalePrice"])

# =========================
# SCALING
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# MODEL
# =========================
model = LinearRegression()
model.fit(X_scaled, y)

# =========================
# SAVE MODEL FILES
# =========================
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(X.columns, open("columns.pkl", "wb"))

# =========================
# SAVE FEATURE IMPORTANCE
# =========================
feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.coef_
}).sort_values(by="importance", key=abs, ascending=False)

feature_importance.to_csv("feature_importance.csv", index=False)

# =========================
# SAVE AVERAGE VALUES
# =========================
X_mean = pd.DataFrame(X.mean()).T
X_mean.to_csv("mean_values.csv", index=False)

print("✅ Model + insights saved successfully!")