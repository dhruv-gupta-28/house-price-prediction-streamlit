import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="House Price Predictor",
    layout="wide",
    page_icon="🏠"
)

# =========================
# LOAD FILES
# =========================
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

feature_importance = pd.read_csv("feature_importance.csv")
mean_values = pd.read_csv("mean_values.csv")

# =========================
# HEADER
# =========================
st.markdown("# 🏠 House Price Prediction Dashboard")
st.caption("AI-powered property valuation system")

st.divider()

# =========================
# LAYOUT
# =========================
col1, col2 = st.columns([1,1])

# =========================
# INPUT SECTION
# =========================
with col1:
    st.subheader("📥 Property Details")

    grliv = st.number_input("Living Area (sq ft)", 500, 5000, 1500)
    total_bsmt = st.number_input("Basement Area", 0, 3000, 800)
    first_flr = st.number_input("1st Floor Area", 0, 3000, 1000)
    second_flr = st.number_input("2nd Floor Area", 0, 3000, 500)

    full_bath = st.number_input("Full Bathrooms", 0, 5, 2)
    half_bath = st.number_input("Half Bathrooms", 0, 3, 1)

    overall_qual = st.slider("Overall Quality", 1, 10, 5)
    year_built = st.number_input("Year Built", 1900, 2025, 2000)

# =========================
# OUTPUT SECTION
# =========================
with col2:
    st.subheader("📊 Prediction & Insights")

    if st.button("🚀 Predict Price"):

        # =========================
        # CREATE INPUT DATA
        # =========================
        input_dict = {
            "GrLivArea": grliv,
            "TotalBsmtSF": total_bsmt,
            "1stFlrSF": first_flr,
            "2ndFlrSF": second_flr,
            "FullBath": full_bath,
            "HalfBath": half_bath,
            "OverallQual": overall_qual,
            "YearBuilt": year_built
        }

        df = pd.DataFrame([input_dict])

        # FEATURE ENGINEERING
        df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
        df["TotalBathrooms"] = df["FullBath"] + (0.5 * df["HalfBath"])

        # ALIGN COLUMNS
        for col in columns:
            if col not in df.columns:
                df[col] = 0

        df = df[columns]

        # SCALE
        df_scaled = scaler.transform(df)

        # PREDICT
        pred = model.predict(df_scaled)
        pred = np.exp(pred)

        # =========================
        # RESULT DISPLAY
        # =========================
        st.metric("💰 Estimated Price", f"₹ {int(pred[0]):,}")

        st.divider()

        # =========================
        # FEATURE IMPORTANCE GRAPH
        # =========================
        st.subheader("📈 Top Influencing Features")

        top_features = feature_importance.head(10)

        fig, ax = plt.subplots()
        ax.barh(top_features["feature"], top_features["importance"])
        ax.invert_yaxis()
        ax.set_xlabel("Impact")
        st.pyplot(fig)

        # =========================
        # COMPARISON TABLE
        # =========================
        st.subheader("📊 Your Input vs Average")

        compare_df = pd.DataFrame({
            "Feature": ["GrLivArea", "OverallQual", "YearBuilt"],
            "Your Value": [grliv, overall_qual, year_built],
            "Average": [
                mean_values["GrLivArea"][0],
                mean_values["OverallQual"][0],
                mean_values["YearBuilt"][0]
            ]
        })

        st.dataframe(compare_df, use_container_width=True)

        # =========================
        # SIMPLE AI INSIGHTS
        # =========================
        st.subheader("🧠 Smart Insights")

        if grliv > mean_values["GrLivArea"][0]:
            st.success("✔ Larger house size increases price")
        else:
            st.warning("⚠ Smaller size may reduce price")

        if overall_qual > 5:
            st.success("✔ High construction quality boosts value")

        if year_built > 2000:
            st.info("ℹ Newer homes generally have higher prices")