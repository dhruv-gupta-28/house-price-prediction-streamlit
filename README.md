# 🏠 House Price Prediction Streamlit App

An end-to-end **Machine Learning web application** that predicts house prices based on property features and provides **insights + visual analytics**.

🔗 **Live App:** https://housepriceprediction2823.streamlit.app/

---

## 🚀 Overview

This project demonstrates a complete ML pipeline:

- Data preprocessing & cleaning
- Feature engineering
- Model training (Linear Regression)
- Model deployment using Streamlit
- Interactive UI with insights & visualizations

The app allows users to input house details and instantly get a **predicted price along with explanations**.

---

## ✨ Features

- 🧠 **Accurate Price Prediction**
- 📊 **Feature Importance Visualization**
- 📈 **Input vs Average Comparison**
- 💡 **Smart Insights & Interpretations**
- 🎨 **Clean & Interactive UI**
- 🌐 **Deployed Live using Streamlit Cloud**

---

## 🛠️ Tech Stack

- **Python**
- **Pandas & NumPy**
- **Scikit-learn**
- **Matplotlib**
- **Streamlit**

---

## ⚙️ How It Works

### 1. Data Preprocessing

- Removed columns with excessive missing values
- Filled missing numerical values with median
- Filled categorical values with "Missing"
- Applied one-hot encoding

---

### 2. Feature Engineering

Created new features:

- `TotalSF` → Total area (Basement + Floors)
- `TotalBathrooms` → Combined bathroom score

---

### 3. Model Training

- Model: **Linear Regression**
- Target variable transformed using **log scaling**
- Applied **StandardScaler**
- Removed outliers for better accuracy

---

### 4. Prediction Flow

User Input → Feature Engineering → Scaling → Model Prediction → Exponential Transform → Final Price

---

## 📊 Insights Provided

- Top features affecting price
- Comparison with average properties
- Human-readable explanations:
  - Size impact
  - Quality influence
  - Age factor

---

## 📁 Project Structure

```
├── app.py                     # Streamlit app
├── train_model.py             # Model training script
├── model.pkl                  # Trained model
├── scaler.pkl                 # Scaler
├── columns.pkl                # Feature columns
├── feature_importance.csv     # Feature importance
├── mean_values.csv            # Average values
├── requirements.txt           # Dependencies
└── README.md                  # Project documentation
```

---

## ▶️ Run Locally

### 1. Clone the repo

```bash
git clone https://github.com/dhruv-gupta-28/house-price-prediction-streamlit.git
cd house-price-prediction-streamlit
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

---

## 🌐 Deployment

This app is deployed using **Streamlit Cloud**:

- Connected to GitHub repo
- Auto-deploy on push
- Publicly accessible via link

---

## 📌 Future Improvements

- 🔥 Use advanced models (XGBoost / Random Forest)
- 📊 Add SHAP explainability
- 📂 Batch prediction via CSV upload
- 📍 Include location-based pricing
- 🎨 Enhance UI/UX further

---

## 👨‍💻 Author

**Dhruv Gupta**

---

## ⭐ If you like this project

Give it a ⭐ on GitHub and feel free to fork it!
