# 🚀 Streamlit-Based Interactive Machine Learning & Data Preprocessing App

## 📌 Overview
This project is a **Streamlit-based interactive application** for **data preprocessing, exploratory data analysis (EDA), feature engineering, and machine learning model training** (both classification & regression). Users can upload their dataset, handle missing values, explore data visually, and train ML models with custom parameters.

## 🎯 Key Features

### 🗂 Data Preprocessing
- Upload CSV files for analysis.
- Handle **missing values** using mean, median, mode, or custom values.
- Detect and fix **data inconsistencies** (e.g., inconsistent capitalization, duplicate entries).

### 📊 Exploratory Data Analysis (EDA)
- **Univariate Analysis**: Histograms, Box Plots, and Bar Charts.
- **Bivariate Analysis**: Scatter Plots and Line Charts.
- **Multivariate Analysis**: Scatter and Line Charts with a hue feature.
- Interactive selection of columns for visualization.

### ⚙️ Feature Engineering
- Encode categorical variables using **Label Encoding** or **One-Hot Encoding**.
- Scale numerical features using **StandardScaler**.

### 🤖 Machine Learning Model Training
- Select and train **classification** and **regression models**.
- Supported models:
  - **Classification**: Logistic Regression, Decision Tree, Random Forest, SVM, etc.
  - **Regression**: Linear Regression, Decision Tree, Random Forest, SVR, etc.
- Customize hyperparameters before training.
- Evaluate models using accuracy (classification) and RMSE (regression).

### 📥 Download Processed Data
- Users can download the preprocessed dataset after feature engineering.

## 🛠 How to Run
### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Run the Streamlit App
```bash
streamlit run app.py
```

## 📂 Project Structure
```
📁 project_root/
 ├── 📂 assets/            # Stores images & other assets
 ├── 📂 database/          # Data storage & processing
 ├── 📂 ui/                # UI components (Streamlit elements)
 ├── 📂 modules/           # ML models & data processing
 ├── app.py               # Main Streamlit application
 ├── helper.py            # Utility functions for preprocessing, visualization, etc.
 ├── requirements.txt      # Dependencies
 ├── README.md            # Project documentation
```

## 📌 Future Enhancements
- Add deep learning models.
- Implement real-time data visualization.
- Deploy as a web application.

---
💡 **Contributions are welcome!** Feel free to fork, modify, and improve the project. 🚀
