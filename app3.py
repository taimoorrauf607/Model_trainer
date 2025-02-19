import streamlit as st
import io
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Title
st.title("🎯 Machine Learning Model Trainer")

# Upload Dataset
st.header("1️⃣ Upload Dataset")
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("📌 Dataset Preview:")
    st.dataframe(df.head())

    # Handling Missing Values
    preprocess = st.sidebar.radio("Choose preprocessing",('Data Overview','⚙️ Handle Missing Values','Inconsistency','Feature Eng','Train Model only'))
    if preprocess=='⚙️ Handle Missing Values':
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                option = st.selectbox(f"Select method to fill NaN values in {col}", ("Mean", "Median", "Mode", "Drop", "Specific Value"))
                if option == "Mean":
                    df[col].fillna(df[col].mean(), inplace=True)
                elif option == "Median":
                    df[col].fillna(df[col].median(), inplace=True)
                elif option == "Mode":
                    df[col].fillna(df[col].mode()[0], inplace=True)
                elif option == "Drop":
                    df.dropna(subset=[col], inplace=True)
                elif option == "Specific Value":
                    value = st.text_input(f"Enter specific value for {col}")
                    if value:
                        df[col].fillna(value, inplace=True)
        
        st.success("✅ Missing values handled successfully!")
        st.write("📌 Updated Dataset Preview After Handling Missing Values:")
        st.dataframe(df.head())
        st.dataframe(df.info())
        

    elif preprocess =='Feature Eng':
        # Selecting Features and Target
        target_column = st.selectbox("🎯 Select Target Column", df.columns)
        feature_columns = st.multiselect("📊 Select Feature Columns", df.columns, default=[col for col in df.columns if col != target_column])

        if feature_columns and target_column:
            X = df[feature_columns]
            y = df[target_column]
            # Identify categorical features
            cat_features = [col for col in feature_columns if df[col].dtype == 'object']

            # Encoding Categorical Features
            st.header("🛠️ Encode Categorical Features")
            encoding_method = st.selectbox("Select Encoding Method", ("Label Encoding", "One-Hot Encoding"))
            if encoding_method == "Label Encoding":
                label_encoders = {}
                for col in cat_features:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                    label_encoders[col] = le

                st.success("✅ Categorical features encoded successfully!")
                st.write("📌 Updated Dataset Preview After Encoding:")
                st.dataframe(X.head())

            elif encoding_method == "One-Hot Encoding":
                X = pd.get_dummies(X, columns=cat_features)
        
                st.success("✅ Categorical features encoded successfully!")
                st.write("📌 Updated Dataset Preview After Encoding:")
                st.dataframe(X.head())
            

        # Feature Scaling
        st.header("📏 Feature Scaling")
        scale_features = st.multiselect("Select Features to Scale", X.columns)
        if scale_features:
            scaler = StandardScaler()
            X[scale_features] = scaler.fit_transform(X[scale_features])
            st.write(f'scaled your {scale_features}')
            
            st.success("✅ Feature scaling applied successfully!")
            st.write("📌 Updated Dataset Preview After Feature Scaling:")
            st.dataframe(X.head())
    
         # Download Processed Dataset
        st.header("📥 Download Processed Dataset")
        
        # Convert DataFrame to CSV
        csv_buffer = io.StringIO()
        X.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        
        # Download button
        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name="processed_dataset.csv",
            mime="text/csv")
        st.success("✅ Processed dataset is ready for download!")

   
    elif preprocess=='Train Model only':
         # # Selecting Features and Target
        target_column = st.selectbox("🎯 Select Target Column", df.columns)
        feature_columns = st.multiselect("📊 Select Feature Columns", df.columns, default=[col for col in df.columns if col != target_column])

        if feature_columns and target_column:
            X = df[feature_columns]
            y = df[target_column]
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Sidebar: Choose Model Type
        model_type = st.sidebar.radio("Select Model Type", ("Regression", "Classification"))

        if model_type == "Regression":
            regressor_choice = st.sidebar.selectbox("Choose Regressor", ("CatBoost", "Stacking"))

            if regressor_choice == "CatBoost":
                # Sidebar: CatBoost Parameters
                st.sidebar.header("⚙️ CatBoost Parameters")
                iterations = st.sidebar.number_input("Iterations", value=1000, step=10, format="%d")
                learning_rate = st.sidebar.number_input("Learning Rate", value=0.1, format="%.6f")
                l2_leaf_reg = st.sidebar.number_input("L2 Leaf Regularization", value=3.5, format="%.6f")
                depth = st.sidebar.number_input("Depth", 1, 16, 6, 1)

                if st.button("🚀 Train Model"):
                    model = CatBoostRegressor(
                        iterations=iterations,
                        learning_rate=learning_rate,
                        depth=depth,
                        l2_leaf_reg=l2_leaf_reg,
                        verbose=0
                    )
                    model.fit(X_train, y_train, early_stopping_rounds=50, verbose=False)

                    y_pred = model.predict(X_test)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    st.success(f"✅ Model Trained Successfully! RMSE Score: {rmse:.6f}")
                    st.session_state["trained_model"] = model
                    st.session_state["feature_columns"] = feature_columns

            elif regressor_choice == "Stacking":
                base_learners = [
                    ("lr", LinearRegression()),
                    ("dt", DecisionTreeRegressor()),
                    ("rf", RandomForestRegressor())
                ]
                meta_learner = LinearRegression()
                model = StackingRegressor(estimators=base_learners, final_estimator=meta_learner)

                if st.button("🚀 Train Model"):
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    st.success(f"✅ Model Trained Successfully! RMSE Score: {rmse:.6f}")
                    st.session_state["trained_model"] = model
                    st.session_state["feature_columns"] = feature_columns

# Upload Test Dataset for Prediction
st.header("2️⃣ Upload Test Dataset for Prediction")
test_file = st.file_uploader("Upload your test dataset (CSV) for predictions", type=["csv"])

if test_file and "trained_model" in st.session_state:
    test_df = pd.read_csv(test_file)
    st.write("📌 Test Dataset Preview:")
    st.dataframe(test_df.head())

    missing_cols = [col for col in st.session_state["feature_columns"] if col not in test_df.columns]
    if missing_cols:
        st.error(f"❌ Missing columns in test file: {missing_cols}")
    else:
        if st.button("📊 Make Predictions"):
            predictions = st.session_state["trained_model"].predict(test_df[st.session_state["feature_columns"]])
            submission_df = pd.DataFrame({
                "id": range(300000, 300000 + len(test_df)),
                "submission": predictions
            })

            st.success("✅ Predictions made successfully!")
            st.write("📌 Preview of Submission File:")
            st.dataframe(submission_df.head())

            csv = submission_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Submission File", data=csv, file_name="submission.csv", mime="text/csv")