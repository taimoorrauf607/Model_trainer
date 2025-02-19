import streamlit as st
import pandas as pd
import numpy as np
import io  # For handling file conversion
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

    # Drop Unnecessary Columns
    st.header("🗑️ Drop Unnecessary Columns")
    columns_to_drop = st.multiselect("Select columns to drop", df.columns)
    if columns_to_drop:
        df.drop(columns=columns_to_drop, inplace=True)
        st.success(f"✅ Dropped columns: {', '.join(columns_to_drop)}")
        st.write("📌 Updated Dataset After Dropping Columns:")
        st.dataframe(df.head())

    # Check if dataset has missing values
    has_missing_values = df.isnull().sum().sum() > 0

    # Preprocessing Options
    preprocess_options = ['Feature Eng', 'Train Model only']
    if has_missing_values:
        preprocess_options.insert(0, '⚙️ Handle Missing Values')  # Add option if missing values exist

    preprocess = st.sidebar.radio("Choose preprocessing", preprocess_options)

    # Handle Missing Values
    if preprocess == '⚙️ Handle Missing Values':
        df_missing = df.copy()  # Keep a copy of the dataset

        for col in df_missing.columns:
            if df_missing[col].isnull().sum() > 0:
                # Check if column is categorical or numerical
                if df_missing[col].dtype == 'object':  # Categorical column
                    option = st.selectbox(f"Select method to fill NaN values in {col}", 
                                          ("Specific Value", "Drop"))
                    if option == "Specific Value":
                        value = st.text_input(f"Enter specific value for {col}")
                        if value:
                            df_missing[col].fillna(value, inplace=True)
                    elif option == "Drop":
                        df_missing.dropna(subset=[col], inplace=True)
                else:  # Numerical column
                    option = st.selectbox(f"Select method to fill NaN values in {col}", 
                                          ("Mean", "Median", "Mode"))
                    if option == "Mean":
                        df_missing[col].fillna(df_missing[col].mean(), inplace=True)
                    elif option == "Median":
                        df_missing[col].fillna(df_missing[col].median(), inplace=True)
                    elif option == "Mode":
                        df_missing[col].fillna(df_missing[col].mode()[0], inplace=True)

        st.success("✅ Missing values handled successfully!")
        st.write("📌 Updated Dataset Preview After Handling Missing Values:")
        st.dataframe(df_missing.head())

        # Store df_missing in session_state for future steps
        st.session_state["df_missing"] = df_missing

    elif preprocess == 'Feature Eng':
        if 'df_missing' in st.session_state:
            df_feature_eng = st.session_state["df_missing"].copy()
        else:
            df_feature_eng = df.copy()  # Use original dataset if missing values were skipped

        # Selecting Features and Target
        target_column = st.selectbox("🎯 Select Target Column", df_feature_eng.columns)
        feature_columns = st.multiselect("📊 Select Feature Columns", df_feature_eng.columns, 
                                         default=[col for col in df_feature_eng.columns if col != target_column])

        if feature_columns and target_column:
            X = df_feature_eng[feature_columns].copy()  # Work only on selected features
            y = df_feature_eng[target_column]

            # Identify categorical features
            cat_features = [col for col in feature_columns if df_feature_eng[col].dtype == 'object']

            # Encoding Categorical Features
            st.header("🛠️ Encode Categorical Features")
            encoding_method = st.selectbox("Select Encoding Method", ("Label Encoding", "One-Hot Encoding"))

            if encoding_method == "Label Encoding":
                label_encoders = {}
                for col in cat_features:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                    label_encoders[col] = le

            elif encoding_method == "One-Hot Encoding":
                X = pd.get_dummies(X, columns=cat_features)

            st.success("✅ Categorical features encoded successfully!")
            st.write("📌 Updated Feature Set After Encoding:")
            st.dataframe(X.head())

        # Feature Scaling
        st.header("📏 Feature Scaling")
        scale_features = st.multiselect("Select Features to Scale", X.columns)
        if scale_features:
            scaler = StandardScaler()
            X[scale_features] = scaler.fit_transform(X[scale_features])
            st.write(f"✅ Scaled Features: {', '.join(scale_features)}")

            st.success("✅ Feature scaling applied successfully!")
            st.write("📌 Updated Feature Set After Feature Scaling:")
            st.dataframe(X.head())

        # Merge Processed Features into the Original Dataset
        df_feature_eng = df_feature_eng.drop(columns=feature_columns, errors='ignore')  # Remove old versions of selected features
        df_feature_eng = pd.concat([df_feature_eng, X], axis=1)  # Add updated features

        st.write("📌 Updated Dataset After Feature Engineering:")
        st.dataframe(df_feature_eng.head())

        # Store df_feature_eng in session_state for download
        st.session_state["df_feature_eng"] = df_feature_eng

    # Download Processed Dataset
    if 'df_feature_eng' in st.session_state:
        st.header("📥 Download Processed Dataset")
        
        # Convert DataFrame to CSV
        df_to_download = st.session_state["df_feature_eng"]
        csv_buffer = io.StringIO()
        df_to_download.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        
        # Download button
        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name="processed_dataset.csv",
            mime="text/csv"
        )

        st.success("✅ Processed dataset is ready for download!")
