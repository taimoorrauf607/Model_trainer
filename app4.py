import streamlit as st
import pandas as pd
import numpy as np
import io  # For file handling
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Title
st.title("🎯 Machine Learning Model Trainer")

# Upload Dataset
st.header("1️⃣ Upload Dataset")
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    st.title("welcome to Data preprocessing part")
        # Ensure df is stored in session state to persist changes
    if "df_processed" not in st.session_state:
        st.session_state["df_processed"] = df.copy()

    df = st.session_state["df_processed"]  # Always use the latest processed dataset

    # Sidebar - Preprocessing Steps
    preprocess_options = ["📊 Data Overview", "🔎 Inconsistency", "📈 EDA", "⚙️ Feature Eng", "🤖 Train Model only"]
    if df.isnull().sum().sum() > 0:
        preprocess_options.insert(1, "⚙️ Handle Missing Values")

    preprocess = st.sidebar.radio("Choose Step", preprocess_options)

    # 📊 DATA OVERVIEW
    if preprocess == "📊 Data Overview":
        st.header("📊 Dataset Overview")
        st.write("📌 **Shape:**", df.shape)

        col1, col2 = st.columns(2)
        with col1:
            st.write("🔍 **Data Types**")
            dtype_df = pd.DataFrame(df.dtypes, columns=["Data Type"])
            st.dataframe(dtype_df)

        with col2:
            st.write("⚠️ **Missing Values Summary:**")
            missing_df = pd.DataFrame(df.isnull().sum(), columns=["Missing Values"])
            st.dataframe(missing_df)

        col1, col2 = st.columns(2)
        with col1:
            st.write("📊 **Statistical Summary:**")
            st.dataframe(df.describe())

        with col2:
            st.write("📌 Dataset Information:")
            buffer = io.StringIO()
            df.info(buf=buffer)
            st.text(buffer.getvalue())

        selected_column = st.selectbox("Select a column to check value counts", df.columns)
        if selected_column:
            st.write(f"📊 Value Counts for {selected_column}:")
            st.write(df[selected_column].value_counts())

    # ⚙️ HANDLE MISSING VALUES
    elif preprocess == "⚙️ Handle Missing Values":
        df_missing = df.copy()

        for col in df_missing.columns:
            if df_missing[col].isnull().sum() > 0:
                # Categorical Columns
                if df_missing[col].dtype == 'object':  
                    option = st.selectbox(f"Fill NaN values in {col}", ("Specific Value", "Drop"), key=col)
                    if option == "Specific Value":
                        value = st.text_input(f"Enter specific value for {col}")
                        if value:
                            df_missing[col].fillna(value, inplace=True)
                    elif option == "Drop":
                        df_missing.dropna(subset=[col], inplace=True)

                # Numerical Columns
                else:  
                    option = st.selectbox(f"Fill NaN values in {col}", 
                                        ("Mean", "Median", "Mode", "Interquartile Range (IQR)"), key=col)
                    if option == "Mean":
                        df_missing[col].fillna(df_missing[col].mean(), inplace=True)
                    elif option == "Median":
                        df_missing[col].fillna(df_missing[col].median(), inplace=True)
                    elif option == "Mode":
                        df_missing[col].fillna(df_missing[col].mode()[0], inplace=True)
                    elif option == "Interquartile Range (IQR)":
                        Q1 = df_missing[col].quantile(0.25)
                        Q3 = df_missing[col].quantile(0.75)
                        IQR = Q3 - Q1
                        df_missing[col].fillna(df_missing[col].median(), inplace=True)  

        st.success("✅ Missing values handled successfully!")
        st.write("📌 Updated Dataset Preview After Handling Missing Values:")
        st.dataframe(df_missing.head())

        # Store in session state to keep modifications
        st.session_state["df_processed"] = df_missing

    # 🔎 INCONSISTENCY HANDLING
    elif preprocess == "Inconsistency":
        # Drop Unnecessary Columns
        st.header("🗑️ Drop Unnecessary Columns")
        columns_to_drop = st.multiselect("Select columns to drop", df.columns)
        if columns_to_drop:
            df.drop(columns=columns_to_drop, inplace=True)
            st.success(f"✅ Dropped columns: {', '.join(columns_to_drop)}")
            st.write("📌 Updated Dataset After Dropping Columns:")
            st.dataframe(df.head())

        inconsistency_option = st.radio("Choose inconsistency handling:", ("Filter/Search", "Replace Values"))

        if inconsistency_option == "Filter/Search":
            col1 , col2= st.columns(2)
            with col1:
                selected_col = st.selectbox("🔎 Select Column to Search/Filter", df.columns)
            with col2:
                search_value = st.text_input("🔍 Enter value to search")

            if search_value:
                filtered_df = df[df[selected_col].astype(str).str.contains(search_value, case=False, na=False)]
                st.write(f"📌 **Filtered Results for '{search_value}' in {selected_col}:**")
                st.dataframe(filtered_df)

        elif inconsistency_option == "Replace Values":
            selected_col = st.selectbox("🛠 Select Column to Replace Values", df.columns)
            col1 , col2= st.columns(2)
            with col1:
                old_value = st.text_input("✏️ Enter value to replace")
            with col2:
                new_value = st.text_input("✅ Enter new value")

            if old_value and new_value:
                df[selected_col] = df[selected_col].replace(old_value, new_value)
                st.success(f"✅ '{old_value}' replaced with '{new_value}' in column {selected_col}")
                st.write("📌 Updated Dataset Preview:")
                st.dataframe(df.head())

        # Store in session state to keep modifications
        st.session_state["df_processed"] = df

            # 📈 EDA (Exploratory Data Analysis)
    elif preprocess == "📈 EDA":
        st.header("📊 Exploratory Data Analysis")

        if "eda_plots" not in st.session_state:
            st.session_state.eda_plots = []

        # Select Univariate or Multivariate Analysis
        analysis_type = st.radio("📊 Select Analysis Type", ["Univariate Analysis", "Multivariate Analysis"], key=f"eda_type_{len(st.session_state.eda_plots)}")

        # Univariate Analysis
        if analysis_type == "Univariate Analysis":
            selected_column = st.selectbox("📌 Select Column for Univariate Analysis", df.columns, key=f"univar_col_{len(st.session_state.eda_plots)}")
            chart_option = st.selectbox("📊 Select Chart Type", ["Histogram", "Box Plot", "Bar Chart"], key=f"univar_chart_{len(st.session_state.eda_plots)}")

            if st.button("➕ Add Univariate Plot", key=f"add_univar_{len(st.session_state.eda_plots)}"):
                st.session_state.eda_plots.append(("Univariate", selected_column, chart_option))

        # Multivariate Analysis
        elif analysis_type == "Multivariate Analysis":
            selected_columns = st.multiselect("📌 Select Columns for Multivariate Analysis", df.columns, key=f"multi_col_{len(st.session_state.eda_plots)}")
            multi_chart_option = st.selectbox("📊 Select Multivariate Chart Type", ["Scatter Plot", "Line Chart", "Pair Plot", "Heatmap"], key=f"multi_chart_{len(st.session_state.eda_plots)}")

            if st.button("➕ Add Multivariate Plot", key=f"add_multi_{len(st.session_state.eda_plots)}"):
                st.session_state.eda_plots.append(("Multivariate", selected_columns, multi_chart_option))

        # 🔥 Display All Selected Plots Separately
        for idx, (analysis, cols, chart) in enumerate(st.session_state.eda_plots):
            st.subheader(f"📊 {chart} for {cols}")

            fig, ax = plt.subplots()
            
            if analysis == "Univariate":
                if chart == "Histogram":
                    sns.histplot(df[cols], kde=True, ax=ax)
                elif chart == "Box Plot":
                    sns.boxplot(x=df[cols], ax=ax)
                elif chart == "Bar Chart":
                    df[cols].value_counts().plot(kind="bar", ax=ax)

            elif analysis == "Multivariate":
                if chart == "Scatter Plot":
                    if len(cols) >= 2:
                        sns.scatterplot(x=df[cols[0]], y=df[cols[1]], ax=ax)
                    else:
                        st.warning("⚠️ Please select at least two columns for a Scatter Plot.")
                elif chart == "Line Chart":
                    if len(cols) >= 2:
                        sns.lineplot(x=df[cols[0]], y=df[cols[1]], ax=ax)
                    else:
                        st.warning("⚠️ Please select at least two columns for a Line Chart.")
                elif chart == "Pair Plot":
                    if len(cols) > 1:
                        fig = sns.pairplot(df[cols])
                    else:
                        st.warning("⚠️ Please select at least two columns for a Pair Plot.")
                elif chart == "Heatmap":
                    if len(cols) > 1:
                        sns.heatmap(df[cols].corr(), annot=True, cmap="coolwarm", ax=ax)
                    else:
                        st.warning("⚠️ Please select at least two columns for a Heatmap.")

            st.pyplot(fig)



    # FEATURE ENGINEERING
    elif preprocess == "Feature Eng":
        df_feature_eng = st.session_state.get("df_processed", df).copy()

        # Select Features & Target
        target_column = st.selectbox("🎯 Select Target Column", df_feature_eng.columns)
        feature_columns = st.multiselect("📊 Select Feature Columns", df_feature_eng.columns, 
                                         default=[col for col in df_feature_eng.columns if col != target_column])

        if feature_columns and target_column:
            X = df_feature_eng[feature_columns].copy()
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
                X = pd.get_dummies(X, columns=cat_features).replace({True:1,False:0})

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

        # Merge Processed Features
        df_feature_eng = df_feature_eng.drop(columns=feature_columns, errors='ignore')
        df_feature_eng = pd.concat([df_feature_eng, X], axis=1)

        st.write("📌 Updated Dataset After Feature Engineering:")
        st.dataframe(df_feature_eng.head())

        # Store df_feature_eng in session_state for download
        st.session_state["df_feature_eng"] = df_feature_eng

    # 📥 DOWNLOAD PROCESSED DATASET
    if "df_feature_eng" in st.session_state:
        df_to_download = st.session_state["df_feature_eng"]
        csv_buffer = io.StringIO()
        df_to_download.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()

        st.header("📥 Download Processed Dataset")
        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name="processed_dataset.csv",
            mime="text/csv"
        )

        st.success("✅ Processed dataset is ready for download!")
