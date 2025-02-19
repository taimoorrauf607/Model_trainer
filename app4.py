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
    st.title("Welcome to Data Preprocessing")

    if "df_processed" not in st.session_state:
        st.session_state["df_processed"] = df.copy()
    df = st.session_state["df_processed"]

    # Sidebar - Preprocessing Steps
    preprocess_options = ["📊 Data Overview", "Inconsistency", "EDA", "Feature Eng", "Train Model only"]
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
        drop_button = st.button("Drop Column")
        if drop_button:
            if columns_to_drop:
                df.drop(columns=columns_to_drop, inplace=True)
                st.success(f"✅ Dropped columns: {', '.join(columns_to_drop)}")
                st.write("📌 Updated Dataset After Dropping Columns:")
                st.dataframe(df.head())
            # **Choose Inconsistency Handling Method**
        inconsistency_option = st.radio("📌 Choose Inconsistency Handling:", ("Data Type Conversion", "Filter/Search", "Replace Values"))

        # **Data Type Conversion Handling**
        if inconsistency_option == "Data Type Conversion":
            with st.form("data_type_conversion_form"):
                # **Column Selection**
                column_to_convert = st.selectbox("📌 Select Column to Change Data Type", df.columns, key="inconsistency_col")

                # **Data Type Selection**
                new_dtype = st.selectbox("🔄 Convert Data Type To", ["int", "float", "string", "date"], key="dtype_selection")

                # **Date Component Selection (Only when 'date' is selected)**
                date_component = None
                if new_dtype == "date":
                    date_component = st.radio("📌 Extract Date Component", ["Full Date", "Year", "Month", "Day"], horizontal=True, key="date_component")

                # **Submit Button**
                submit_button = st.form_submit_button("✅ Convert Data Type")

            # **Process conversion after submit button is clicked**
            if submit_button:
                if new_dtype == "int":
                    df[column_to_convert] = pd.to_numeric(df[column_to_convert], errors="coerce").astype("Int64")

                elif new_dtype == "float":
                    df[column_to_convert] = pd.to_numeric(df[column_to_convert], errors="coerce").astype("float")

                elif new_dtype == "string":
                    df[column_to_convert] = df[column_to_convert].astype("string")

                elif new_dtype == "date":
                    df[column_to_convert] = pd.to_datetime(df[column_to_convert], errors="coerce")

                    # **Extract Date Components if selected**
                    if date_component == "Year":
                        df[column_to_convert] = df[column_to_convert].dt.year
                    elif date_component == "Month":
                        df[column_to_convert] = df[column_to_convert].dt.month
                    elif date_component == "Day":
                        df[column_to_convert] = df[column_to_convert].dt.day

                st.success(f"✅ Column **{column_to_convert}** converted to **{new_dtype}**")
                # **Display Updated Data**
                col1,col2 = st.columns(2)
                with col1:
                    st.subheader("📌 Updated Data Types")
                    st.write(df.dtypes)
                with col2:
                    st.subheader("📊 Updated Data Preview")
                    st.dataframe(df)

        # **Filter/Search Handling**
        elif inconsistency_option == "Filter/Search":
            col1, col2 = st.columns(2)
            with col1:
                selected_col = st.selectbox("🔎 Select Column to Search/Filter", df.columns)
            with col2:           
                search_value = st.text_input("🔍 Enter value to search")

            if search_value:
                filtered_df = df[df[selected_col].astype(str).str.contains(search_value, case=False, na=False)]
                st.write(f"📌 **Filtered Results for '{search_value}' in {selected_col}:**")
                st.dataframe(filtered_df)

        # **Replace Values Handling**
        elif inconsistency_option == "Replace Values":
            selected_col = st.selectbox("🛠 Select Column to Replace Values", df.columns)
            col1, col2 = st.columns(2)
            with col1:
                old_value = st.text_input("✏️ Enter value to replace")
            with col2:
                new_value = st.text_input("✅ Enter new value")
            
            if old_value and new_value:
                df[selected_col] = df[selected_col].replace(old_value, new_value)
                st.success(f"✅ '{old_value}' replaced with '{new_value}' in column {selected_col}")
                # st.write("📌 Updated Dataset Preview:")
                st.dataframe(df.head())

        # Store Updated DataFrame
        st.session_state.df = df
    
        
        

        # Store in session state to keep modifications
        st.session_state["df_processed"] = df

    # **📊 EDA Section (Only Runs if `preprocess == "EDA"`)**
    elif preprocess == "EDA":
       # **📊 EDA Section (Only Runs if `preprocess == "EDA"`)**
        st.header("📊 Exploratory Data Analysis")

        # Initialize session state for EDA plots
        if "eda_plots" not in st.session_state:
            st.session_state.eda_plots = []

        # **Step 1: Select Analysis Type**
        analysis_type = st.radio("📊 Select Analysis Type", ["Univariate Analysis", "Multivariate Analysis"], key="eda_type")

        # **Univariate Analysis**
        if analysis_type == "Univariate Analysis":
            col1, col2 = st.columns(2)
            with col1:
                selected_column = st.selectbox("📌 Select Column", df.columns, key="univar_col")
            with col2:
                chart_option = st.selectbox("📊 Select Chart Type", ["Histogram", "Box Plot", "Bar Chart"], key="univar_chart")

            if st.button("➕ Add Univariate Plot", key="add_univar"):
                st.session_state.eda_plots.append(("Univariate", selected_column, chart_option))

        # **Multivariate Analysis**
        elif analysis_type == "Multivariate Analysis":
            col1 ,col2 = st.columns(2)
            with col1:
                x_axis = st.selectbox("📌 Select X-Axis", df.columns, key="multi_x")
                hue = st.selectbox("📌 Select Hue column", df.columns, key="multi_hue")
            with col2:
                y_axis = st.selectbox("📌 Select Y-Axis", df.columns, key="multi_y")
            multi_chart_option = st.selectbox("📊 Select Multivariate Chart Type", ["Scatter Plot", "Line Chart"], key="multi_chart")

            if st.button("➕ Add Multivariate Plot", key="add_multi"):
                st.session_state.eda_plots.append(("Multivariate", x_axis, y_axis, hue,multi_chart_option))

        # **Step 3: Display All Selected Plots**
        for idx, plot in enumerate(st.session_state.eda_plots):
            fig, ax = plt.subplots()

            # Handle Univariate Analysis
            if plot[0] == "Univariate":
                analysis, col, chart = plot  # Unpack only 3 elements
                st.subheader(f"📊 {chart} for {col}")

                if chart == "Histogram":
                    sns.histplot(df[col], kde=True, ax=ax)
                    plt.xticks(rotation='vertical')
                elif chart == "Box Plot":
                    sns.boxplot(x=df[col], ax=ax)
                elif chart == "Bar Chart":
                    df[col].value_counts().plot(kind="bar", ax=ax)
                    plt.xticks(rotation='vertical')

            # Handle Multivariate Analysis
            elif plot[0] == "Multivariate":
                analysis, x_col, y_col,hue,chart = plot  # Unpack 4 elements
                st.subheader(f"📊 {chart} for {x_col} vs {y_col}")

                if chart == "Scatter Plot":
                    sns.scatterplot(x=df[x_col], y=df[y_col], ax=ax,hue=df[hue])
                    plt.xticks(rotation='vertical')
                elif chart == "Line Chart":
                    sns.lineplot(x=df[x_col], y=df[y_col], ax=ax,hue=df[hue])
                    plt.xticks(rotation='vertical')

            st.pyplot(fig)


        # **Refresh Button**
        if st.button("🔄 Refresh EDA"):
            st.session_state.eda_plots = []  # Clear session state
            st.rerun()



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
