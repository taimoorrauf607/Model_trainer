import streamlit as st
import io
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import io,re, sys  # For file handling
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

    if "df_processed" not in st.session_state:
        st.session_state["df_processed"] = df.copy()
    df = st.session_state["df_processed"]

    # Sidebar - Preprocessing Steps
    preprocess_options = ["📊 Data Overview","🔎 Inconsistency","custom code", "📈 EDA", "⚙️ Feature Eng"]

    if df.isnull().sum().sum() > 0:
        preprocess_options.insert(1, "⚙️ Handle Missing Values")

    preprocess = st.sidebar.radio("Choose Step", preprocess_options)

    # 📊 DATA OVERVIEW
    if preprocess == "📊 Data Overview":
        st.title("Welcome to Data Preprocessing")
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
        st.title("Welcome to Data Preprocessing")
        df_missing = df.copy()
        for col in df_missing.columns:
            if df_missing[col].isnull().sum() > 0:
                st.subheader(f"⚠️ Handling Missing Values in **{col}**")
                # Categorical Columns
                if df_missing[col].dtype == 'object':  
                    option = st.selectbox(f"Select option to Fill:", ("Specific Value", "Drop"), key=col)
                    if option == "Specific Value":
                        value = st.text_input(f"Enter specific value for {col}", key=f"value_{col}")
                        if value and st.button(f"✅ Confirm", key=f"confirm_{col}"):
                            df_missing[col].fillna(value, inplace=True)
                            st.success(f"✅ Filled NaN values in **{col}** with **{value}**")
                    elif option == "Drop":
                        if st.button(f"✅ Confirm Drop NaNs in {col}", key=f"drop_{col}"):
                            df_missing.dropna(subset=[col], inplace=True)
                            st.success(f"✅ Dropped rows with NaN in **{col}**")

                # Numerical Columns
                else:  
                    option = st.selectbox(f"Select option to Fill:", 
                                        ("Mean", "Median", "Mode", "Interquartile Range (IQR)","Specific Value"), key=col)
                    if option == "Specific Value":
                        value = st.text_input(f"Enter specific value for {col}", key=f"num_value_{col}")
                        if value and st.button(f"✅ Confirm", key=f"confirm_num_value_{col}"):
                            df_missing[col].fillna(float(value), inplace=True)
                            st.success(f"✅ Filled NaN values in **{col}** with **{value}**")
                    elif st.button(f"✅ Confirm", key=f"confirm_num_{col}"):
                        if option == "Mean":
                            df_missing[col].fillna(df_missing[col].mean(), inplace=True)
                            st.success(f"✅ Filled NaN values in **{col}** with Mean")
                        elif option == "Median":
                            df_missing[col].fillna(df_missing[col].median(), inplace=True)
                            st.success(f"✅ Filled NaN values in **{col}** with Median")
                        elif option == "Mode":
                            df_missing[col].fillna(df_missing[col].mode()[0], inplace=True)
                            st.success(f"✅ Filled NaN values in **{col}** with Mode")
                        elif option == "Interquartile Range (IQR)":
                            Q1 = df_missing[col].quantile(0.25)
                            Q3 = df_missing[col].quantile(0.75)
                            IQR = Q3 - Q1
                            df_missing[col].fillna(df_missing[col].median(), inplace=True)  
                            st.success(f"✅ Filled NaN values in **{col}** using IQR")

        st.write("📌 Updated Dataset Preview After Handling Missing Values:")
        st.dataframe(df_missing.head())

        # Store in session state to keep modifications
        st.session_state["df_processed"] = df_missing


       # 🔎 INCONSISTENCY HANDLING
    elif preprocess == "🔎 Inconsistency":
        st.title("Welcome to Data Preprocessing")
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

        # Choose Inconsistency Handling Method
        inconsistency_option = st.radio("📌 Choose Inconsistency Handling:", ("Data Type Conversion", "Filter/Search", "Replace Values", "Remove Char"))

        # Data Type Conversion Handling
        if inconsistency_option == "Data Type Conversion":
            with st.form("data_type_conversion_form"):
                column_to_convert = st.selectbox("📌 Select Column to Change Data Type", df.columns, key="inconsistency_col")
                new_dtype = st.selectbox("🔄 Convert Data Type To", ["int", "float", "string", "date"], key="dtype_selection")
                date_component = None
                if new_dtype == "date":
                    date_component = st.radio("📌 Extract Date Component", ["Full Date", "Year", "Month", "Day"], horizontal=True, key="date_component")
                submit_button = st.form_submit_button("✅ Convert Data Type")

            if submit_button:
                if new_dtype == "int":
                    df[column_to_convert] = pd.to_numeric(df[column_to_convert], errors="coerce").astype("Int64")
                elif new_dtype == "float":
                    df[column_to_convert] = pd.to_numeric(df[column_to_convert], errors="coerce").astype("float")
                elif new_dtype == "string":
                    df[column_to_convert] = df[column_to_convert].astype("string")
                elif new_dtype == "date":
                    df[column_to_convert] = pd.to_datetime(df[column_to_convert], errors="coerce")
                    if date_component == "Year":
                        df[column_to_convert] = df[column_to_convert].dt.year
                    elif date_component == "Month":
                        df[column_to_convert] = df[column_to_convert].dt.month
                    elif date_component == "Day":
                        df[column_to_convert] = df[column_to_convert].dt.day

                st.success(f"✅ Column **{column_to_convert}** converted to **{new_dtype}**")
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("📌 Updated Data Types")
                    st.write(df.dtypes)
                with col2:
                    st.subheader("📊 Updated Data Preview")
                    st.dataframe(df)

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
                st.dataframe(df.head())

        elif inconsistency_option == "Remove Char":
            st.dataframe(df)
            selected_col = st.selectbox("🛠 Select Column to remove Values", df.columns)
            col1, col2 = st.columns(2)
            with col1:
                old_vl = st.text_input("✏️ Enter value to remove:")
            with col2:
                exp = st.text_input("Enter Regular expression:")

            if old_vl or exp:
                df[selected_col] = df[selected_col].apply(lambda x: np.nan if str(x) == old_vl or (exp and re.fullmatch(exp, str(x))) 
                                                        else str(x).replace(old_vl, '').strip() if old_vl 
                                                        else re.sub(exp, '', str(x)).strip())
                st.success(f"✅ '{old_vl or exp}' removed from column {selected_col}")
                st.dataframe(df)
                st.write("📌 Updated Dataset Preview:")


    elif preprocess == "custom code":
        st.title("Welcome to Data Preprocessing")
        st.dataframe(df)
        st.subheader("🧑‍💻 Write Your Custom Data Cleaning Code")
        code_template = """
        # Example:
        # df['column_name'] = df['column_name'].str.strip()
        # df['date_column'] = pd.to_datetime(df['date_column'], errors='coerce')
        """
        custom_code = st.text_area("Write your Python code here:", code_template, height=150)
        if st.button("🚀 Run Custom Code"):
            try:
                exec(custom_code, {"df": df, "pd": pd})
                st.success("✅ Custom code executed successfully!")
                st.dataframe(df.head())
            except Exception as e:
                st.error(f"❌ Error executing custom code: {e}")

        st.session_state.df = df
        st.session_state["df_processed"] = df

    # **📊 EDA Section (Only Runs if `preprocess == "EDA"`)**
    elif preprocess == "📈 EDA":
        st.title("Welcome to Data Preprocessing")
       # **📊 EDA Section (Only Runs if `preprocess == "EDA"`)**
        st.header("📊 Exploratory Data Analysis")

        # Initialize session state for EDA plots
        if "eda_plots" not in st.session_state:
            st.session_state.eda_plots = []

        # **Step 1: Select Analysis Type**
        analysis_type = st.radio("📊 Select Analysis Type", ["Univariate Analysis","Bi-variate Analysis" ,"Multivariate Analysis"], key="eda_type")

        # **Univariate Analysis**
        if analysis_type == "Univariate Analysis":
            col1, col2 = st.columns(2)
            with col1:
                selected_column = st.selectbox("📌 Select Column", df.columns, key="univar_col")
            with col2:
                chart_option = st.selectbox("📊 Select Chart Type", ["Histogram", "Box Plot", "Bar Chart","Violin Plot","Pie Chart","Density Plot"], key="univar_chart")

            if st.button("➕ Add Univariate Plot", key="add_univar"):
                st.session_state.eda_plots.append(("Univariate", selected_column, chart_option))

        # **bivariate Analysis**
        elif analysis_type == "Bi-variate Analysis":
            col1 ,col2 = st.columns(2)
            with col1:
                x_axis = st.selectbox("📌 Select X-Axis", df.columns, key="bi_x")
            with col2:
                y_axis = st.selectbox("📌 Select Y-Axis", df.columns, key="bi_y")
            multi_chart_option = st.selectbox("📊 Select bivariate Chart Type", ["Scatter Plot", "Line Chart","Heatmap","Joint Plot","Reg Plot","Hexbin Plot"], key="bi_chart")

            if st.button("➕ Add Bi-variate Plot", key="add_bi"):
                st.session_state.eda_plots.append(("bivariate", x_axis, y_axis,multi_chart_option))
        
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
                elif chart == "Violin Plot":
                    sns.violinplot(x=df[col], ax=ax)
                elif chart == "Pie Chart":
                    df[col].value_counts().plot(kind="pie", autopct='%1.1f%%', ax=ax)
                    plt.ylabel('')
                elif chart == "Density Plot":
                    sns.kdeplot(df[col], ax=ax)

            # Handle Bivariate Analysis
            elif plot[0] == "bivariate":
                analysis, x_col, y_col, chart = plot  # Unpack 4 elements
                st.subheader(f"📊 {chart} for {x_col} vs {y_col}")

                if chart == "Scatter Plot":
                    sns.scatterplot(x=df[x_col], y=df[y_col], ax=ax)
                    plt.xticks(rotation='vertical')
                elif chart == "Line Chart":
                    sns.lineplot(x=df[x_col], y=df[y_col], ax=ax)
                    plt.xticks(rotation='vertical')
                elif chart == "Heatmap":
                    sns.heatmap(df[[x_col, y_col]].corr(), annot=True, cmap='coolwarm', ax=ax)
                elif chart == "Hexbin Plot":
                    if df[x_col].dtype in ['int64', 'float64'] and df[y_col].dtype in ['int64', 'float64']:
                        ax.hexbin(df[x_col], df[y_col], gridsize=30, cmap='Blues')
                elif chart == "Reg Plot":
                    sns.regplot(x=df[x_col], y=df[y_col], ax=ax)
                elif chart == "Joint Plot":
                    sns.jointplot(x=df[x_col], y=df[y_col], kind='scatter')
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
    elif preprocess == "⚙️ Feature Eng":
        st.title("Welcome to Data Preprocessing")
        df_feature_eng = st.session_state.get("df_processed", df).copy()

        # Select Features to Encode
        st.header("🛠️ Encode Categorical Features")
        cat_features = st.multiselect("📊 Select Features to Encode", 
                                      [col for col in df_feature_eng.columns if df_feature_eng[col].dtype == 'object'])
        
        # Encoding Categorical Features
        if cat_features:
            encoding_method = st.selectbox("Select Encoding Method", ("Label Encoding", "One-Hot Encoding"))
            encode_button = st.button("✅Confirm",key=f"{cat_features} is encoded✅")
            if encode_button:
                if encoding_method == "Label Encoding":
                    label_encoders = {}
                    for col in cat_features:
                        le = LabelEncoder()
                        df_feature_eng[col] = le.fit_transform(df_feature_eng[col].astype(str))
                        label_encoders[col] = le

                elif encoding_method == "One-Hot Encoding":
                    df_feature_eng = pd.get_dummies(df_feature_eng, columns=cat_features).replace({True:1,False:0})

                st.success("✅ Categorical features encoded successfully!")
                st.write("📌 Updated Dataset After Encoding:")
                st.dataframe(df_feature_eng.head())

        # Feature Scaling
        st.header("📏 Feature Scaling")
        scale_features = st.multiselect("Select Features to Scale", df_feature_eng.columns)
        scale_button = st.button("✅Confirm",key=f"{scale_features} is scaled✅")
        if scale_button:
            if scale_features:
                scaler = StandardScaler()
                df_feature_eng[scale_features] = scaler.fit_transform(df_feature_eng[scale_features])
                st.write(f"✅ Scaled Features: {', '.join(scale_features)}")

                st.success("✅ Feature scaling applied successfully!")
                st.write("📌 Updated Dataset After Feature Scaling:")
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
         # 💾 Store Updated DataFrame in Session State
        st.session_state.df = df
        # Store df_feature_eng in session_state for download
        st.session_state["df_feature_eng"] = df_feature_eng