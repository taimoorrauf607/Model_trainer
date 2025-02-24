import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.ensemble import StackingRegressor, StackingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 📊 Univariate Analysis Function
def univariate_analysis(df, selected_column, chart_option):
    fig, ax = plt.subplots()
    if chart_option == "Histogram":
        sns.histplot(df[selected_column], kde=True, ax=ax)
        plt.xticks(rotation='vertical')
    elif chart_option == "Box Plot":
        sns.boxplot(x=df[selected_column], ax=ax)
    elif chart_option == "Bar Chart":
        df[selected_column].value_counts().plot(kind="bar", ax=ax)
        plt.xticks(rotation='vertical')
    st.pyplot(fig)

# 📊 Bivariate Analysis Function
def bivariate_analysis(df, x_axis, y_axis, chart_option):
    fig, ax = plt.subplots()
    if chart_option == "Scatter Plot":
        sns.scatterplot(x=df[x_axis], y=df[y_axis], ax=ax)
        plt.xticks(rotation='vertical')
    elif chart_option == "Line Chart":
        sns.lineplot(x=df[x_axis], y=df[y_axis], ax=ax)
        plt.xticks(rotation='vertical')
    st.pyplot(fig)

# 📊 Multivariate Analysis Function
def multivariate_analysis(df, x_axis, y_axis, hue, chart_option):
    fig, ax = plt.subplots()
    if chart_option == "Scatter Plot":
        sns.scatterplot(x=df[x_axis], y=df[y_axis], hue=df[hue], ax=ax)
        plt.xticks(rotation='vertical')
    elif chart_option == "Line Chart":
        sns.lineplot(x=df[x_axis], y=df[y_axis], hue=df[hue], ax=ax)
        plt.xticks(rotation='vertical')
    st.pyplot(fig)

# 🛠️ Categorical Feature Encoding Function
def encode_categorical_features(df, cat_features, encoding_method):
    if encoding_method == "Label Encoding":
        label_encoders = {}
        for col in cat_features:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    elif encoding_method == "One-Hot Encoding":
        df = pd.get_dummies(df, columns=cat_features).replace({True: 1, False: 0})
    return df

# 📏 Feature Scaling Function
def scale_features(df, scale_features):
    scaler = StandardScaler()
    df[scale_features] = scaler.fit_transform(df[scale_features])
    return df


# 🔄 Function to Split Data
def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

# 🌟 Function to Train Regression Models
def train_regression_model(model_name, X_train, y_train, X_test, y_test, params):
    model = None
    if model_name == "CatBoost":
        model = CatBoostRegressor(**params, verbose=0)
        model.fit(X_train, y_train, early_stopping_rounds=50, verbose=False)
        y_pred = model.predict(X_test)
        metric = np.sqrt(mean_squared_error(y_test, y_pred))
    
    elif model_name == "Stacking":
        base_learners = [
            ("lr", LinearRegression()),
            ("dt", DecisionTreeRegressor()),
            ("rf", RandomForestRegressor())
        ]
        meta_learner = LinearRegression()
        model = StackingRegressor(estimators=base_learners, final_estimator=meta_learner)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metric = np.sqrt(mean_squared_error(y_test, y_pred))
    
    return model, metric

# 🌟 Function to Train Classification Models
def train_classification_model(model_name, X_train, y_train, X_test, y_test, params):
    model = None
    if model_name == "CatBoost":
        model = CatBoostClassifier(**params, verbose=0)
        model.fit(X_train, y_train, early_stopping_rounds=50, verbose=False)
        y_pred = model.predict(X_test)
        metric = accuracy_score(y_test, y_pred)
    
    elif model_name == "Stacking":
        base_classifiers = [
            ("lr", LogisticRegression()),
            ("dt", DecisionTreeClassifier()),
            ("rf", RandomForestClassifier())
        ]
        meta_classifier = LogisticRegression()
        model = StackingClassifier(estimators=base_classifiers, final_estimator=meta_classifier)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metric = accuracy_score(y_test, y_pred)
    
    return model, metric
