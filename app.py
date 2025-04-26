# ---------------------------------
# ✅ Step 1: Import Libraries
# ---------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings('ignore')

# ---------------------------------
# ✅ Step 2: Define Helper Functions
# ---------------------------------

def preprocess_data(df):
    df['Hour'] = (df['Time'] // 3600) % 24
    df = df.drop(columns=['Time'])
    scaler = StandardScaler()
    df[['Amount']] = scaler.fit_transform(df[['Amount']])
    return df

def engineer_features(df):
    df['Rolling_Mean_Amount'] = df['Amount'].rolling(window=10).mean().fillna(0)
    df['Amount_Diff'] = df['Amount'].diff().fillna(0)
    return df

def split_and_balance(df):
    X = df.drop('Class', axis=1)
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    sm = SMOTE(random_state=42)
    X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
    return X_train_sm, X_test, y_train_sm, y_test

def train_model(X_train, y_train, model_type='RandomForest'):
    if model_type == 'RandomForest':
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    elif model_type == 'XGBoost':
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    elif model_type == 'LogisticRegression':
        model = LogisticRegression(max_iter=1000, random_state=42)
    else:
        raise ValueError("Unsupported model type.")
    
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

    # ROC Curve
    st.subheader("ROC Curve")
    y_prob = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_prob)
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    fig2, ax2 = plt.subplots()
    ax2.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    ax2.plot([0, 1], [0, 1], 'k--')
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve")
    ax2.legend()
    st.pyplot(fig2)

# ---------------------------------
# ✅ Step 3: Streamlit App
# ---------------------------------

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title("🚀 Credit Card Fraud Detection App")

uploaded_file = st.file_uploader("📄 Upload your Credit Card CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    st.subheader("🔍 Sample Data")
    st.dataframe(df.head())

    model_choice = st.selectbox(
        "Choose a Model to Train",
        ("RandomForest", "XGBoost", "LogisticRegression")
    )

    if st.button("🚀 Start Processing & Training"):
        with st.spinner("Processing your data..."):

            # Preprocess
            df = preprocess_data(df)
            df = engineer_features(df)

            # Split and Balance
            X_train, X_test, y_train, y_test = split_and_balance(df)

            # Train
            model = train_model(X_train, y_train, model_type=model_choice)

            # Evaluate
            evaluate_model(model, X_test, y_test)

            st.success("✅ Processing & Training Completed!")

else:
    st.info("👆 Please upload a CSV file to get started.")

