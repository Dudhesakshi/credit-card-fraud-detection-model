# --------------------------------------
# ✅ Step 1: Import Required Libraries
# --------------------------------------

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

# --------------------------------------
# ✅ Step 2: Define All Helper Functions
# --------------------------------------

def load_data(path):
    df = pd.read_csv(path)
    return df

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

def train_model(X_train, y_train):
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=2, random_state=42)
    rf.fit(X_train, y_train)
    return rf

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show(block=False)
    plt.pause(2)
    plt.close()

    # ROC Curve
    y_prob = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_prob)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show(block=False)
    plt.pause(2)
    plt.close()

def plot_feature_importance(model, feature_names):
    importance = model.feature_importances_
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, importance)
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.title("Feature Importance Plot")
    plt.show(block=False)
    plt.pause(2)
    plt.close()

def compare_models(X_train, y_train, X_test, y_test):
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42)
    }

    scores = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        f1 = classification_report(y_test, y_pred, output_dict=True)['weighted avg']['f1-score']
        scores[name] = {
            'ROC AUC': roc_auc,
            'F1-Score': f1
        }
        print(f"\n{name} Performance:")
        print(classification_report(y_test, y_pred))
        
    # Plotting
    roc_auc_scores = [v['ROC AUC'] for v in scores.values()]
    f1_scores = [v['F1-Score'] for v in scores.values()]
    model_names = list(scores.keys())

    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width/2, roc_auc_scores, width, label='ROC AUC')
    rects2 = ax.bar(x + width/2, f1_scores, width, label='F1 Score')

    ax.set_ylabel('Scores')
    ax.set_title('Model Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()

    plt.show(block=False)
    plt.pause(2)
    plt.close()

    return scores

# --------------------------------------
# ✅ Step 3: Run the Full Pipeline
# --------------------------------------

if __name__ == "__main__":
    file_path = "C:/Users/HP/OneDrive/Desktop/CreditCardFault/creditcard.csv"
    df = load_data(file_path)
    df = df.sample(frac=0.1, random_state=42)  # Use 10% of the dataset

    df = preprocess_data(df)
    df = engineer_features(df)

    X_train, X_test, y_train, y_test = split_and_balance(df)

    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)

    features = df.drop('Class', axis=1).columns
    plot_feature_importance(model, features)

    # Compare multiple models
    compare_models(X_train, y_train, X_test, y_test)
