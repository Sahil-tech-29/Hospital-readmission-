import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, roc_auc_score,
                             confusion_matrix, average_precision_score)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
import os

os.makedirs("model", exist_ok=True)


def train(data_path='data/preprocessed_data.csv'):
    print("Loading processed data...")
    df = pd.read_csv(data_path)

    X = df.drop(columns=['readmitted_30'])
    y = df['readmitted_30']

    print(f"  Features: {X.shape[1]}  |  Samples: {len(X)}")
    print(f"  Class balance — 0: {(y==0).sum()}  1: {(y==1).sum()}")

    # Save feature names for SHAP later
    joblib.dump(list(X.columns), 'model/feature_names.pkl')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    joblib.dump(scaler, 'model/scaler.pkl')

    # SMOTE on training set only
    print("\nApplying SMOTE...")
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train_sc, y_train)
    print(f"  After SMOTE — 0: {(y_res==0).sum()}  1: {(y_res==1).sum()}")

    # XGBoost
    print("\nTraining XGBoost...")
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_res, y_res,
              eval_set=[(X_test_sc, y_test)],
              verbose=50)

    joblib.dump(model, 'model/model.pkl')
    print("\nModel saved -> model/model.pkl")

    # Evaluate
    y_pred  = model.predict(X_test_sc)
    y_proba = model.predict_proba(X_test_sc)[:, 1]

    auc  = roc_auc_score(y_test, y_proba)
    ap   = average_precision_score(y_test, y_proba)
    cm   = confusion_matrix(y_test, y_pred)

    print("\n=== Evaluation ===")
    print(f"  ROC-AUC : {auc:.4f}")
    print(f"  Avg Prec: {ap:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not Readmitted', 'Readmitted <30d']))

    return model, scaler


if __name__ == '__main__':
    train()