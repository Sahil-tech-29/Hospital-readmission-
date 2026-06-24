# Hospital Readmission Predictor: An Explainable Machine Learning System for 30-Day Readmission Risk Assessment

## Overview

Hospital readmissions within 30 days of discharge are a major challenge in healthcare, leading to increased treatment costs, resource utilization, and patient risk. This project leverages Machine Learning and Explainable AI to predict whether a diabetic patient is likely to be readmitted within 30 days of discharge.

The system uses the Diabetes 130-US Hospitals Dataset from the UCI Machine Learning Repository and employs an XGBoost classifier to generate readmission risk predictions. To improve transparency and trust, SHAP (SHapley Additive exPlanations) is integrated to explain the contribution of each feature to the final prediction.

The application is deployed through a Flask-based web interface where users can enter patient information, receive a risk prediction, and understand the key factors influencing the model's decision.

---

## Problem Statement

Hospital readmissions increase healthcare costs and negatively impact patient outcomes. Early identification of high-risk patients enables healthcare providers to implement preventive measures such as follow-up monitoring, medication management, and personalized care plans.

This project aims to build an explainable machine learning system that predicts 30-day hospital readmission risk and provides interpretable explanations for every prediction.

---

## Dataset

**Dataset:** Diabetes 130-US Hospitals Dataset

**Source:** UCI Machine Learning Repository

**Dataset Characteristics:**

* 101,766 patient encounters
* 130 hospitals across the United States
* Data collected between 1999 and 2008
* 50+ clinical and demographic features

### Key Features

* Race
* Gender
* Age
* Admission Type
* Time in Hospital
* Number of Lab Procedures
* Number of Medications
* Number of Diagnoses
* Number of Emergency Visits
* Number of Inpatient Visits
* Primary Diagnosis (ICD-9)
* Secondary Diagnosis (ICD-9)
* Tertiary Diagnosis (ICD-9)

### Target Variable

| Original Value | Target |
| -------------- | ------ |
| <30            | 1      |
| >30            | 0      |
| NO             | 0      |

Where:

* 1 = Readmitted within 30 days
* 0 = Not readmitted within 30 days

---

## Tech Stack

Python, Flask, Pandas, NumPy, Scikit-Learn, XGBoost, Imbalanced-Learn (SMOTE), SHAP, Matplotlib, SQLite, HTML, CSS, JavaScript, Chart.js, Joblib, Git, GitHub

---

## Project Architecture

```text
Dataset
   │
   ▼
Data Preprocessing
   │
   ▼
Feature Engineering
(ICD-9 Grouping)
   │
   ▼
Train-Test Split
   │
   ▼
SMOTE
   │
   ▼
XGBoost Classifier
   │
   ▼
SHAP Explainability
   │
   ▼
Flask Application
   │
   ▼
SQLite Database
   │
   ▼
Analytics Dashboard
```

---

## Data Preprocessing

The following preprocessing steps were performed:

* Replaced '?' values with NaN
* Removed highly sparse columns

  * weight
  * payer_code
  * medical_specialty
* Removed identifier columns

  * encounter_id
  * patient_nbr
* Converted age groups into numerical values
* Encoded medication-related features
* Grouped ICD-9 diagnosis codes into disease categories
* Label encoded categorical variables
* Handled missing values using median and mode imputation

---

## Feature Engineering

### ICD-9 Diagnosis Grouping

To reduce feature cardinality and improve interpretability, diagnosis codes were grouped into major disease categories:

* Circulatory
* Respiratory
* Digestive
* Diabetes
* Injury
* Musculoskeletal
* Genitourinary
* Neoplasms
* Other

---

## Class Imbalance Handling

The dataset contains approximately:

* 89% Non-readmitted patients
* 11% Readmitted patients

To address class imbalance, SMOTE (Synthetic Minority Over-sampling Technique) was applied exclusively to the training data.

---

## Model Development

### Algorithm Used

**XGBoost Classifier**

### Model Parameters

```python
XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
```

### Why XGBoost?

* Excellent performance on tabular healthcare data
* Handles non-linear relationships effectively
* Robust against overfitting
* Compatible with SHAP explainability
* Strong predictive performance compared to traditional models

---

## Evaluation Metrics

The model is evaluated using:

* ROC-AUC Score
* Average Precision Score
* Precision
* Recall
* F1 Score
* Confusion Matrix

### Why Not Accuracy?

Since the dataset is highly imbalanced, accuracy alone can be misleading. ROC-AUC and Precision-Recall metrics provide a more reliable evaluation of model performance.

---

## Explainable AI with SHAP

SHAP (SHapley Additive exPlanations) is used to explain model predictions.

### Features

* Patient-level explanations
* SHAP Waterfall Plot
* Feature contribution analysis
* Global feature importance ranking

Example:

```text
Prediction Risk: 82%

Top Contributors:
+ Previous Inpatient Visits
+ Advanced Age
+ High Medication Count

Risk Reducing Factors:
- Short Hospital Stay
- Fewer Emergency Visits
```

---

## Flask Application

### Main Routes

### Home Page

```text
/
```

Displays patient input form.

### Prediction Endpoint

```text
/ predict
```

Generates readmission prediction and SHAP explanation.

### Dashboard

```text
/ dashboard
```

Displays analytics and prediction insights.

---

## Database

SQLite is used to store:

* Prediction history
* Risk scores
* Prediction timestamps
* Analytics data

---

## Dashboard Features

* Total Predictions
* High-Risk Patients Percentage
* Average Risk Score
* Readmission Trends
* SHAP Global Feature Importance
* Risk Distribution Analysis

---

## Project Structure

```text
hospital-readmission-predictor/
│
├── app.py
├── preprocess.py
├── train.py
├── requirements.txt
├── README.md
│
├── data/
│   └── diabetic_data.csv
│
├── model/
│   ├── model.pkl
│   ├── scaler.pkl
│   ├── encoders.pkl
│   └── feature_names.pkl
│
├── templates/
│   ├── index.html
│   ├── result.html
│   └── dashboard.html
│
├── static/
│   └── style.css
│
├── utils/
│   ├── db.py
│   └── shap_utils.py
│
└── requirements.txt
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/hospital-readmission-predictor.git
cd hospital-readmission-predictor
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Virtual Environment

Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run Preprocessing

```bash
python preprocess.py
```

---

## Train Model

```bash
python train.py
```

---

## Start Flask Application

```bash
python app.py
```

---

## Future Enhancements

* Deep Learning Models
* Real-Time Hospital Integration
* Patient Follow-Up Recommendations
* Cloud Deployment
* Multi-Hospital Analytics
* LLM-Based Clinical Explanation Generation

---

## Key Learning Outcomes

* Data Cleaning and Preprocessing
* Feature Engineering
* Handling Class Imbalance with SMOTE
* XGBoost Model Development
* Explainable AI with SHAP
* Flask Web Development
* SQLite Integration
* Healthcare Analytics
* End-to-End Machine Learning Deployment

---

## Author

Sahil Bhardwaj

B.Tech Computer Science and Engineering

Guru Nanak Dev University (GNDU)

Aspiring Data Analyst | Machine Learning Enthusiast
