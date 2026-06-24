from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
import os
import xgboost as xgb

from utils.db import init_db, log_prediction, get_recent_predictions, get_stats
from utils.shap_utils import (get_waterfall_chart, get_global_importance_chart,
                               get_contribs, explain_top_factors)

app = Flask(__name__)

# ── Load artifacts once at startup ────────────────────────────────────────────
MODEL_DIR     = 'model'
model         = joblib.load(os.path.join(MODEL_DIR, 'model.pkl'))
scaler        = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
encoders      = joblib.load(os.path.join(MODEL_DIR, 'encoders.pkl'))
feature_names = joblib.load(os.path.join(MODEL_DIR, 'feature_names.pkl'))

_global_chart = None

def get_cached_global_chart():
    global _global_chart
    if _global_chart is None:
        _global_chart = get_global_importance_chart(model, feature_names)
    return _global_chart

# ── Form option data ───────────────────────────────────────────────────────────
RACE_OPTIONS   = ['Caucasian', 'AfricanAmerican', 'Hispanic', 'Asian', 'Other']
GENDER_OPTIONS = ['Male', 'Female']
DIAG_GROUPS    = ['Circulatory', 'Respiratory', 'Digestive', 'Diabetes',
                  'Injury', 'Musculoskeletal', 'Genitourinary', 'Neoplasms', 'Other']
ADMISSION_TYPE = {
    1: 'Emergency',
    2: 'Urgent',
    3: 'Elective (Planned)',
    4: 'Newborn',
    7: 'Trauma Center',
}
DISCHARGE_DISP = {
    1:  'Discharged to Home',
    2:  'Transferred to Another Hospital',
    3:  'Transferred to Skilled Nursing Facility',
    4:  'Transferred to Intermediate Care Facility',
    6:  'Discharged to Home with Health Service',
    13: 'Hospice Care (Home)',
    14: 'Hospice Care (Medical Facility)',
    11: 'Patient Deceased',
}
ADMISSION_SOURCE = {
    1: 'Physician Referral',
    2: 'Clinic Referral',
    3: 'HMO Referral',
    4: 'Transferred from Another Hospital',
    5: 'Transferred from Skilled Nursing Facility',
    6: 'Transferred from Another Healthcare Facility',
    7: 'Emergency Room',
    8: 'Court / Law Enforcement',
}

init_db()


def encode_input(form):
    """Convert raw form POST values → scaled numpy array."""
    age_map = {'5':5,'15':15,'25':25,'35':35,'45':45,
               '55':55,'65':65,'75':75,'85':85,'95':95}

    def label_enc(col, val):
        if col in encoders:
            le = encoders[col]
            val = str(val)
            if val in le.classes_:
                return int(le.transform([val])[0])
        return 0

    row = {fn: 0 for fn in feature_names}

    row['race']                     = label_enc('race',   form.get('race', 'Caucasian'))
    row['gender']                   = label_enc('gender', form.get('gender', 'Male'))
    row['age']                      = int(age_map.get(form.get('age', '55'), 55))
    row['admission_type_id']        = int(form.get('admission_type_id', 1))
    row['discharge_disposition_id'] = int(form.get('discharge_disposition_id', 1))
    row['admission_source_id']      = int(form.get('admission_source_id', 7))
    row['time_in_hospital']         = int(form.get('time_in_hospital', 3))
    row['num_lab_procedures']       = int(form.get('num_lab_procedures', 40))
    row['num_procedures']           = int(form.get('num_procedures', 1))
    row['num_medications']          = int(form.get('num_medications', 10))
    row['number_outpatient']        = int(form.get('number_outpatient', 0))
    row['number_emergency']         = int(form.get('number_emergency', 0))
    row['number_inpatient']         = int(form.get('number_inpatient', 0))
    row['diag_1']                   = label_enc('diag_1', form.get('diag_1', 'Other'))
    row['diag_2']                   = label_enc('diag_2', form.get('diag_2', 'Other'))
    row['diag_3']                   = label_enc('diag_3', form.get('diag_3', 'Other'))
    row['number_diagnoses']         = int(form.get('number_diagnoses', 5))
    row['change']                   = 1 if form.get('change') == 'Ch' else 0
    row['diabetesMed']              = 1 if form.get('diabetesMed') == 'Yes' else 0
    row['insulin']                  = int(form.get('insulin', 0))
    row['metformin']                = int(form.get('metformin', 0))
    row['A1Cresult']                = label_enc('A1Cresult',    form.get('A1Cresult', 'None'))
    row['max_glu_serum']            = label_enc('max_glu_serum', form.get('max_glu_serum', 'None'))

    input_df    = pd.DataFrame([row])[feature_names]
    input_array = scaler.transform(input_df)
    return input_array, row


@app.route('/')
def index():
    return render_template('index.html',
                           races=RACE_OPTIONS,
                           genders=GENDER_OPTIONS,
                           diag_groups=DIAG_GROUPS,
                           admission_types=ADMISSION_TYPE,
                           discharge_disps=DISCHARGE_DISP,
                           admission_sources=ADMISSION_SOURCE)


@app.route('/predict', methods=['POST'])
def predict():
    form = request.form
    input_array, raw_row = encode_input(form)

    prob       = model.predict_proba(input_array)[0][1]
    risk_score = round(float(prob) * 100, 1)

    if risk_score >= 60:
        risk_label = 'High Risk';   risk_color = '#ef4444'
    elif risk_score >= 35:
        risk_label = 'Moderate Risk'; risk_color = '#f59e0b'
    else:
        risk_label = 'Low Risk';    risk_color = '#22c55e'

    # Use XGBoost's built-in pred_contribs — no SHAP library needed
    sv              = get_contribs(model, input_array, feature_names)
    waterfall_chart = get_waterfall_chart(model, scaler, input_array, feature_names)
    top_factors     = explain_top_factors(sv, feature_names, input_array[0])

    log_prediction(prob, risk_label, raw_row)

    return render_template('result.html',
                           risk_score=risk_score,
                           risk_label=risk_label,
                           risk_color=risk_color,
                           waterfall_chart=waterfall_chart,
                           top_factors=top_factors)


@app.route('/dashboard')
def dashboard():
    stats        = get_stats()
    recent       = get_recent_predictions(20)
    global_chart = get_cached_global_chart()
    return render_template('dashboard.html',
                           stats=stats,
                           recent=recent,
                           global_chart=global_chart)


if __name__ == '__main__':
    app.run(debug=True)