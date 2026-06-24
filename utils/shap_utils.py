import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io
import base64


def _fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight',
                facecolor='#0d1117', dpi=120)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_b64


def get_waterfall_chart(model, scaler, input_array, feature_names, top_n=12):
    import xgboost as xgb

    booster = model.get_booster()
    dmatrix = xgb.DMatrix(input_array, feature_names=feature_names)

    # pred_contribs returns shape (n_samples, n_features + 1)
    # last column is the bias term
    contribs = booster.predict(dmatrix, pred_contribs=True)
    sv = contribs[0, :-1]  # drop bias

    indices = np.argsort(np.abs(sv))[-top_n:]
    vals    = sv[indices]
    names   = [feature_names[i] for i in indices]
    colors  = ['#ef4444' if v > 0 else '#22c55e' for v in vals]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')

    ax.barh(names, vals, color=colors, height=0.6, edgecolor='none')
    ax.axvline(0, color='#6e7681', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Contribution to readmission risk', color='#8b949e', fontsize=9)
    ax.tick_params(colors='#c9d1d9', labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')

    red_patch   = mpatches.Patch(color='#ef4444', label='Increases risk')
    green_patch = mpatches.Patch(color='#22c55e', label='Decreases risk')
    ax.legend(handles=[red_patch, green_patch], loc='lower right',
              facecolor='#161b22', edgecolor='#30363d',
              labelcolor='#c9d1d9', fontsize=8)

    ax.set_title('Feature impact on this prediction', color='#e6edf3',
                 fontsize=11, pad=10)
    fig.tight_layout()
    return _fig_to_base64(fig)


def get_global_importance_chart(model, feature_names, top_n=15):
    """Global feature importance from XGBoost's built-in scores."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]
    vals    = importances[indices]
    names   = [feature_names[i] for i in indices]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')

    ax.barh(names, vals, color='#3b82f6', height=0.6, edgecolor='none')
    ax.set_xlabel('Feature importance score', color='#8b949e', fontsize=9)
    ax.tick_params(colors='#c9d1d9', labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')

    ax.set_title('Global feature importance (XGBoost)', color='#e6edf3',
                 fontsize=11, pad=10)
    fig.tight_layout()
    return _fig_to_base64(fig)


def get_contribs(model, input_array, feature_names):
    """Return raw per-feature contributions for a single patient."""
    import xgboost as xgb
    booster = model.get_booster()
    dmatrix = xgb.DMatrix(input_array, feature_names=feature_names)
    contribs = booster.predict(dmatrix, pred_contribs=True)
    return contribs[0, :-1]  # drop bias


def explain_top_factors(sv, feature_names, input_row, top_n=3):
    """Plain-English top factor explanations."""
    indices = np.argsort(np.abs(sv))[-top_n:][::-1]
    explanations = []
    for i in indices:
        name   = feature_names[i]
        value  = input_row[i]
        impact = 'increases' if sv[i] > 0 else 'decreases'
        explanations.append({
            'feature': name.replace('_', ' ').title(),
            'value':   round(float(value), 2),
            'impact':  impact,
            'shap':    round(float(sv[i]), 4)
        })
    return explanations