"""
predict.py - Prédictions avec les modèles entraînés
Supporte : classification (Churn), clustering (segment client), régression (MonetaryTotal)
"""

import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from utils import load_model

# ── Chemins ──────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.join(BASE_DIR, '..')
MODEL_DIR = os.path.join(ROOT_DIR, 'models')


# ─────────────────────────────────────────────────────────────────────────────
# UTILITAIRE : aligner les colonnes d'une entrée sur feature_names
# ─────────────────────────────────────────────────────────────────────────────

def _align_features(input_dict_or_df, feature_names):
    """
    Construit un DataFrame avec exactement les colonnes attendues.
    Colonnes manquantes → 0.  Colonnes inconnues → ignorées.
    """
    if isinstance(input_dict_or_df, dict):
        df = pd.DataFrame([input_dict_or_df])
    else:
        df = input_dict_or_df.copy()

    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    return df[feature_names]


# ─────────────────────────────────────────────────────────────────────────────
# 1. CLASSIFICATION — CHURN
# ─────────────────────────────────────────────────────────────────────────────

def predict_churn_single(features_dict,
                         model_path=None,
                         scaler_path=None,
                         feature_names_path=None):
    """
    Prédit le risque de churn pour un seul client.

    Parameters
    ----------
    features_dict : dict — features brutes du client (avant scaling)

    Returns
    -------
    dict : prediction, prob_churn, prob_fidele, risque
    """
    model_path         = model_path         or os.path.join(MODEL_DIR, 'best_model.pkl')
    scaler_path        = scaler_path        or os.path.join(MODEL_DIR, 'scaler.pkl')
    feature_names_path = feature_names_path or os.path.join(MODEL_DIR, 'feature_names.pkl')

    model         = load_model(model_path)
    scaler        = load_model(scaler_path)
    feature_names = load_model(feature_names_path)

    df_input = _align_features(features_dict, feature_names)
    df_scaled = pd.DataFrame(
        scaler.transform(df_input), columns=feature_names
    )

    prediction  = model.predict(df_scaled)[0]
    probability = model.predict_proba(df_scaled)[0]
    prob_churn  = float(probability[1])

    risque = ('ÉLEVÉ'  if prob_churn > 0.7 else
              'MOYEN'  if prob_churn > 0.3 else
              'FAIBLE')

    return {
        'prediction':     int(prediction),
        'prob_churn':     round(prob_churn, 4),
        'prob_fidele':    round(float(probability[0]), 4),
        'risque':         risque,
        'interpretation': f"Client {'à risque de départ' if prediction == 1 else 'fidèle'} "
                          f"(probabilité churn : {prob_churn*100:.1f}%)"
    }


def predict_churn_batch(X,
                        model_path=None,
                        scaler_path=None,
                        feature_names_path=None):
    """
    Prédit le churn pour un DataFrame de clients.

    Parameters
    ----------
    X : pd.DataFrame — features brutes (avant scaling)

    Returns
    -------
    pd.DataFrame avec colonnes prediction, prob_churn, prob_fidele, risque
    """
    model_path         = model_path         or os.path.join(MODEL_DIR, 'best_model.pkl')
    scaler_path        = scaler_path        or os.path.join(MODEL_DIR, 'scaler.pkl')
    feature_names_path = feature_names_path or os.path.join(MODEL_DIR, 'feature_names.pkl')

    model         = load_model(model_path)
    scaler        = load_model(scaler_path)
    feature_names = load_model(feature_names_path)

    X_aligned = _align_features(X, feature_names)
    X_scaled  = pd.DataFrame(
        scaler.transform(X_aligned), columns=feature_names
    )

    predictions   = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)

    results = pd.DataFrame({
        'prediction':  predictions,
        'prob_churn':  probabilities[:, 1].round(4),
        'prob_fidele': probabilities[:, 0].round(4),
    })
    results['risque'] = results['prob_churn'].apply(
        lambda x: 'ÉLEVÉ' if x > 0.7 else 'MOYEN' if x > 0.3 else 'FAIBLE'
    )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 2. CLUSTERING — SEGMENT CLIENT
# ─────────────────────────────────────────────────────────────────────────────

# Labels métier associés aux 4 clusters K-Means
# (à affiner après analyse des profils dans train_model.py)
CLUSTER_LABELS = {
    0: 'Champions',
    1: 'Clients fidèles',
    2: 'Clients potentiels',
    3: 'Clients dormants',
}


def predict_cluster_single(features_dict,
                            scaler_path=None,
                            pca_path=None,
                            kmeans_path=None,
                            feature_names_path=None):
    """
    Identifie le segment d'un client via K-Means.

    Returns
    -------
    dict : cluster_id, cluster_label, distances_to_centroids
    """
    scaler_path        = scaler_path        or os.path.join(MODEL_DIR, 'scaler.pkl')
    pca_path           = pca_path           or os.path.join(MODEL_DIR, 'pca.pkl')
    kmeans_path        = kmeans_path        or os.path.join(MODEL_DIR, 'kmeans.pkl')
    feature_names_path = feature_names_path or os.path.join(MODEL_DIR, 'feature_names.pkl')

    scaler        = load_model(scaler_path)
    pca           = load_model(pca_path)
    kmeans        = load_model(kmeans_path)
    feature_names = load_model(feature_names_path)

    df_input  = _align_features(features_dict, feature_names)
    df_scaled = pd.DataFrame(scaler.transform(df_input), columns=feature_names)
    X_pca     = pca.transform(df_scaled)
    cluster   = int(kmeans.predict(X_pca)[0])

    distances = kmeans.transform(X_pca)[0].round(4).tolist()

    return {
        'cluster_id':    cluster,
        'cluster_label': CLUSTER_LABELS.get(cluster, f'Cluster {cluster}'),
        'distances_to_centroids': {f'Cluster {i}': d
                                   for i, d in enumerate(distances)}
    }


def predict_cluster_batch(X,
                          scaler_path=None,
                          pca_path=None,
                          kmeans_path=None,
                          feature_names_path=None):
    """
    Segmente un DataFrame de clients.

    Returns
    -------
    pd.DataFrame avec colonnes cluster_id, cluster_label
    """
    scaler_path        = scaler_path        or os.path.join(MODEL_DIR, 'scaler.pkl')
    pca_path           = pca_path           or os.path.join(MODEL_DIR, 'pca.pkl')
    kmeans_path        = kmeans_path        or os.path.join(MODEL_DIR, 'kmeans.pkl')
    feature_names_path = feature_names_path or os.path.join(MODEL_DIR, 'feature_names.pkl')

    scaler        = load_model(scaler_path)
    pca           = load_model(pca_path)
    kmeans        = load_model(kmeans_path)
    feature_names = load_model(feature_names_path)

    X_aligned = _align_features(X, feature_names)
    X_scaled  = pd.DataFrame(scaler.transform(X_aligned), columns=feature_names)
    X_pca     = pca.transform(X_scaled)
    clusters  = kmeans.predict(X_pca)

    results = pd.DataFrame({
        'cluster_id':    clusters,
        'cluster_label': [CLUSTER_LABELS.get(c, f'Cluster {c}') for c in clusters]
    })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 3. RÉGRESSION — PRÉDICTION MONETARY TOTAL
# ─────────────────────────────────────────────────────────────────────────────

def predict_monetary_single(features_dict,
                             regressor_path=None,
                             scaler_path=None,
                             feature_names_path=None):
    """
    Prédit le montant total dépensé (MonetaryTotal) d'un client.

    Returns
    -------
    dict : predicted_monetary, interpretation
    """
    regressor_path     = regressor_path     or os.path.join(MODEL_DIR, 'regressor.pkl')
    scaler_path        = scaler_path        or os.path.join(MODEL_DIR, 'scaler.pkl')
    feature_names_path = feature_names_path or os.path.join(MODEL_DIR, 'feature_names.pkl')

    regressor     = load_model(regressor_path)
    scaler        = load_model(scaler_path)
    feature_names = load_model(feature_names_path)

    df_input  = _align_features(features_dict, feature_names)
    df_scaled = pd.DataFrame(scaler.transform(df_input), columns=feature_names)

    predicted = float(regressor.predict(df_scaled)[0])

    return {
        'predicted_monetary': round(predicted, 2),
        'interpretation': f"Dépense totale estimée : £{predicted:,.2f}"
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. EXPLICATION DE PRÉDICTION (feature importance locale)
# ─────────────────────────────────────────────────────────────────────────────

def explain_prediction(features_dict,
                       model_path=None,
                       scaler_path=None,
                       feature_names_path=None,
                       top_n=10):
    """
    Retourne les features les plus influentes pour la prédiction
    d'un client donné (basé sur feature_importances_ ou coef_).

    Returns
    -------
    dict : top_features (liste triée), valeurs du client
    """
    model_path         = model_path         or os.path.join(MODEL_DIR, 'best_model.pkl')
    scaler_path        = scaler_path        or os.path.join(MODEL_DIR, 'scaler.pkl')
    feature_names_path = feature_names_path or os.path.join(MODEL_DIR, 'feature_names.pkl')

    model         = load_model(model_path)
    scaler        = load_model(scaler_path)
    feature_names = load_model(feature_names_path)

    if not (hasattr(model, 'feature_importances_') or hasattr(model, 'coef_')):
        return {'error': 'Modèle non compatible avec explain_prediction'}

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        importances = np.abs(model.coef_[0])

    df_input = _align_features(features_dict, feature_names)

    imp_df = pd.DataFrame({
        'feature':    feature_names,
        'importance': importances,
        'value':      df_input.values[0]
    }).sort_values('importance', ascending=False).head(top_n)

    return {
        'top_features': imp_df.to_dict('records'),
        'note': f"Top {top_n} features les plus influentes pour ce client"
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. PRÉDICTION COMPLÈTE (toutes tâches)
# ─────────────────────────────────────────────────────────────────────────────

def full_predict(features_dict):
    """
    Lance les 3 prédictions (churn, cluster, monetary) pour un client.
    Retourne un dict consolidé prêt pour l'interface Flask.
    """
    result = {'client_features': features_dict}

    # Classification churn
    try:
        result['churn'] = predict_churn_single(features_dict)
    except Exception as e:
        result['churn'] = {'error': str(e)}

    # Clustering segment
    try:
        result['cluster'] = predict_cluster_single(features_dict)
    except Exception as e:
        result['cluster'] = {'error': str(e)}

    # Régression monetary
    try:
        result['monetary'] = predict_monetary_single(features_dict)
    except Exception as e:
        result['monetary'] = {'error': str(e)}

    # Explication
    try:
        result['explanation'] = explain_prediction(features_dict)
    except Exception as e:
        result['explanation'] = {'error': str(e)}

    return result


# ─────────────────────────────────────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    example_client = {
        'Recency':        50,
        'Frequency':      10,
        'MonetaryTotal':  500,
        'MonetaryStd':    80,
        'MonetaryMax':    200,
        'TotalQuantity':  120,
        'Age':            35,
        'CustomerTenure': 365,
        'WeekendRatio':   0.3,
        'ReturnRatio':    0.05,
        'CancelledTrans': 1,
        'UniqueProducts': 15,
    }

    print("\n" + "="*60)
    print("  DÉMO — Prédiction complète pour un client")
    print("="*60)

    result = full_predict(example_client)

    if 'error' not in result.get('churn', {}):
        c = result['churn']
        print(f"\n  Churn     : {c['interpretation']}")
        print(f"  Risque    : {c['risque']}")

    if 'error' not in result.get('cluster', {}):
        cl = result['cluster']
        print(f"\n  Segment   : {cl['cluster_label']} (cluster {cl['cluster_id']})")

    if 'error' not in result.get('monetary', {}):
        m = result['monetary']
        print(f"\n  Monétaire : {m['interpretation']}")

    if 'error' not in result.get('explanation', {}):
        print(f"\n  Top features influentes :")
        for f in result['explanation']['top_features'][:5]:
            print(f"    {f['feature']:35s}  importance={f['importance']:.4f}  valeur={f['value']:.2f}")