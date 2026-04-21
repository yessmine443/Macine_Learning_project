"""
preprocessing.py - Nettoyage complet + ACP
Pipeline : imputation → parsing → suppression → outliers →
           multicolinéarité → encodage → split → scaling → SMOTE → ACP
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ── Chemins (à adapter si nécessaire) ────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.join(BASE_DIR, '..')
RAW_PATH  = os.path.join(ROOT_DIR, 'data', 'raw',
                          'retail_customers_COMPLETE_CATEGORICAL.csv')
PROC_DIR  = os.path.join(ROOT_DIR, 'data', 'processed')
TT_DIR    = os.path.join(ROOT_DIR, 'data', 'train_test')
MODEL_DIR = os.path.join(ROOT_DIR, 'models')

for d in [PROC_DIR, TT_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 0 : CHARGEMENT
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_csv(RAW_PATH)

print("=" * 60)
print("AVANT NETTOYAGE")
print("=" * 60)
print(f"Shape : {df.shape}")
print(f"\nValeurs manquantes :")
print(df.isnull().sum()[df.isnull().sum() > 0])
print(f"\nTypes :")
print(df.dtypes.value_counts())


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 1 : IMPUTATION DES VALEURS MANQUANTES
# ─────────────────────────────────────────────────────────────────────────────

# Age — médiane (distribution asymétrique)
median_age = df['Age'].median()
df['Age'] = df['Age'].fillna(median_age)
print(f"\n[1] Age          — médiane : {median_age:.1f}")

# AvgDaysBetweenPurchases — médiane
median_days = df['AvgDaysBetweenPurchases'].median()
df['AvgDaysBetweenPurchases'] = df['AvgDaysBetweenPurchases'].fillna(median_days)
print(f"[1] AvgDaysBtw   — médiane : {median_days:.1f}")

# SupportTicketsCount — valeurs aberrantes (-1, 999) → NaN → médiane
df['SupportTicketsCount'] = df['SupportTicketsCount'].replace([-1, 999], np.nan)
median_tickets = df['SupportTicketsCount'].median()
df['SupportTicketsCount'] = df['SupportTicketsCount'].fillna(median_tickets)
print(f"[1] SupportTkt   — médiane : {median_tickets:.1f}")

# SatisfactionScore — valeurs aberrantes (-1, 99) → NaN → mode
df['SatisfactionScore'] = df['SatisfactionScore'].replace([-1, 99], np.nan)
mode_sat = df['SatisfactionScore'].mode()[0]
df['SatisfactionScore'] = df['SatisfactionScore'].fillna(mode_sat)
print(f"[1] Satisfaction — mode    : {mode_sat}")

print(f"\n    Valeurs manquantes restantes : {df.isnull().sum().sum()}")


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 2 : PARSING DES DATES
# ─────────────────────────────────────────────────────────────────────────────

df['RegistrationDate'] = pd.to_datetime(
    df['RegistrationDate'], dayfirst=True, errors='coerce'
)
df['RegYear']    = df['RegistrationDate'].dt.year
df['RegMonth']   = df['RegistrationDate'].dt.month
df['RegDay']     = df['RegistrationDate'].dt.day
df['RegWeekday'] = df['RegistrationDate'].dt.weekday
df.drop('RegistrationDate', axis=1, inplace=True)
print(f"\n[2] RegistrationDate parsée → RegYear/Month/Day/Weekday")


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 3 : SUPPRESSION DES FEATURES INUTILES
# ─────────────────────────────────────────────────────────────────────────────

# LastLoginIP : extraire is_private avant suppression
if 'LastLoginIP' in df.columns:
    df['IP_IsPrivate'] = df['LastLoginIP'].astype(str).apply(
        lambda ip: 1 if (ip.startswith('10.') or
                         ip.startswith('192.168.') or
                         ip.startswith('172.')) else 0
    )
    df.drop('LastLoginIP', axis=1, inplace=True)
    print("[3] LastLoginIP  → IP_IsPrivate (feature engineered)")

cols_to_drop = ['NewsletterSubscribed', 'CustomerID']
existing = [c for c in cols_to_drop if c in df.columns]
df.drop(existing, axis=1, inplace=True)
print(f"[3] Supprimées   : {existing}")

# ⚠ ANTI-LEAKAGE : supprimer toutes les features dérivées de Churn
# Ces colonnes encodent directement ou indirectement la variable cible
# Les laisser = donner la réponse au modèle → accuracy artificielle de 1.0
leakage_cols = [
    'ChurnRiskCategory',   # encodage direct du risque de churn = quasi = Churn
    'RFMSegment',          # calculé à partir de Recency/Frequency/Monetary
                           # qui sont déjà dans les features → redondance totale
    'LoyaltyLevel',        # dérivé de CustomerTenure déjà présent
    'SpendingCategory',    # dérivé de MonetaryTotal déjà présent
    'AgeCategory',         # dérivé de Age déjà présent
    'CustomerType',        # profil comportemental construit sur les autres features
    'BasketSizeCategory',  # dérivé de AvgQuantityPerTransaction déjà présent
]
existing_leakage = [c for c in leakage_cols if c in df.columns]
df.drop(existing_leakage, axis=1, inplace=True)
print(f"[3] Anti-leakage : {existing_leakage}")
print(f"    Shape        : {df.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 4 : OUTLIERS — méthode IQR (capping)
# ─────────────────────────────────────────────────────────────────────────────

def cap_outliers_iqr(series):
    Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
    IQR = Q3 - Q1
    return series.clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

outlier_cols = ['MonetaryTotal', 'MonetaryAvg', 'MonetaryStd',
                'MonetaryMin', 'MonetaryMax', 'TotalQuantity',
                'AvgQuantityPerTransaction', 'Age']

for col in outlier_cols:
    if col in df.columns:
        df[col] = cap_outliers_iqr(df[col])
print(f"\n[4] Outliers cappés (IQR) sur {len([c for c in outlier_cols if c in df.columns])} colonnes")


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 5 : FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

if 'MonetaryTotal' in df.columns and 'Recency' in df.columns:
    df['MonetaryPerDay']  = df['MonetaryTotal'] / (df['Recency'] + 1)
if 'MonetaryTotal' in df.columns and 'Frequency' in df.columns:
    df['AvgBasketValue']  = df['MonetaryTotal'] / (df['Frequency'] + 1)
if 'Recency' in df.columns and 'CustomerTenure' in df.columns:
    df['TenureRatio']     = df['Recency'] / (df['CustomerTenure'] + 1)

print(f"\n[5] Nouvelles features créées : MonetaryPerDay, AvgBasketValue, TenureRatio")
print(f"    Shape : {df.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 6 : MULTICOLINÉARITÉ
# ─────────────────────────────────────────────────────────────────────────────

corr_drop = ['MonetaryAvg', 'TotalTransactions', 'UniqueInvoices',
             'UniqueDescriptions', 'MinQuantity', 'AvgProductsPerTransaction',
             'NegativeQuantityCount']
existing_corr = [c for c in corr_drop if c in df.columns]
df.drop(existing_corr, axis=1, inplace=True)
print(f"\n[6] Multicolinéarité — supprimées : {existing_corr}")
print(f"    Shape : {df.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 6b : SUPPRIMER LES FEATURES PROXIES DE CHURN
# ─────────────────────────────────────────────────────────────────────────────
# Ces features sont des signaux quasi-directs du départ client.
# Les conserver permet au modèle de "deviner" Churn sans vraiment apprendre.
# SatisfactionScore=1 ou CancelledTrans élevé → client parti dans 95% des cas.
leakage_proxy = [
    'SatisfactionScore',  # note satisfaction → proxy direct churn
    'CancelledTrans',     # nb annulations → signal fort de départ
    'ZeroPriceCount',     # anomalie transactionnelle corrélée aux churners
]
existing_proxy = [c for c in leakage_proxy if c in df.columns]
df.drop(existing_proxy, axis=1, inplace=True)
print(f"\n[6b] Proxies churn supprimés : {existing_proxy}")
print(f"     Shape : {df.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 7 : ENCODAGE DES VARIABLES CATÉGORIELLES
# ─────────────────────────────────────────────────────────────────────────────

# One-Hot — features catégorielles restantes (les cols leakage déjà supprimées)
nominal_cols = ['FavoriteSeason', 'PreferredTimeOfDay', 'Region',
                'WeekendPreference', 'ProductDiversity', 'Gender',
                'AccountStatus', 'Country']
nominal_cols = [c for c in nominal_cols if c in df.columns]
df = pd.get_dummies(df, columns=nominal_cols, drop_first=True)

print(f"\n[7] Encodage terminé — Shape : {df.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 8 : SPLIT TRAIN / TEST (avant tout scaling)
# ─────────────────────────────────────────────────────────────────────────────

X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y          # préserver le ratio Churn
)
print(f"\n[8] Split 80/20 stratifié")
print(f"    X_train : {X_train.shape}  |  X_test : {X_test.shape}")
print(f"    Churn train : {y_train.mean()*100:.1f}%  |  Churn test : {y_test.mean()*100:.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 9 : SCALING — fit sur X_train uniquement
# ─────────────────────────────────────────────────────────────────────────────

feature_names = X_train.columns.tolist()
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train), columns=feature_names
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test), columns=feature_names
)

joblib.dump(scaler,        os.path.join(MODEL_DIR, 'scaler.pkl'))
joblib.dump(feature_names, os.path.join(MODEL_DIR, 'feature_names.pkl'))
print(f"\n[9] StandardScaler — fit sur X_train")
print(f"    scaler.pkl et feature_names.pkl sauvegardés")


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 10 : ÉQUILIBRAGE — SMOTE sur X_train uniquement
# ─────────────────────────────────────────────────────────────────────────────

smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)

print(f"\n[10] SMOTE appliqué sur train uniquement")
print(f"    Avant : {dict(y_train.value_counts().sort_index())}")
print(f"    Après : {dict(pd.Series(y_train_bal).value_counts().sort_index())}")


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 11 : ACP sur X_train_bal (réduction dimensionnelle)
# ─────────────────────────────────────────────────────────────────────────────
from sklearn.decomposition import PCA

# Choisir n_components pour 95% de variance expliquée
pca_full = PCA(random_state=42)
pca_full.fit(X_train_bal)
cumvar = np.cumsum(pca_full.explained_variance_ratio_)
n_comp_95 = int(np.argmax(cumvar >= 0.95) + 1)

print(f"\n[11] ACP : {n_comp_95} composantes retiennent 95% de la variance")

pca = PCA(n_components=n_comp_95, random_state=42)
X_train_pca = pd.DataFrame(
    pca.fit_transform(X_train_bal),
    columns=[f'PC{i+1}' for i in range(n_comp_95)]
)
X_test_pca = pd.DataFrame(
    pca.transform(X_test_scaled),
    columns=[f'PC{i+1}' for i in range(n_comp_95)]
)

joblib.dump(pca, os.path.join(MODEL_DIR, 'pca.pkl'))
print(f"    pca.pkl sauvegardé ({n_comp_95} composantes)")
print(f"    X_train_pca : {X_train_pca.shape}")
print(f"    X_test_pca  : {X_test_pca.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 12 : SAUVEGARDE
# ─────────────────────────────────────────────────────────────────────────────

# Données scalées (pour classification sans ACP)
X_train_bal.to_csv(os.path.join(TT_DIR, 'X_train.csv'),     index=False)
X_test_scaled.to_csv(os.path.join(TT_DIR, 'X_test.csv'),    index=False)
pd.Series(y_train_bal, name='Churn').to_csv(
    os.path.join(TT_DIR, 'y_train.csv'), index=False)
y_test.to_csv(os.path.join(TT_DIR, 'y_test.csv'),           index=False)

# Données ACP (pour clustering + modèles sur espace réduit)
X_train_pca.to_csv(os.path.join(TT_DIR, 'X_train_pca.csv'), index=False)
X_test_pca.to_csv(os.path.join(TT_DIR, 'X_test_pca.csv'),   index=False)

# Dataset nettoyé complet
df_clean = pd.concat([X_train_bal, X_test_scaled])
df_clean.to_csv(os.path.join(PROC_DIR, 'retail_customers_CLEANED.csv'), index=False)

print(f"\n[12] Fichiers sauvegardés dans data/train_test/ et data/processed/")


# ─────────────────────────────────────────────────────────────────────────────
# RÉCAPITULATIF
# ─────────────────────────────────────────────────────────────────────────────
print(f"""
{'='*60}
RÉCAPITULATIF PREPROCESSING
{'='*60}
Données originales    : 4 372 lignes × 52 colonnes
Après nettoyage       : {df.shape[0]} lignes × {df.shape[1]} colonnes
Après encodage        : {len(feature_names)} features

Train (avant SMOTE)   : {X_train.shape[0]} échantillons
Train (après SMOTE)   : {X_train_bal.shape[0]} échantillons
Test                  : {X_test.shape[0]} échantillons

ACP                   : {n_comp_95} composantes (95% variance)

Artefacts models/     :
  • scaler.pkl
  • feature_names.pkl
  • pca.pkl
{'='*60}
""")