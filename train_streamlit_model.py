from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'outputs' / 'final_conservation_dataset.csv'
OUT = ROOT / 'outputs' / 'streamlit_rf_model.joblib'

categorical_features = ['class_final', 'order_final', 'family_final']
numeric_features = [
    'log_range_area', 'log_range_length', 'log_num_spatial_records',
    'has_marine', 'has_terrestrial', 'has_freshwater', 'yrcompiled',
    'log_gbif_occurrence_count', 'log_gbif_country_count', 'log_gbif_geo_spread'
]

usecols = categorical_features + numeric_features + ['threatened']
df = pd.read_csv(DATA, usecols=usecols)
for c in categorical_features:
    df[c] = df[c].fillna('Unknown').astype(str)
for c in numeric_features:
    df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

X = df[categorical_features + numeric_features]
y = df['threatened'].astype(int)
X_train, _, y_train, _ = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=True), categorical_features),
    ('num', StandardScaler(), numeric_features),
])
rf = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_leaf=3,
        random_state=42, class_weight='balanced', n_jobs=-1
    ))
])
rf.fit(X_train, y_train)
joblib.dump(rf, OUT, compress=3)
print(OUT)
