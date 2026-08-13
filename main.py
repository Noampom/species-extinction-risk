import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from dbfread import DBF

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
    classification_report, RocCurveDisplay, PrecisionRecallDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from matplotlib.colors import LinearSegmentedColormap


warnings.filterwarnings("ignore")


project_colors = ['#FFD700', '#DAA520', '#BDB76B', '#D8BFD8', '#6B8E23']


sns.set(style="whitegrid")
sns.set_palette(sns.color_palette(project_colors))

DATA_DIR = "data"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IUCN_BIG_CSV = os.path.join(DATA_DIR, "filtered_data.csv")
ANIMAL_TRAITS_CSV = os.path.join(DATA_DIR, "Animal Dataset.csv")
REPTILES_HYBAS_CSV = os.path.join(DATA_DIR, "reptiles_hybas_table.csv")
GBIF_CSV = os.path.join(DATA_DIR, "gbif_occurrences.csv")


def clean_columns(df):

    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )
    return df


def clean_name(x):

    if pd.isna(x):
        return np.nan
    return str(x).strip().lower()


def mode_or_first(x):

    x = x.dropna()
    if len(x) == 0:
        return np.nan
    mode = x.mode()
    if len(mode) > 0:
        return mode.iloc[0]
    return x.iloc[0]


def bool_to_int(x):

    if pd.isna(x):
        return 0
    x = str(x).strip().lower()
    return int(x in ["true", "1", "yes", "y"])


def threatened_from_category(cat):

    return int(cat in ["VU", "EN", "CR"])


def safe_log1p(series):

    return np.log1p(pd.to_numeric(series, errors="coerce").fillna(0).clip(lower=0))


def read_dbf(path):

    table = DBF(path, load=True, encoding="latin1", char_decode_errors="ignore")
    df = pd.DataFrame(iter(table))
    return clean_columns(df)


def standardize_iucn_spatial_table(df, source_name):

    df = clean_columns(df)

    rename_map = {
        "sci_name": "scientific_name",
        "class": "class_name",
        "order_": "order_name",
        "family": "family_name",
        "genus": "genus_name",
        "shape_area": "range_area",
        "shape_leng": "range_length",
        "shape_length": "range_length"
    }

    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    df["source_file"] = source_name

    if "scientific_name" not in df.columns:
        return pd.DataFrame()

    useful_cols = [
        "scientific_name", "category", "class_name", "order_name", "family_name",
        "genus_name", "presence", "origin", "seasonal", "legend",
        "marine", "terrestria", "freshwater",
        "range_area", "range_length", "yrcompiled", "source_file"
    ]

    for col in useful_cols:
        if col not in df.columns:
            df[col] = np.nan

    return df[useful_cols]


print("\nLoading main IUCN dataset...")

iucn_big = pd.read_csv(IUCN_BIG_CSV)
iucn_big = clean_columns(iucn_big)

iucn_big["scientific_name_clean"] = iucn_big["scientific_name"].apply(clean_name)

print("Main IUCN shape:", iucn_big.shape)
print("Main IUCN columns:", iucn_big.columns.tolist())


print("\nLoading IUCN DBF spatial tables...")

dbf_paths = glob.glob(os.path.join(DATA_DIR, "**", "*.dbf"), recursive=True)

spatial_parts = []

for path in dbf_paths:
    try:
        df_dbf = read_dbf(path)
        df_std = standardize_iucn_spatial_table(df_dbf, os.path.basename(path))
        if not df_std.empty:
            spatial_parts.append(df_std)
            print("Loaded:", path, df_std.shape)
    except Exception as e:
        print("Could not read:", path, e)

if os.path.exists(REPTILES_HYBAS_CSV):
    print("\nLoading reptiles_hybas_table.csv...")
    rept = pd.read_csv(REPTILES_HYBAS_CSV)
    rept = standardize_iucn_spatial_table(rept, "reptiles_hybas_table.csv")
    spatial_parts.append(rept)
    print("Loaded reptiles table:", rept.shape)

if len(spatial_parts) == 0:
    print("WARNING: No spatial DBF/CSV tables found.")
    spatial_raw = pd.DataFrame()
else:
    spatial_raw = pd.concat(spatial_parts, ignore_index=True)

print("\nSpatial raw shape:", spatial_raw.shape)

if not spatial_raw.empty:
    spatial_raw["scientific_name_clean"] = spatial_raw["scientific_name"].apply(clean_name)

    for col in ["range_area", "range_length", "yrcompiled"]:
        spatial_raw[col] = pd.to_numeric(spatial_raw[col], errors="coerce")

    for col in ["marine", "terrestria", "freshwater"]:
        spatial_raw[col] = spatial_raw[col].apply(bool_to_int)

    spatial_features = spatial_raw.groupby("scientific_name_clean").agg(
        scientific_name_spatial=("scientific_name", mode_or_first),
        category_spatial=("category", mode_or_first),
        class_spatial=("class_name", mode_or_first),
        order_spatial=("order_name", mode_or_first),
        family_spatial=("family_name", mode_or_first),
        genus_spatial=("genus_name", mode_or_first),
        legend=("legend", mode_or_first),
        range_area=("range_area", "sum"),
        range_length=("range_length", "sum"),
        num_spatial_records=("scientific_name_clean", "size"),
        has_marine=("marine", "max"),
        has_terrestrial=("terrestria", "max"),
        has_freshwater=("freshwater", "max"),
        yrcompiled=("yrcompiled", "max")
    ).reset_index()

else:
    spatial_features = pd.DataFrame(columns=["scientific_name_clean"])

print("Spatial features shape:", spatial_features.shape)


print("\nMerging datasets...")

data = iucn_big.merge(spatial_features, on="scientific_name_clean", how="left")

data["category_final"] = data["category"].fillna(data.get("category_spatial"))
data["class_final"] = data["class_name"].fillna(data.get("class_spatial"))
data["order_final"] = data["order_name"].fillna(data.get("order_spatial"))
data["family_final"] = data["family_name"].fillna(data.get("family_spatial"))
data["genus_final"] = data["genus_name"].fillna(data.get("genus_spatial"))

valid_categories = ["LC", "NT", "VU", "EN", "CR", "DD"]
data = data[data["category_final"].isin(valid_categories)].copy()

model_data_base = data[data["category_final"].isin(["LC", "NT", "VU", "EN", "CR"])].copy()
model_data_base["threatened"] = model_data_base["category_final"].apply(threatened_from_category)

for col in [
    "range_area", "range_length", "num_spatial_records",
    "has_marine", "has_terrestrial", "has_freshwater", "yrcompiled"
]:
    if col not in model_data_base.columns:
        model_data_base[col] = 0
    model_data_base[col] = pd.to_numeric(model_data_base[col], errors="coerce").fillna(0)

model_data_base["log_range_area"] = safe_log1p(model_data_base["range_area"])
model_data_base["log_range_length"] = safe_log1p(model_data_base["range_length"])
model_data_base["log_num_spatial_records"] = safe_log1p(model_data_base["num_spatial_records"])

print("Merged full data:", data.shape)
print("Model data:", model_data_base.shape)

print("\nCategory counts:")
print(model_data_base["category_final"].value_counts())

print("\nThreatened counts:")
print(model_data_base["threatened"].value_counts())


gbif_valid = None
lat_col = None
lon_col = None

if os.path.exists(GBIF_CSV):
    print("\nLoading GBIF occurrences...")

    gbif = pd.read_csv(GBIF_CSV, sep="\t")
    gbif = clean_columns(gbif)

    species_candidates = ["species", "scientificname", "scientific_name"]
    species_col = next((c for c in species_candidates if c in gbif.columns), None)

    lat_candidates = ["decimallatitude", "decimal_latitude", "latitude", "lat"]
    lon_candidates = ["decimallongitude", "decimal_longitude", "longitude", "lon"]

    lat_col = next((c for c in lat_candidates if c in gbif.columns), None)
    lon_col = next((c for c in lon_candidates if c in gbif.columns), None)

    if species_col is not None:
        gbif["scientific_name_clean"] = gbif[species_col].apply(clean_name)

        if lat_col is not None and lon_col is not None:
            gbif[lat_col] = pd.to_numeric(gbif[lat_col], errors="coerce")
            gbif[lon_col] = pd.to_numeric(gbif[lon_col], errors="coerce")
            gbif_valid = gbif.dropna(subset=[lat_col, lon_col]).copy()
        else:
            gbif_valid = gbif.copy()

        if lat_col is not None and lon_col is not None:
            if "country" in gbif_valid.columns:
                gbif_features = gbif_valid.groupby("scientific_name_clean").agg(
                    gbif_occurrence_count=("scientific_name_clean", "size"),
                    gbif_country_count=("country", pd.Series.nunique),
                    lat_min=(lat_col, "min"),
                    lat_max=(lat_col, "max"),
                    lon_min=(lon_col, "min"),
                    lon_max=(lon_col, "max")
                ).reset_index()
            elif "countrycode" in gbif_valid.columns:
                gbif_features = gbif_valid.groupby("scientific_name_clean").agg(
                    gbif_occurrence_count=("scientific_name_clean", "size"),
                    gbif_country_count=("countrycode", pd.Series.nunique),
                    lat_min=(lat_col, "min"),
                    lat_max=(lat_col, "max"),
                    lon_min=(lon_col, "min"),
                    lon_max=(lon_col, "max")
                ).reset_index()
            else:
                gbif_features = gbif_valid.groupby("scientific_name_clean").agg(
                    gbif_occurrence_count=("scientific_name_clean", "size"),
                    lat_min=(lat_col, "min"),
                    lat_max=(lat_col, "max"),
                    lon_min=(lon_col, "min"),
                    lon_max=(lon_col, "max")
                ).reset_index()
                gbif_features["gbif_country_count"] = 0

            gbif_features["gbif_lat_range"] = gbif_features["lat_max"] - gbif_features["lat_min"]
            gbif_features["gbif_lon_range"] = gbif_features["lon_max"] - gbif_features["lon_min"]
            gbif_features["gbif_geo_spread"] = (
                gbif_features["gbif_lat_range"] * gbif_features["gbif_lon_range"]
            )
        else:
            gbif_features = gbif_valid.groupby("scientific_name_clean").agg(
                gbif_occurrence_count=("scientific_name_clean", "size")
            ).reset_index()
            gbif_features["gbif_country_count"] = 0
            gbif_features["gbif_geo_spread"] = 0

        model_data_base = model_data_base.merge(gbif_features, on="scientific_name_clean", how="left")

        for col in ["gbif_occurrence_count", "gbif_country_count", "gbif_geo_spread"]:
            model_data_base[col] = pd.to_numeric(model_data_base[col], errors="coerce").fillna(0)

        model_data_base["log_gbif_occurrence_count"] = safe_log1p(
            model_data_base["gbif_occurrence_count"]
        )
        model_data_base["log_gbif_country_count"] = safe_log1p(
            model_data_base["gbif_country_count"]
        )
        model_data_base["log_gbif_geo_spread"] = safe_log1p(
            model_data_base["gbif_geo_spread"]
        )

        print("GBIF merged.")
        print("GBIF matched species:", (model_data_base["gbif_occurrence_count"] > 0).sum())
    else:
        print("GBIF file found, but no species/scientific name column detected.")
else:
    print("\nNo GBIF file yet. Continuing without GBIF features.")


print("\nCreating visualizations for Problem 1...")

project_cmap = LinearSegmentedColormap.from_list("project_map", project_colors)

category_order = ["LC", "NT", "VU", "EN", "CR"]

top_classes = model_data_base["class_final"].value_counts().head(10).index
class_data = model_data_base[model_data_base["class_final"].isin(top_classes)].copy()

class_cat = pd.crosstab(
    class_data["class_final"],
    class_data["category_final"],
    normalize="index"
)

class_cat = class_cat[[c for c in category_order if c in class_cat.columns]]

class_cat.plot(kind="bar", stacked=True, figsize=(12, 6))
plt.title("Conservation Status Composition by Taxonomic Class")
plt.xlabel("Taxonomic Class")
plt.ylabel("Proportion of Species")
plt.legend(title="IUCN Category", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "p1_fig2_status_by_class.png"), dpi=300)
plt.show()

order_summary = (
    model_data_base.groupby("order_final")
    .agg(
        n_species=("scientific_name_clean", "count"),
        threatened_rate=("threatened", "mean")
    )
    .reset_index()
)

order_summary = order_summary[order_summary["n_species"] >= 50]
order_summary = order_summary.sort_values("threatened_rate", ascending=False).head(15)

plt.figure(figsize=(10, 6))
sns.barplot(data=order_summary, x="threatened_rate", y="order_final")
plt.title("Taxonomic Orders with the Highest Share of Threatened Species")
plt.xlabel("Percentage of threatened species within each order")
plt.ylabel("Taxonomic Order")

plt.xlim(0, 1.05)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "p1_fig3_threatened_by_order.png"), dpi=300)
plt.show()

print("\nCreating visualizations for Problem 2...")

range_df = model_data_base[model_data_base["range_area"] > 0].copy()

range_df["range_bin"] = pd.qcut(
    range_df["log_range_area"],
    q=5,
    duplicates="drop"
)

range_bucket = (
    range_df.groupby("range_bin", observed=False)
    .agg(
        threatened_percentage=("threatened", lambda x: x.mean() * 100),
        n_species=("scientific_name_clean", "count"),
        min_log_range=("log_range_area", "min"),
        max_log_range=("log_range_area", "max")
    )
    .reset_index()
)

range_bucket["range_label"] = range_bucket.apply(
    lambda row: f"{row.min_log_range:.1f}–{row.max_log_range:.1f}",
    axis=1
)

plt.figure(figsize=(10, 5))
sns.lineplot(
    data=range_bucket,
    x="range_label",
    y="threatened_percentage",
    marker="o",
    linewidth=2,
    color='#6B8E23'
)

plt.title("Threatened Species Percentage by Geographic Range Size")
plt.xlabel("Geographic range size: log(1 + range area)")
plt.ylabel("Percentage of threatened species")

for i, row in enumerate(range_bucket.itertuples()):
    plt.text(
        i,
        row.threatened_percentage + 1,
        f"{row.threatened_percentage:.1f}%",
        ha="center",
        fontsize=9
    )

plt.ylim(0, min(100, range_bucket["threatened_percentage"].max() + 10))
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "p2_fig2_threatened_percentage_by_range_size_line.png"), dpi=300)
plt.show()


print("\nTraining models for Problem 3...")

categorical_features = [
    "class_final",
    "order_final",
    "family_final"
]

numeric_features = [
    "log_range_area",
    "log_range_length",
    "log_num_spatial_records",
    "has_marine",
    "has_terrestrial",
    "has_freshwater",
    "yrcompiled"
]

if "log_gbif_occurrence_count" in model_data_base.columns:
    numeric_features += [
        "log_gbif_occurrence_count",
        "log_gbif_country_count",
        "log_gbif_geo_spread"
    ]

categorical_features = [c for c in categorical_features if c in model_data_base.columns]
numeric_features = [c for c in numeric_features if c in model_data_base.columns]

ml_df = model_data_base[categorical_features + numeric_features + ["threatened"]].copy()

for col in categorical_features:
    ml_df[col] = ml_df[col].fillna("Unknown").astype(str)

for col in numeric_features:
    ml_df[col] = pd.to_numeric(ml_df[col], errors="coerce").fillna(0)

X = ml_df[categorical_features + numeric_features]
y = ml_df["threatened"]

print("ML rows:", len(ml_df))
print("Features:", X.columns.tolist())
print("Threatened rate:", y.mean())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

try:
    onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    onehot = OneHotEncoder(handle_unknown="ignore", sparse=False)

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", onehot, categorical_features),
        ("num", StandardScaler(), numeric_features)
    ],
    remainder="drop"
)

results = []

majority_class = y_train.value_counts().idxmax()
baseline_pred = np.full(y_test.shape, majority_class)

results.append({
    "model": "Majority Baseline",
    "accuracy": accuracy_score(y_test, baseline_pred),
    "precision": precision_score(y_test, baseline_pred, zero_division=0),
    "recall": recall_score(y_test, baseline_pred, zero_division=0),
    "f1": f1_score(y_test, baseline_pred, zero_division=0),
    "roc_auc": np.nan
})

logreg = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

logreg.fit(X_train, y_train)
logreg_pred = logreg.predict(X_test)
logreg_proba = logreg.predict_proba(X_test)[:, 1]

results.append({
    "model": "Logistic Regression",
    "accuracy": accuracy_score(y_test, logreg_pred),
    "precision": precision_score(y_test, logreg_pred, zero_division=0),
    "recall": recall_score(y_test, logreg_pred, zero_division=0),
    "f1": f1_score(y_test, logreg_pred, zero_division=0),
    "roc_auc": roc_auc_score(y_test, logreg_proba)
})

rf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=3,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ))
])

rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_proba = rf.predict_proba(X_test)[:, 1]

results.append({
    "model": "Random Forest",
    "accuracy": accuracy_score(y_test, rf_pred),
    "precision": precision_score(y_test, rf_pred, zero_division=0),
    "recall": recall_score(y_test, rf_pred, zero_division=0),
    "f1": f1_score(y_test, rf_pred, zero_division=0),
    "roc_auc": roc_auc_score(y_test, rf_proba)
})

results_df = pd.DataFrame(results)
print("\nModel results:")
print(results_df)

results_df.to_csv(os.path.join(OUTPUT_DIR, "model_results.csv"), index=False)

print("\nRandom Forest classification report:")
print(classification_report(
    y_test,
    rf_pred,
    target_names=["Not Threatened", "Threatened"]
))


print("\nCreating visualizations for Problem 3...")

cm_norm = confusion_matrix(y_test, rf_pred, normalize="true")
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_norm,
    display_labels=["Not Threatened", "Threatened"]
)
disp.plot(values_format=".2f", cmap=project_cmap)
plt.title("Normalized Confusion Matrix - Random Forest")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "p3_fig2_normalized_confusion_matrix.png"), dpi=300)
plt.show()

final_path = os.path.join(OUTPUT_DIR, "final_conservation_dataset.csv")
model_data_base.to_csv(final_path, index=False)

print("\nDONE.")
print("Saved final dataset to:", final_path)
print("Saved all figures and CSV outputs in:", OUTPUT_DIR)