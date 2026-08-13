from pathlib import Path
import math
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

st.markdown("""
<style>
/* Hide Fork button */
[data-testid="stAppDeployButton"] {
    display: none;
}

/* Hide GitHub icon/link */
a[href*="github.com"] {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "outputs" / "final_conservation_dataset.csv"
MODEL_PATH = ROOT / "outputs" / "streamlit_rf_model.joblib"
RESULTS_PATH = ROOT / "outputs" / "model_results.csv"

st.set_page_config(
    page_title="Species Extinction Risk Explorer",
    page_icon="🌿",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        --cream: #fffdf2;
        --gold: #d7b400;
        --olive: #6b7f2b;
        --ink: #3f3a2f;
    }
    .stApp { background: linear-gradient(180deg, #fffdf5 0%, #ffffff 35%); }
    .hero {
        padding: 1.2rem 1.4rem;
        border-radius: 18px;
        background: #f6efbe;
        border: 1px solid #e7d977;
        margin-bottom: 1rem;
    }
    .hero h1 { color: #3f3a2f; margin: 0 0 .25rem 0; }
    .hero p { margin: 0; color: #5d5749; }
    .small-note { color:#6f685a; font-size:.9rem; }
    .risk-high { padding: 1rem; border-radius: 14px; background:#f7e9b7; border:1px solid #d7b400; }
    .risk-low { padding: 1rem; border-radius: 14px; background:#eef3d8; border:1px solid #86964a; }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #eee7c8;
        padding: .7rem .8rem;
        border-radius: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

CATEGORY_LABELS = {
    "LC": "Least Concern",
    "NT": "Near Threatened",
    "VU": "Vulnerable",
    "EN": "Endangered",
    "CR": "Critically Endangered",
}
CATEGORY_ORDER = ["LC", "NT", "VU", "EN", "CR"]

@st.cache_data(show_spinner=False)
def load_data():
    cols = [
        "scientific_name", "main_common_name", "category_final", "class_final",
        "order_final", "family_final", "genus_final", "threatened",
        "range_area", "range_length", "num_spatial_records",
        "has_marine", "has_terrestrial", "has_freshwater", "yrcompiled",
        "log_range_area", "log_range_length", "log_num_spatial_records",
        "gbif_occurrence_count", "gbif_country_count", "gbif_geo_spread",
        "log_gbif_occurrence_count", "log_gbif_country_count", "log_gbif_geo_spread",
    ]
    df = pd.read_csv(DATA_PATH, usecols=lambda c: c in cols)
    for c in ["class_final", "order_final", "family_final", "genus_final"]:
        if c in df.columns:
            df[c] = df[c].fillna("Unknown").astype(str)
    return df

@st.cache_data(show_spinner=False)
def load_results():
    if RESULTS_PATH.exists():
        return pd.read_csv(RESULTS_PATH)
    return pd.DataFrame()

@st.cache_resource(show_spinner="Loading the Random Forest model…")
def load_model():
    return joblib.load(MODEL_PATH)


def pct(x):
    return f"{100*x:.1f}%"


def friendly_category(cat):
    return f"{cat} — {CATEGORY_LABELS.get(cat, cat)}"


df = load_data()
results_df = load_results()

st.markdown(
    """
    <div class="hero">
      <h1>🌿 Species Extinction Risk Explorer</h1>
      <p><b>A Needle in a Data Haystack</b> an interactive version of our project.</p>
      <p class="small-note">Explore taxonomic risk, geographic range effects, and a Random Forest extinction-risk demo.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Global filters
with st.sidebar:
    st.header("Explore the data")
    kingdom_note = st.caption("Threatened = VU, EN, or CR · Not threatened = LC or NT")
    min_group_size = st.slider("Minimum species per taxonomic group", 10, 500, 50, 10)
    st.divider()
    st.caption("Project data: IUCN taxonomy/spatial features + GBIF-derived features where available.")

# Main navigation mirrors the project questions.
tab0, tab1, tab2, tab3 = st.tabs([
    "🏠 Overview",
    "🧬 Taxonomic groups",
    "🗺️ Geographic range",
    "🤖 Risk predictor",
])

with tab0:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Species in final dataset", f"{df['scientific_name'].nunique():,}")
    c2.metric("Threatened species", f"{int(df['threatened'].sum()):,}", pct(df['threatened'].mean()))
    c3.metric("Taxonomic classes", f"{df['class_final'].nunique():,}")
    c4.metric("Species with range data", f"{int((pd.to_numeric(df['range_area'], errors='coerce').fillna(0) > 0).sum()):,}")

    st.subheader("IUCN conservation status in our dataset")
    cat_counts = (
        df[df["category_final"].isin(CATEGORY_ORDER)]
        .groupby("category_final")
        .size()
        .reindex(CATEGORY_ORDER, fill_value=0)
        .rename("species")
        .reset_index()
    )
    cat_counts["label"] = cat_counts["category_final"].map(friendly_category)
    chart = (
        alt.Chart(cat_counts)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("label:N", title=None, sort=[friendly_category(c) for c in CATEGORY_ORDER]),
            y=alt.Y("species:Q", title="Number of species"),
            tooltip=["label:N", alt.Tooltip("species:Q", format=",")],
            color=alt.Color("category_final:N", legend=None, scale=alt.Scale(range=["#e0c400", "#bda638", "#b6ad70", "#a58f8f", "#6b7f2b"])),
        )
        .properties(height=330)
    )
    st.altair_chart(chart, use_container_width=True)

    st.subheader("What the model comparison showed")
    if not results_df.empty:
        show = results_df.copy()
        show["roc_auc"] = show["roc_auc"].fillna(0)
        long = show.melt(
            id_vars="model",
            value_vars=["accuracy", "precision", "recall", "f1", "roc_auc"],
            var_name="metric",
            value_name="score",
        )
        mchart = (
            alt.Chart(long)
            .mark_bar()
            .encode(
                x=alt.X("metric:N", title=None),
                y=alt.Y("score:Q", scale=alt.Scale(domain=[0, 1]), title="Score"),
                xOffset="model:N",
                color=alt.Color("model:N", title="Model"),
                tooltip=["model:N", "metric:N", alt.Tooltip("score:Q", format=".3f")],
            )
            .properties(height=320)
        )
        st.altair_chart(mchart, use_container_width=True)
        st.caption("The project emphasized recall for threatened species, because missing a species at risk is especially costly.")

with tab1:
    st.subheader("Problem 1 : Are some taxonomic groups more threatened than others?")
    level = st.radio("Group by", ["Class", "Order"], horizontal=True)
    col = "class_final" if level == "Class" else "order_final"

    summary = (
        df.groupby(col)
        .agg(n_species=("scientific_name", "count"), threatened_rate=("threatened", "mean"))
        .reset_index()
    )
    summary = summary[(summary["n_species"] >= min_group_size) & (summary[col] != "Unknown")]
    summary = summary.sort_values(["threatened_rate", "n_species"], ascending=[False, False]).head(20)
    summary["threatened_pct"] = 100 * summary["threatened_rate"]

    tax_chart = (
        alt.Chart(summary)
        .mark_bar()
        .encode(
            y=alt.Y(f"{col}:N", sort="-x", title=level),
            x=alt.X("threatened_pct:Q", title="Threatened species (%)", scale=alt.Scale(domain=[0, 100])),
            tooltip=[
                alt.Tooltip(f"{col}:N", title=level),
                alt.Tooltip("n_species:Q", title="Species", format=","),
                alt.Tooltip("threatened_pct:Q", title="Threatened", format=".1f"),
            ],
            color=alt.Color("threatened_pct:Q", legend=None, scale=alt.Scale(range=["#d9d2a6", "#6b7f2b"])),
        )
        .properties(height=max(360, 22 * len(summary)))
    )
    st.altair_chart(tax_chart, use_container_width=True)

    if len(summary):
        top = summary.iloc[0]
        st.info(
            f"Among {level.lower()} groups with at least {min_group_size} species, "
            f"**{top[col]}** has the highest threatened share in this view: "
            f"**{top['threatened_pct']:.1f}%** across **{int(top['n_species']):,}** species."
        )

    st.markdown("#### Inspect one group")
    choices = summary[col].tolist()
    if choices:
        chosen = st.selectbox(f"Choose a {level.lower()}", choices)
        sub = df[df[col] == chosen]
        comp = (
            sub[sub["category_final"].isin(CATEGORY_ORDER)]
            .groupby("category_final")
            .size()
            .reindex(CATEGORY_ORDER, fill_value=0)
            .rename("species")
            .reset_index()
        )
        comp["share"] = comp["species"] / max(comp["species"].sum(), 1)
        comp["label"] = comp["category_final"].map(friendly_category)
        comp_chart = (
            alt.Chart(comp)
            .mark_bar()
            .encode(
                x=alt.X("share:Q", title="Share of species", axis=alt.Axis(format="%")),
                y=alt.Y("label:N", title=None, sort=[friendly_category(c) for c in CATEGORY_ORDER]),
                tooltip=["label:N", alt.Tooltip("species:Q", format=","), alt.Tooltip("share:Q", format=".1%")],
                color=alt.Color("category_final:N", legend=None, scale=alt.Scale(range=["#e0c400", "#bda638", "#b6ad70", "#a58f8f", "#6b7f2b"])),
            )
            .properties(height=240)
        )
        st.altair_chart(comp_chart, use_container_width=True)

with tab2:
    st.subheader("Problem 2 : Does geographic range size influence extinction risk?")
    range_df = df[pd.to_numeric(df["range_area"], errors="coerce").fillna(0) > 0].copy()
    range_df["log_range_area"] = pd.to_numeric(range_df["log_range_area"], errors="coerce")
    range_df = range_df.dropna(subset=["log_range_area"])

    if len(range_df) >= 5:
        range_df["range_bin"] = pd.qcut(range_df["log_range_area"], q=5, duplicates="drop")
        buckets = (
            range_df.groupby("range_bin", observed=False)
            .agg(
                threatened_percentage=("threatened", lambda x: 100 * x.mean()),
                n_species=("scientific_name", "count"),
                min_log_range=("log_range_area", "min"),
                max_log_range=("log_range_area", "max"),
            )
            .reset_index()
        )
        buckets["range_label"] = buckets.apply(
            lambda r: f"{r.min_log_range:.1f}–{r.max_log_range:.1f}", axis=1
        )
        line = (
            alt.Chart(buckets)
            .mark_line(point=True, strokeWidth=3)
            .encode(
                x=alt.X("range_label:N", title="Geographic range size: log(1 + range area)"),
                y=alt.Y("threatened_percentage:Q", title="Threatened species (%)", scale=alt.Scale(domain=[0, 100])),
                tooltip=[
                    "range_label:N",
                    alt.Tooltip("threatened_percentage:Q", format=".1f"),
                    alt.Tooltip("n_species:Q", format=","),
                ],
                color=alt.value("#6b7f2b"),
            )
            .properties(height=360)
        )
        st.altair_chart(line, use_container_width=True)

        first = buckets.iloc[0]["threatened_percentage"]
        last = buckets.iloc[-1]["threatened_percentage"]
        st.success(
            f"In the smallest-range bucket, **{first:.1f}%** of species are threatened; "
            f"in the largest-range bucket, only **{last:.1f}%** are threatened."
        )

        st.markdown("#### Explore the raw distribution")
        status_choice = st.multiselect(
            "Show species", ["Threatened", "Not threatened"], default=["Threatened", "Not threatened"]
        )
        mask = pd.Series(False, index=range_df.index)
        if "Threatened" in status_choice:
            mask |= range_df["threatened"] == 1
        if "Not threatened" in status_choice:
            mask |= range_df["threatened"] == 0
        plot_df = range_df.loc[mask, ["log_range_area", "threatened"]].copy()
        plot_df["status"] = np.where(plot_df["threatened"] == 1, "Threatened", "Not threatened")
        # Sample only for smoother browser rendering.
        if len(plot_df) > 12000:
            plot_df = plot_df.sample(12000, random_state=42)
        hist = (
            alt.Chart(plot_df)
            .mark_bar(opacity=0.75)
            .encode(
                x=alt.X("log_range_area:Q", bin=alt.Bin(maxbins=40), title="log(1 + range area)"),
                y=alt.Y("count():Q", title="Species (sampled for display)"),
                color=alt.Color("status:N", title="Status"),
                tooltip=["status:N", "count():Q"],
            )
            .properties(height=300)
        )
        st.altair_chart(hist, use_container_width=True)

with tab3:
    st.subheader("Problem 3 : Interactive extinction-risk prediction")
    st.caption(
        "This is a demo around the same Random Forest feature family used in the project. "
        "It is an educational model output, not a conservation assessment."
    )

    mode = st.radio("Prediction mode", ["Look up a real species", "What-if simulator"], horizontal=True)

    cat_features = ["class_final", "order_final", "family_final"]
    num_features = [
        "log_range_area", "log_range_length", "log_num_spatial_records",
        "has_marine", "has_terrestrial", "has_freshwater", "yrcompiled",
        "log_gbif_occurrence_count", "log_gbif_country_count", "log_gbif_geo_spread",
    ]

    input_row = None
    actual_category = None
    selected_name = None

    if mode == "Look up a real species":
        query = st.text_input("Search scientific or common name", placeholder="e.g. Panthera, turtle, orchid…")
        if query.strip():
            q = query.strip().lower()
            common = df["main_common_name"].fillna("").astype(str) if "main_common_name" in df.columns else pd.Series("", index=df.index)
            matches = df[
                df["scientific_name"].fillna("").str.lower().str.contains(q, regex=False)
                | common.str.lower().str.contains(q, regex=False)
            ].head(30)
            if matches.empty:
                st.warning("No matching species found. Try a broader search.")
            else:
                labels = []
                for i, r in matches.iterrows():
                    cn = r.get("main_common_name")
                    suffix = f" — {cn}" if pd.notna(cn) and str(cn).strip() else ""
                    labels.append((i, f"{r['scientific_name']}{suffix}"))
                chosen_label = st.selectbox("Choose a species", [x[1] for x in labels])
                chosen_idx = next(i for i, label in labels if label == chosen_label)
                r = df.loc[chosen_idx]
                selected_name = r["scientific_name"]
                actual_category = r["category_final"]
                input_row = {c: r.get(c, 0) for c in cat_features + num_features}

                a, b, c = st.columns(3)
                a.metric("IUCN category", friendly_category(actual_category))
                b.metric("Taxonomic class", str(r["class_final"]))
                area = pd.to_numeric(pd.Series([r.get("range_area", 0)]), errors="coerce").fillna(0).iloc[0]
                c.metric("Mapped range area", f"{area:,.0f}" if area > 0 else "No mapped value")
    else:
        c1, c2, c3 = st.columns(3)
        classes = sorted(x for x in df["class_final"].dropna().unique() if x != "Unknown")
        chosen_class = c1.selectbox("Class", classes)
        orders = sorted(x for x in df.loc[df["class_final"] == chosen_class, "order_final"].dropna().unique() if x != "Unknown")
        chosen_order = c2.selectbox("Order", orders if orders else ["Unknown"])
        fams = sorted(x for x in df.loc[(df["class_final"] == chosen_class) & (df["order_final"] == chosen_order), "family_final"].dropna().unique() if x != "Unknown")
        chosen_family = c3.selectbox("Family", fams if fams else ["Unknown"])

        r1, r2 = st.columns(2)
        log_range_area = r1.slider("log(1 + geographic range area)", 0.0, 12.0, 4.0, 0.1)
        log_range_length = r2.slider("log(1 + range boundary length)", 0.0, 15.0, 5.0, 0.1)
        r3, r4 = st.columns(2)
        log_num_spatial_records = r3.slider("log(1 + spatial records)", 0.0, 8.0, 1.0, 0.1)
        yrcompiled = r4.number_input("Spatial data compilation year (0 = unknown)", min_value=0, max_value=2030, value=0, step=1)

        st.markdown("**Habitats**")
        h1, h2, h3 = st.columns(3)
        marine = int(h1.checkbox("Marine"))
        terrestrial = int(h2.checkbox("Terrestrial", value=True))
        freshwater = int(h3.checkbox("Freshwater"))

        st.markdown("**GBIF occurrence features**")
        g1, g2, g3 = st.columns(3)
        occ = g1.number_input("Occurrence count", min_value=0, value=0, step=1)
        countries = g2.number_input("Countries observed", min_value=0, value=0, step=1)
        spread = g3.number_input("Geographic spread proxy", min_value=0.0, value=0.0, step=1.0)

        input_row = {
            "class_final": chosen_class,
            "order_final": chosen_order,
            "family_final": chosen_family,
            "log_range_area": log_range_area,
            "log_range_length": log_range_length,
            "log_num_spatial_records": log_num_spatial_records,
            "has_marine": marine,
            "has_terrestrial": terrestrial,
            "has_freshwater": freshwater,
            "yrcompiled": yrcompiled,
            "log_gbif_occurrence_count": math.log1p(occ),
            "log_gbif_country_count": math.log1p(countries),
            "log_gbif_geo_spread": math.log1p(spread),
        }

    if input_row is not None:
        model_input = pd.DataFrame([input_row])
        for c in cat_features:
            model_input[c] = model_input[c].fillna("Unknown").astype(str)
        for c in num_features:
            model_input[c] = pd.to_numeric(model_input[c], errors="coerce").fillna(0)

        model = load_model()
        probability = float(model.predict_proba(model_input[cat_features + num_features])[:, 1][0])
        prediction = int(probability >= 0.5)

        st.markdown("#### Model output")
        left, right = st.columns([1, 2])
        left.metric("Predicted threatened probability", f"{100*probability:.1f}%")
        left.metric("Predicted class", "Threatened" if prediction else "Not threatened")

        gauge_df = pd.DataFrame({"label": ["Risk score"], "probability": [probability]})
        gauge = (
            alt.Chart(gauge_df)
            .mark_bar(size=42, cornerRadius=8)
            .encode(
                x=alt.X("probability:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%"), title="Model probability"),
                y=alt.Y("label:N", title=None, axis=None),
                color=alt.condition(
                    alt.datum.probability >= 0.5,
                    alt.value("#d7b400"),
                    alt.value("#6b7f2b"),
                ),
                tooltip=[alt.Tooltip("probability:Q", format=".1%")],
            )
            .properties(height=110)
        )
        right.altair_chart(gauge, use_container_width=True)

        if selected_name:
            actual_threat = int(actual_category in ["VU", "EN", "CR"])
            if actual_threat == prediction:
                st.success(f"For **{selected_name}**, the model's binary prediction agrees with the recorded IUCN threatened/not-threatened grouping.")
            else:
                st.warning(f"For **{selected_name}**, the model's binary prediction differs from the recorded IUCN threatened/not-threatened grouping.")

        st.caption(
            "Important: this probability reflects patterns in the project dataset and model. "
            "It should not be interpreted as an official IUCN status or a real-world conservation decision."
        )

st.divider()
st.caption("· Noam Marcu · Orli Nagar · Shahar Sammy")
