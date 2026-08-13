# Species Extinction Risk Explorer

Interactive companion app for the final project **A Needle in a Data Haystack — Predicting Species Extinction Risk**.

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## What the app contains

1. **Overview** — dataset summary and model comparison.
2. **Taxonomic groups** — interactive version of Problem 1.
3. **Geographic range** — interactive version of Problem 2.
4. **Risk predictor** — species lookup + Random Forest probability demo for Problem 3.

The predictor is for demonstration/education only and is not an official conservation assessment.

## Deploy to Streamlit Community Cloud

Push this project folder to GitHub, then deploy `streamlit_app.py` as the app entrypoint. Keep `requirements.txt`, `outputs/final_conservation_dataset.csv`, and `outputs/streamlit_rf_model.joblib` in the repository.
