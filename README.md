# EpiWatch — Epidemic Spread Prediction System

Codecure AI Hackathon 2026 | SPIRIT IIT BHU Varanasi
Team ShadowHealers — Sant Longowal Institute of Engineering and Technology

---

## The Problem

Epidemic response has always been reactive. By the time health authorities recognize a rising curve, mobilize resources, and issue warnings — the outbreak has already taken hold. The goal of this project is to flip that. EpiWatch analyzes historical COVID-19 data to predict where outbreak risk is rising before cases spike, giving a 14 to 30 day early warning window.

---

## What the System Does

- Predicts outbreak risk (Low, Medium, High) for 201 countries using a trained XGBoost classifier
- Forecasts case counts for the next 30 days using Facebook Prophet
- Calculates the effective reproduction number Rt for every country — the key epidemiological signal indicating whether a disease is spreading or declining
- Flags countries showing simultaneous early warning signals before a spike occurs
- Explains every prediction using SHAP — showing exactly which factors drove each risk classification
- Visualizes everything through an interactive three-page Streamlit dashboard with a global choropleth risk map

---

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.10+ |
| Dashboard | Streamlit |
| Risk Classification | XGBoost |
| Forecasting | Facebook Prophet |
| Explainability | SHAP |
| Visualization | Plotly |
| Data Processing | Pandas, NumPy, Scikit-learn |
| Data Sources | Johns Hopkins CSSE, Our World in Data |

---

## Project Structure

```
epidemic-spread-prediction/
│
├── data/
│   ├── processed_covid_data.csv      # Merged and feature-engineered dataset
│   ├── latest_country_data.csv       # Latest risk scores per country
│   ├── feature_cols.json             # Feature column names used by model
│   ├── shap_importance.png           # SHAP feature importance plot
│   ├── shap_summary.png              # SHAP beeswarm summary plot
│   └── shap_country.png              # Country-level SHAP waterfall explanation
│
├── models/
│   └── xgb_model.pkl                 # Trained XGBoost classifier
│
├── notebooks/
│   └── Epidemic_Spread_Prediction.ipynb       # Full analysis notebook (Colab)
│
├── screenshots/
│   ├── global_overview.png
│   ├── country_deepdive.png
│   └── model_insights.png
│
├── app.py                            # Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/saur-v/epidemic-spread-prediction
cd epidemic-spread-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the dashboard
streamlit run app.py
```

Note: Large data files are included in the repository. If any file is missing, run the full analysis notebook in Google Colab to regenerate them and download to the appropriate folders.

---

## Dashboard Screenshots

### Global Risk Overview
![Global Overview](screenshots/global_overview.png)

### Country Deep Dive
![Country Deep Dive](screenshots/country_deepdive.png)

### Model Insights
![Model Insights](screenshots/model_insights.png)

---

---
## How It Works

**Step 1 — Data Ingestion**
Daily time-series case data is loaded from the Johns Hopkins CSSE dataset and merged with vaccination rates, testing capacity, hospitalization figures, and demographic indicators from Our World in Data. The two datasets are joined on country and date to produce one unified dataset covering 201 countries from January 2020 to August 2024.

**Step 2 — Feature Engineering**
The following signals are computed for every country and date:
- 7-day and 14-day rolling averages to smooth out reporting noise
- Daily growth rate and doubling time
- Effective reproduction number Rt — estimated as the ratio of this week's rolling average to last week's
- Cases per million for fair cross-country comparison
- Lag features capturing case counts 7, 14, and 21 days prior
- Acceleration measuring whether the trend is speeding up or slowing down
- Death rate as a proxy for variant severity

**Step 3 — Risk Classification with XGBoost**
A supervised classifier is trained on 110,000+ country-day records. The target label is derived by looking 14 days into the future — if cases more than doubled, the day is labeled High risk. If cases grew more than 30 percent, it is Medium. Otherwise Low. Sample weights are applied to ensure the model pays attention to rare outbreak events rather than defaulting to Low predictions.

**Step 4 — Forecasting with Prophet**
Facebook Prophet is trained per country on the 7-day rolling average of new cases. A log transform is applied before training to prevent negative forecasts. The model generates predicted case counts with confidence intervals for the next 30 days.

**Step 5 — Early Warning System**
A rule-based layer scans all countries for simultaneous warning signals. A country is flagged when two or more of the following are true at the same time: Rt above 1.2, growth rate above 3 percent, vaccination below 30 percent, positivity rate above 10 percent, or 7-day average accelerating above 14-day average by 20 percent.

**Step 6 — SHAP Explainability**
SHAP values are computed for every prediction, showing the individual contribution of each feature. This answers not just what the risk level is but why — making the model transparent and interpretable for public health applications.

---

## Key Findings

- Rt is the single strongest predictor of outbreak risk, accounting for 40 percent of prediction impact according to SHAP analysis
- Government stringency measures show a clear protective effect — higher restriction indices consistently reduce predicted outbreak risk
- Median age and hospital bed capacity are significant structural vulnerability factors independent of active case counts
- Rolling averages carry more predictive power than raw growth rate, confirming that trend direction matters more than single-day fluctuations

---

## Model Performance

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Low | 0.89 | 0.66 | 0.76 |
| Medium | 0.23 | 0.13 | 0.16 |
| High | 0.12 | 0.58 | 0.19 |
| Weighted Average | 0.78 | 0.60 | 0.67 |

High recall on the High risk class is prioritized over precision — in epidemic surveillance it is better to raise a false alarm than to miss a real outbreak.

---

## Dashboard Pages

**Global Overview**
World choropleth map colored by XGBoost risk classification, key metrics, top 10 countries by total cases, and early warning alerts table.

**Country Deep Dive**
Select any of 201 countries to view current risk level, Rt value, vaccination rate, 30-day Prophet forecast with confidence intervals, and Rt trend over time.

**Model Insights**
SHAP feature importance, SHAP beeswarm plot showing how feature values drive predictions, country-level waterfall explanation, risk distribution by continent, and full model documentation.

---

## Datasets Used

| Dataset | Source | Usage |
|---|---|---|
| COVID-19 Time Series | Johns Hopkins CSSE | Core daily case and death data |
| COVID-19 Global Dataset | Our World in Data | Vaccination, testing, hospitalization, demographics |

---

## Team ShadowHealers

Sant Longowal Institute of Engineering and Technology
Codecure AI Hackathon — SPIRIT 2026, IIT BHU Varanasi
