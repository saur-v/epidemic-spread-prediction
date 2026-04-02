import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
from prophet import Prophet
from PIL import Image


st.set_page_config(
    page_title="EpiWatch — Epidemic Spread Prediction",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# data loading
@st.cache_data
def load_data():
    merged_df = pd.read_csv('data/processed_covid_data.csv', parse_dates=['date'])
    latest_df = pd.read_csv('data/latest_country_data.csv')
    with open('data/feature_cols.json') as f:
        feature_cols = json.load(f)
    return merged_df, latest_df, feature_cols

@st.cache_resource
def load_model():
    return joblib.load('models/xgb_model.pkl')

merged_df, latest_df, feature_cols = load_data()
xgb_model = load_model()

st.sidebar.title("EpiWatch")
st.sidebar.markdown("*Epidemic Spread Prediction System*")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["Global Overview", "Country Deep Dive", "Model Insights"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Team ShadowHealers**")

# Page 1 — Global Overview
if page == "Global Overview":
    st.title("Global Epidemic Overview")
    st.markdown("Real-time outbreak risk assessment")

    col1, col2, col3, col4 = st.columns(4)

    total_countries = latest_df['location'].nunique()
    high_risk = len(latest_df[latest_df['xgb_risk_label'] == 'High'])
    medium_risk = len(latest_df[latest_df['xgb_risk_label'] == 'Medium'])
    warned = len(latest_df[latest_df['warning'].str.startswith('WARNING', na=False)])

    col1.metric("Countries Tracked", total_countries)
    col2.metric("High Risk", high_risk)
    col3.metric("Medium Risk", medium_risk)
    col4.metric("Early Warnings", warned)

    st.markdown("---")

    st.subheader("Global Outbreak Risk Map")
    fig_map = px.choropleth(
    latest_df,
    locations='iso_code',
    color='xgb_risk_label',
    hover_name='location',
    hover_data={
        'Rt': ':.2f',
        'people_fully_vaccinated_per_hundred': ':.1f',
        'xgb_risk_label': True,    # show XGBoost risk
        'warning': True,            # also show warning signals
        'iso_code': False
    },
    color_discrete_map={
        'Low': '#2ecc71',
        'Medium': '#f39c12',
        'High': '#e74c3c'
    },
    title='Global Outbreak Risk — Powered by XGBoost'
)
    fig_map.update_layout(
        geo=dict(showframe=False, showcoastlines=True),
        height=500,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    st.plotly_chart(fig_map, use_container_width=True)

    st.caption(
    "Map color represents XGBoost model risk classification. "
    "Warning column indicates rule-based early warning signals. "
    "A country may show High risk from the model without active "
    "warning signals — this reflects learned historical outbreak patterns."
)

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Top 10 Countries by Total Cases")
        latest = merged_df.groupby('location')['total_cases'].max().reset_index()
        top10 = latest.sort_values('total_cases', ascending=False).head(10)
        fig_bar = px.bar(
            top10,
            x='location',
            y='total_cases',
            color='total_cases',
            color_continuous_scale='Reds'
        )
        fig_bar.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_right:
        st.subheader("Early Warning Alerts")
        warned_df = latest_df[latest_df['warning'].str.startswith('WARNING', na=False)][
            ['location', 'xgb_risk_label', 'Rt', 'warning']
        ].sort_values('xgb_risk_label', ascending=False).reset_index(drop=True)

        def color_risk(val):
            if val == 'High':
                return 'background-color: #ffcccc'
            elif val == 'Medium':
                return 'background-color: #fff3cc'
            return ''

        st.dataframe(
            warned_df.style.map(color_risk, subset=['xgb_risk_label']),
            height=350,
            use_container_width=True
        )

# Page 2 — Country Deep Dive
elif page == "Country Deep Dive":
    st.title("Country Deep Dive")
    st.markdown("Select any country to see forecast, Rt trend, and risk assessment")

    countries = sorted(merged_df['location'].unique())
    selected = st.selectbox("Select Country", countries, index=countries.index('India'))

    country_data = merged_df[merged_df['location'] == selected].copy()
    latest_country = latest_df[latest_df['location'] == selected]

    if not latest_country.empty:
        risk = latest_country['xgb_risk_label'].values[0]
        warning = latest_country['warning'].values[0]
        rt_val = latest_country['Rt'].values[0]
        vax_val = latest_country['people_fully_vaccinated_per_hundred'].values[0]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Risk Level", risk)
        col2.metric("Rt Value", f"{rt_val:.2f}")
        col3.metric("Vaccination %", f"{vax_val:.1f}%")
        col4.metric("Status", "Active Warning" if warning.startswith('WARNING') else "Stable")

        if warning.startswith('WARNING'):
            st.warning(f"**Early Warning Detected:** {warning}")

    st.markdown("---")

    st.subheader(f"Case Trend — {selected}")
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=country_data['date'],
        y=country_data['rolling_7day'],
        name='7-Day Rolling Avg',
        line=dict(color='#3498db', width=2)
    ))
    fig_trend.add_trace(go.Scatter(
        x=country_data['date'],
        y=country_data['rolling_14day'],
        name='14-Day Rolling Avg',
        line=dict(color='#e67e22', width=2, dash='dash')
    ))
    fig_trend.update_layout(
        height=350,
        xaxis_title='Date',
        yaxis_title='New Cases',
        hovermode='x unified'
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    st.markdown("---")

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader(f"30-Day Forecast — {selected}")
        with st.spinner("Running Prophet forecast..."):
            try:
                prophet_df = country_data[['date', 'rolling_7day']].rename(
                    columns={'date': 'ds', 'rolling_7day': 'y'}
                )
                prophet_df['y'] = prophet_df['y'].clip(lower=0)
                prophet_df['y'] = np.log1p(prophet_df['y'])

                m = Prophet(
                    yearly_seasonality=True,
                    weekly_seasonality=False,
                    daily_seasonality=False,
                    changepoint_prior_scale=0.05
                )
                m.fit(prophet_df)
                future = m.make_future_dataframe(periods=30)
                forecast = m.predict(future)

                forecast['yhat'] = np.expm1(forecast['yhat']).clip(0)
                forecast['yhat_lower'] = np.expm1(forecast['yhat_lower']).clip(0)
                forecast['yhat_upper'] = np.expm1(forecast['yhat_upper']).clip(0)
                actual_y = np.expm1(prophet_df['y'])

                fig_forecast = go.Figure()
                fig_forecast.add_trace(go.Scatter(
                    x=prophet_df['ds'], y=actual_y,
                    name='Actual', line=dict(color='#3498db')
                ))
                fig_forecast.add_trace(go.Scatter(
                    x=forecast['ds'], y=forecast['yhat'],
                    name='Forecast', line=dict(color='#e74c3c', dash='dash')
                ))
                fig_forecast.add_trace(go.Scatter(
                    x=pd.concat([forecast['ds'], forecast['ds'][::-1]]),
                    y=pd.concat([forecast['yhat_upper'], forecast['yhat_lower'][::-1]]),
                    fill='toself',
                    fillcolor='rgba(231,76,60,0.1)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='Confidence Interval'
                ))
                fig_forecast.update_layout(
                    height=350,
                    xaxis_title='Date',
                    yaxis_title='New Cases (7-day avg)',
                    hovermode='x unified'
                )
                st.plotly_chart(fig_forecast, use_container_width=True)
            except Exception as e:
                st.error(f"Forecast error: {e}")

    with col_r:
        st.subheader(f"Rt Trend — {selected}")
        fig_rt = go.Figure()
        fig_rt.add_trace(go.Scatter(
            x=country_data['date'],
            y=country_data['Rt'],
            name='Rt',
            line=dict(color='#9b59b6', width=2)
        ))
        fig_rt.add_hline(
            y=1.0,
            line_dash='dash',
            line_color='red',
            annotation_text='Rt = 1 (Threshold)'
        )
        fig_rt.update_layout(
            height=350,
            xaxis_title='Date',
            yaxis_title='Rt Value',
            yaxis_range=[0, 3],
            hovermode='x unified'
        )
        st.plotly_chart(fig_rt, use_container_width=True)

# Page 3 — Model Insights
elif page == "Model Insights":
    st.title("Model Insights")
    st.markdown("Understanding what drives outbreak risk predictions")

    st.subheader("XGBoost Feature Importance")

    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=True)

    fig_imp = px.bar(
        importance_df,
        x='importance',
        y='feature',
        orientation='h',
        color='importance',
        color_continuous_scale='Reds',
        title='Feature Importance'
    )
    fig_imp.update_layout(height=450)
    st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Risk Distribution by Continent")
        continent_risk = latest_df.groupby(
            ['continent', 'xgb_risk_label']
        ).size().reset_index(name='count')

        fig_cont = px.bar(
            continent_risk,
            x='continent',
            y='count',
            color='xgb_risk_label',
            color_discrete_map={
                'Low': '#2ecc71',
                'Medium': '#f39c12',
                'High': '#e74c3c'
            },
            barmode='group',
            title='Risk Level Distribution by Continent'
        )
        fig_cont.update_layout(height=400)
        st.plotly_chart(fig_cont, use_container_width=True)

    with col2:
        st.subheader("Model Information")
        st.markdown("""
        **Forecasting Model — Facebook Prophet**
        - Time series decomposition
        - Captures yearly seasonality
        - Log transform to prevent negative predictions
        - 30-day forward forecast per country

        ---

        **Risk Classification — XGBoost**
        - Trained on 110,000+ country-day records
        - 12 epidemiological features
        - Class weights to handle imbalance
        - Time-based train/test split (pre/post June 2022)

        ---

        **Early Warning System**
        - Rule-based signal detection
        - Triggers on 2+ concurrent warning signals
        - Signals: Rising Rt, High Growth Rate,
          Low Vaccination, High Positivity Rate,
          Accelerating Cases
        """)

    st.markdown("---")
    st.subheader("How Risk Score is Calculated")
    st.markdown("""
    | Signal | Threshold | Weight |
    |---|---|---|
    | Rt (Reproduction Number) | > 1.2 rising | Highest (22%) |
    | 14-Day Rolling Average | Trend direction | High (12%) |
    | 7-Day Rolling Average | Short term trend | Medium (8%) |
    | Median Age | Population vulnerability | Medium (8%) |
    | Cases per Million | Disease burden | Medium (8%) |
    | Vaccination Rate | Population protection | Lower (7%) |
    """)



    st.markdown("---")
    st.subheader("SHAP Explainability — Why Does the Model Flag Countries?")
    st.markdown("""
    SHAP (SHapley Additive exPlanations) shows the actual contribution 
    of each feature to every individual prediction — making the model 
    transparent and interpretable.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Feature Impact on High Risk Predictions**")
        img1 = Image.open('data/shap_importance.png')
        st.image(img1, use_container_width=True)

    with col2:
        st.markdown("**How Feature Values Drive Risk Up or Down**")
        img2 = Image.open('data/shap_summary.png')
        st.image(img2, use_container_width=True)

    st.markdown("---")
    st.subheader("Country-Level Explanation")
    st.markdown("*Why is this specific country flagged as High Risk?*")
    img3 = Image.open('data/shap_country.png')
    st.image(img3, use_container_width=True)
    st.caption(
        "Each bar shows how much a feature pushed the risk score up (red) or down (blue)"
)