# Import libraries
import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import requests

# Streamlit page configuration
st.set_page_config(page_title="Biodiversity Monitoring & Climate Resilience Dashboard", layout="wide")

# Title and App Description
st.title("🌍 Biodiversity Monitoring & Climate Resilience Dashboard")
st.markdown("""
This dashboard offers insights into biodiversity trends, ecosystem resilience, and the impact of climate change across regions.
It supports data-driven decision-making for ecosystem management and climate adaptation.
""")

# Sidebar setup for user input options
st.sidebar.header("🔍 Filter Options")
region = st.sidebar.selectbox("Select Region", ["Quebec", "Ontario", "British Columbia", "Alberta", "All"])
indicator = st.sidebar.selectbox("Select Indicator", ["Species Richness", "Phenological Shift", "Climate Resilience", "Ecosystem Integrity"])
year_range = st.sidebar.slider("Select Year Range", 2000, datetime.now().year, (2010, 2020))

# Load biodiversity data from an open API
@st.cache_data
def load_biodiversity_data(region):
    url = f'https://api.gbif.org/v1/occurrence/search?country=CA&stateProvince={region}&limit=500'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()["results"]
        return pd.DataFrame(data)
    return pd.DataFrame()

data = load_biodiversity_data(region)
st.subheader("Biodiversity Data Overview")

# Display data or warning if data is empty
if data.empty:
    st.warning("No data available. Please check the region or your internet connection.")
else:
    st.write("Sample Data:", data.head())

# Load Canadian province boundaries for mapping
@st.cache_data
def load_province_boundaries():
    boundaries = gpd.read_file("https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/canada.geojson")
    return boundaries[boundaries['name'].isin(["Quebec", "Ontario", "British Columbia", "Alberta"])]

province_boundaries = load_province_boundaries()

# Set up tab structure
tabs = st.tabs(["Ecosystem Indicators", "Literature Review", "Regional Variability", 
                "Phenological Changes", "Species Occurrence Modeling", "Advanced Modeling", "Communication Tools"])

# Ecosystem Indicators Tab
with tabs[0]:
    st.header("Ecosystem State and Climate Resilience Indicators")
    if indicator == "Species Richness":
        richness = data.groupby('year')['species'].nunique()
        fig = px.line(
            x=richness.index, y=richness.values,
            labels={'x': 'Year', 'y': 'Species Richness'},
            title=f"Species Richness in {region} Over Time",
            line_shape='spline', markers=True
        )
        st.plotly_chart(fig)

    elif indicator == "Climate Resilience":
        # Placeholder for climate resilience data
        resilience_data = pd.DataFrame({
            'Year': np.arange(year_range[0], year_range[1] + 1),
            'Resilience_Score': np.random.uniform(0.6, 1.0, year_range[1] - year_range[0] + 1)
        })
        fig = px.line(
            resilience_data, x='Year', y='Resilience_Score',
            title=f"Climate Resilience in {region} Over Time",
            line_shape='spline', markers=True
        )
        st.plotly_chart(fig)

# Literature Review Tab
with tabs[1]:
    st.header("Literature Review on Biodiversity Indicators")
    st.markdown("""
    - **Species Richness**: Commonly used to assess ecosystem health.
    - **Phenological Shifts**: Changes in timing of biological events due to climate.
    - **Climate Niche Models**: Predict species distribution changes with climate change.
    - **Ecological Integrity**: Measure of ecosystem health and stability.
    """)

# Regional Variability Tab
with tabs[2]:
    st.header("Regional and Annual Variability in Indicators")
    fig = px.scatter_mapbox(
        data, lat="decimalLatitude", lon="decimalLongitude", color="species",
        mapbox_style="carto-positron", zoom=4, title=f"Species Occurrence Map in {region}"
    )
    st.plotly_chart(fig)

# Phenological Changes, Climate Niches, and Ecological Integrity Tab
with tabs[3]:
    st.header("Phenological Changes, Climate Niches, and Ecological Integrity")
    phenology_data = pd.DataFrame({
        'Year': np.arange(year_range[0], year_range[1] + 1),
        'Shift_Days': np.random.normal(5, 2, year_range[1] - year_range[0] + 1)
    })
    fig = px.line(
        phenology_data, x='Year', y='Shift_Days',
        title=f"Phenological Shifts in {region} Over Years",
        labels={'Year': 'Year', 'Shift_Days': 'Shift in Days'},
        line_shape='spline', markers=True
    )
    st.plotly_chart(fig)

# Species Occurrence Modeling Tab
with tabs[4]:
    st.header("Species Occurrence Prediction Model")
    data['occurrence'] = np.where(data['species'] == 'selected_species', 1, 0)
    features = data[['decimalLatitude', 'decimalLongitude', 'year']]
    labels = data['occurrence']
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"**Model Accuracy**: {accuracy:.2f}")

# Advanced Modeling for Environmental Indicators Tab
with tabs[5]:
    st.header("Advanced Modeling for Environmental Indicators")
    st.write("Future projections based on climate scenarios to come.")

# Communication and Data Exploration Tools Tab
with tabs[6]:
    st.header("Data Exploration and Communication Tools")
    st.markdown("""
    Generate customizable reports, export data, and interactively explore trends.
    """)

# Footer and About Section
st.sidebar.header("About")
st.sidebar.markdown("""
Developed for analyzing biodiversity and resilience in Quebec.
Future updates will add advanced models and climate projections.
""")
