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

# Set up Streamlit page
st.set_page_config(page_title="Biodiversity & Ecosystem Resilience Dashboard", layout="wide")

# Title and Overview
st.title("🌍 Biodiversity & Ecosystem Resilience Dashboard")
st.markdown("""
This dashboard provides insights into biodiversity trends, ecosystem resilience, and climate change impacts across Canadian provinces.
Using open-source data, it aids in data-driven decision-making for ecosystem management and climate adaptation planning.
""")

# Sidebar for User Inputs
st.sidebar.header("🔍 Filter Options")
region = st.sidebar.selectbox("Select Province", ["Quebec", "Ontario", "British Columbia", "Alberta", "All"])
indicator = st.sidebar.selectbox("Select Indicator", ["Species Richness", "Phenological Shift", "Climate Resilience", "Ecosystem Integrity"])
year_range = st.sidebar.slider("Select Year Range", 2000, datetime.now().year, (2010, 2020))

# Load Biodiversity Data
@st.cache_data
def load_data(region):
    url = f'https://api.gbif.org/v1/occurrence/search?country=CA&stateProvince={region}&limit=500'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()["results"]
        return pd.DataFrame(data)
    return pd.DataFrame()

data = load_data(region)
st.subheader("Biodiversity Data Overview")

if data.empty:
    st.warning("Data not available. Ensure API connectivity or load a local dataset.")
else:
    species_list = data['species'].unique()
    selected_species = st.sidebar.selectbox("Select Species", options=species_list, index=0)
    st.write("Sample Data:", data.head())

# Load Canadian Provinces Boundaries
@st.cache_data
def load_province_boundaries():
    boundaries = gpd.read_file("https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/canada.geojson")
    return boundaries[boundaries['name'].isin(["Quebec", "Ontario", "British Columbia", "Alberta"])]

province_boundaries = load_province_boundaries()

# Map Species Occurrences with Dynamic Boundaries
st.subheader("Species Occurrence Map")

def plot_map(data, selected_species):
    species_data = data[data['species'] == selected_species] if selected_species else data
    gdf = gpd.GeoDataFrame(
        species_data, 
        geometry=gpd.points_from_xy(species_data['decimalLongitude'], species_data['decimalLatitude']),
        crs="EPSG:4326"
    )
    
    fig = px.scatter_mapbox(
        gdf, lat="decimalLatitude", lon="decimalLongitude", color="species",
        mapbox_style="carto-positron", zoom=4, title=f"{selected_species} Occurrence Map for {region}",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    
    for _, boundary in province_boundaries.iterrows():
        geometry = boundary['geometry']
        if geometry.type == 'Polygon':
            boundary_coords = geometry.exterior.coords
            fig.add_trace(go.Scattermapbox(
                mode="lines",
                lon=[point[0] for point in boundary_coords],
                lat=[point[1] for point in boundary_coords],
                line=dict(width=2, color="orange"),
                name=boundary['name']
            ))
        elif geometry.type == 'MultiPolygon':
            for polygon in geometry.geoms:  # Corrected: using `.geoms` to iterate over individual polygons in a MultiPolygon
                boundary_coords = polygon.exterior.coords
                fig.add_trace(go.Scattermapbox(
                    mode="lines",
                    lon=[point[0] for point in boundary_coords],
                    lat=[point[1] for point in boundary_coords],
                    line=dict(width=2, color="orange"),
                    name=boundary['name']
                ))

    st.plotly_chart(fig)

if not data.empty:
    plot_map(data, selected_species)

# Biodiversity Indicator Trends by Species and Province
st.subheader(f"Biodiversity Indicator Trends for {selected_species} in {region}")
if indicator == "Species Richness":
    st.write("**Species Richness Analysis**")
    richness = data.groupby('year')['species'].nunique()
    fig = px.line(
        x=richness.index, y=richness.values,
        labels={'x': 'Year', 'y': 'Species Richness'},
        title=f"Species Richness in {region} Over Time",
        line_shape='spline', markers=True
    )
    fig.update_traces(line=dict(color="green", width=4), marker=dict(size=8))
    fig.update_layout(title_font_size=20, xaxis_title_font_size=16, yaxis_title_font_size=16)
    st.plotly_chart(fig)

elif indicator == "Phenological Shift":
    st.write("**Phenological Shift Analysis**")
    years = np.arange(year_range[0], year_range[1] + 1)
    phenology_data = pd.DataFrame({
        'Year': years,
        'Shift_Days': np.random.normal(5, 2, len(years))
    })
    fig = px.line(
        phenology_data, x='Year', y='Shift_Days',
        title=f"Phenological Shifts in {selected_species} in {region} Over Years",
        labels={'Year': 'Year', 'Shift_Days': 'Shift in Days'},
        line_shape='spline', markers=True
    )
    fig.update_traces(line=dict(color="blue", width=4), marker=dict(size=8))
    st.plotly_chart(fig)

# Species Occurrence Prediction Model
st.subheader(f"Species Occurrence Prediction Model for {selected_species}")
if 'year' in data.columns and 'species' in data.columns:
    data['occurrence'] = np.where(data['species'] == selected_species, 1, 0)
    features = data[['decimalLatitude', 'decimalLongitude', 'year']]
    labels = data['occurrence']
    
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"**Model Accuracy**: {accuracy:.2f}")
    st.write(f"This model predicts occurrences of {selected_species} in {region} based on geographic and temporal data.")

# Future Climate Scenario Simulation for Species
st.subheader(f"Future Climate Scenario Simulation for {selected_species} in {region}")
years_future = np.arange(2025, 2051)
scenario_data = pd.DataFrame({
    'Year': years_future,
    'Occurrence_Probability': np.random.uniform(0.5, 1, len(years_future))
})
fig = px.line(
    scenario_data, x='Year', y='Occurrence_Probability',
    title=f"Projected Occurrence Probability for {selected_species} under Climate Change in {region}",
    labels={'Year': 'Year', 'Occurrence Probability': 'Occurrence Probability'},
    line_shape='spline', markers=True
)
fig.update_traces(line=dict(color="purple", width=4), marker=dict(size=8))
st.plotly_chart(fig)

# Recent Literature on Biodiversity Indicators for Species and Province
st.subheader(f"Recent Literature on Biodiversity Indicators for {selected_species} in {region}")
literature_data = {
    "keywords": ["climate resilience", "species richness", "phenological shifts", "biodiversity indicators"],
    "summary": [
        "Studies indicate a strong correlation between climate resilience and biodiversity.",
        "Species richness is a critical measure for ecosystem health.",
        "Phenological shifts are observed as species respond to climate change."
    ]
}
st.write("**Key Research Topics**:", ", ".join(literature_data["keywords"]))
for i, summary in enumerate(literature_data["summary"]):
    st.write(f"{i+1}. {summary}")

# Footer and About
st.sidebar.header("About")
st.sidebar.markdown("""
Developed to support biodiversity and ecosystem resilience analysis for Canadian provinces.
Future updates will integrate live data and advanced climate models.
""")
