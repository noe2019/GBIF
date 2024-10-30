import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import requests

# Streamlit App Layout
st.set_page_config(page_title="Biodiversity and Ecosystem Resilience Dashboard", layout="wide")

# Title and Overview
st.title("🌎 Biodiversity and Ecosystem Resilience Dashboard")
st.write("""
A comprehensive dashboard to analyze biodiversity data, monitor ecosystem resilience, and assess climate change impacts across Canadian provinces. 
This tool uses open-source data to support data-driven decision-making for ecosystem management and climate adaptation planning.
""")

# Sidebar for User Input
st.sidebar.header("🔍 Filter Options")
region = st.sidebar.selectbox("Select Province", ("All provinces", "Quebec", "Ontario", "British Columbia", "Alberta"))
indicator = st.sidebar.selectbox("Select Indicator", ("Species Richness", "Phenological Shift", "Climate Resilience", "Ecosystem Integrity"))
year_range = st.sidebar.slider("Select Year Range", 2000, datetime.now().year, (2010, 2020))

# Load Sample Data
@st.cache_data
def load_biodiversity_data():
    # Placeholder: replace with actual data loading logic
    return pd.DataFrame({
        "species": ["Species A", "Species B", "Species C", "Species D"],
        "region": ["Quebec", "Ontario", "British Columbia", "Alberta"],
        "richness": [200, 150, 180, 120],
        "phenological_shift": [5, 3, 7, 4],
        "climate_resilience": [0.8, 0.6, 0.75, 0.7],
        "ecosystem_integrity": [0.9, 0.85, 0.8, 0.78],
        "year": [2010, 2011, 2012, 2013]
    })

data = load_biodiversity_data()

# Load GeoJSON boundaries for Canadian provinces
@st.cache_data
def load_province_boundaries():
    boundaries = gpd.read_file("https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/canada.geojson")
    boundaries = boundaries[boundaries['name'].isin(["Quebec", "Ontario", "British Columbia", "Alberta"])]
    return boundaries

province_boundaries = load_province_boundaries()

# Define region coordinates
province_coords = {
    "Quebec": [52.9399, -73.5491],
    "Ontario": [51.2538, -85.3232],
    "British Columbia": [53.7267, -127.6476],
    "Alberta": [53.9333, -116.5765],
    "All provinces": [56.1304, -106.3468]
}

# Function to Plot Maps and Graphs based on Selected Indicator
def plot_indicator(indicator, region, year_range):
    if indicator == "Species Richness":
        st.subheader("Species Richness Map")
        
        # Map: Species Richness Heatmap
        fig = px.choropleth(
            data_frame=data,
            locations="region",
            locationmode="geojson-id",
            color="richness",
            hover_name="species",
            color_continuous_scale="Viridis",
            range_color=[100, 200],
            title="Species Richness across Provinces"
        )
        
        fig.update_geos(fitbounds="locations", visible=False)
        st.plotly_chart(fig)
        
        # Richness by Year
        st.subheader("Species Richness Over Time")
        richness_over_time = data.groupby("year")["richness"].mean().reset_index()
        fig = px.line(richness_over_time, x="year", y="richness", title="Average Species Richness Over Time")
        st.plotly_chart(fig)

    elif indicator == "Phenological Shift":
        st.subheader("Phenological Shifts Map")

        # Placeholder for Phenological Shifts Visualization
        fig = px.scatter_mapbox(
            data_frame=data,
            lat=province_coords[region][0] if region != "All provinces" else 56.1304,
            lon=province_coords[region][1] if region != "All provinces" else -106.3468,
            size="phenological_shift",
            color="region",
            mapbox_style="carto-positron",
            zoom=4,
            title="Phenological Shifts by Province"
        )
        st.plotly_chart(fig)
        
        # Time Series for Phenological Shift
        st.subheader("Phenological Shifts Over Time")
        shifts_over_time = data.groupby("year")["phenological_shift"].mean().reset_index()
        fig = px.line(shifts_over_time, x="year", y="phenological_shift", title="Average Phenological Shifts Over Time")
        st.plotly_chart(fig)

    elif indicator == "Climate Resilience":
        st.subheader("Climate Resilience Map")
        
        # Placeholder for Climate Resilience Visualization
        fig = px.choropleth(
            data_frame=data,
            locations="region",
            locationmode="geojson-id",
            color="climate_resilience",
            color_continuous_scale="Bluered",
            range_color=[0, 1],
            title="Climate Resilience Index by Province"
        )
        fig.update_geos(fitbounds="locations", visible=False)
        st.plotly_chart(fig)
        
        # Radar Chart for Resilience Attributes
        st.subheader("Climate Resilience Attributes by Province")
        fig = go.Figure()
        provinces = data["region"].unique()
        for province in provinces:
            province_data = data[data["region"] == province]
            fig.add_trace(go.Scatterpolar(
                r=[province_data["richness"].values[0], province_data["ecosystem_integrity"].values[0]],
                theta=["Richness", "Integrity"],
                fill="toself",
                name=province
            ))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True)), title="Climate Resilience Attributes")
        st.plotly_chart(fig)

    elif indicator == "Ecosystem Integrity":
        st.subheader("Ecosystem Integrity Map")
        
        # Placeholder for Ecosystem Integrity Visualization
        fig = px.choropleth(
            data_frame=data,
            locations="region",
            locationmode="geojson-id",
            color="ecosystem_integrity",
            color_continuous_scale="Greens",
            range_color=[0.7, 1],
            title="Ecosystem Integrity by Province"
        )
        fig.update_geos(fitbounds="locations", visible=False)
        st.plotly_chart(fig)
        
        # Integrity Over Time
        st.subheader("Ecosystem Integrity Over Time")
        integrity_over_time = data.groupby("year")["ecosystem_integrity"].mean().reset_index()
        fig = px.line(integrity_over_time, x="year", y="ecosystem_integrity", title="Ecosystem Integrity Over Time")
        st.plotly_chart(fig)

# Plot based on the selected indicator
plot_indicator(indicator, region, year_range)