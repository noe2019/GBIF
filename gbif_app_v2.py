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

# Load and Display Biodiversity Data
st.subheader("Biodiversity Data Overview")
@st.cache_data
def load_data(region):
    # Example: Loading biodiversity data from GBIF API
    url = f'https://api.gbif.org/v1/occurrence/search?country=CA&stateProvince={region if region != "All provinces" else ""}&limit=500'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()["results"]
        return pd.DataFrame(data)
    return pd.DataFrame()

data = load_data(region)
if data.empty:
    st.write("Data not available. Ensure API connectivity or load a local dataset.")
else:
    species_list = np.append(["All species"], data['species'].unique())
    selected_species = st.sidebar.selectbox("Select Species", options=species_list, index=0)
    st.write("Sample Data:", data[['species', 'stateProvince', 'decimalLatitude', 'decimalLongitude']].head())  # Display relevant columns

# Load GeoJSON boundaries for Canadian provinces
@st.cache_data
def load_province_boundaries():
    boundaries = gpd.read_file("https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/canada.geojson")
    boundaries = boundaries[boundaries['name'].isin(["Quebec", "Ontario", "British Columbia", "Alberta"])]
    return boundaries

province_boundaries = load_province_boundaries()

# Coordinates for centering map on selected province
province_coords = {
    "Quebec": [52.9399, -73.5491],
    "Ontario": [51.2538, -85.3232],
    "British Columbia": [53.7267, -127.6476],
    "Alberta": [53.9333, -116.5765],
    "All provinces": [56.1304, -106.3468]  # Center on Canada
}

def plot_map(data, selected_species):
    # Filter data based on both selected region and species
    if region != "All provinces":
        data = data[data['stateProvince'] == region]  # Only show species in the selected province

    # Filter for selected species, or show all species if "All species" is selected
    if selected_species != "All species":
        data = data[data['species'] == selected_species]

    # If the filtered data is empty, display a message and return
    if data.empty:
        st.write(f"No data available for {selected_species} in {region}.")
        return

    # Convert to GeoDataFrame for plotting
    gdf = gpd.GeoDataFrame(
        data,
        geometry=gpd.points_from_xy(data['decimalLongitude'], data['decimalLatitude']),
        crs="EPSG:4326"
    )

    # Configure map center and zoom based on the selected region
    if region == "All provinces":
        center_coords = {"lat": 56.1304, "lon": -106.3468}
        zoom_level = 3
        boundary_data = province_boundaries  # Show all province boundaries
        title = "All Species Occurrence Map for Canada"
    else:
        center_coords = {"lat": province_coords[region][0], "lon": province_coords[region][1]}
        zoom_level = 6
        boundary_data = province_boundaries[province_boundaries['name'] == region]  # Only selected province boundary
        title = f"{selected_species if selected_species != 'All species' else 'All Species'} Occurrence Map for {region}"

    # Create Plotly map with species occurrences
    fig = px.scatter_mapbox(
        gdf, lat="decimalLatitude", lon="decimalLongitude", color="species" if selected_species == "All species" else None,
        color_discrete_sequence=px.colors.qualitative.Bold,
        mapbox_style="carto-positron", zoom=zoom_level, title=title,
        center=center_coords
    )
    
    # Add province or country boundaries without showing in the legend
    for _, boundary in boundary_data.iterrows():
        geometry = boundary['geometry']
        
        if geometry.type == 'Polygon':
            boundary_coords = geometry.exterior.coords
            fig.add_trace(go.Scattermapbox(
                mode="lines",
                lon=[point[0] for point in boundary_coords],
                lat=[point[1] for point in boundary_coords],
                line=dict(width=2, color="orange"),
                showlegend=False
            ))
        
        elif geometry.type == 'MultiPolygon':
            for polygon in geometry.geoms:
                boundary_coords = polygon.exterior.coords
                fig.add_trace(go.Scattermapbox(
                    mode="lines",
                    lon=[point[0] for point in boundary_coords],
                    lat=[point[1] for point in boundary_coords],
                    line=dict(width=2, color="orange"),
                    showlegend=False
                ))

    # Display the map with the configured settings
    st.plotly_chart(fig)

# Run plot_map function if data is not empty
if not data.empty:
    plot_map(data, selected_species)

# Recent Literature on Biodiversity Indicators
st.subheader(f"Recent Literature on Biodiversity Indicators for {selected_species} in {region}")
st.write("This section highlights key research findings on biodiversity indicators relevant to selected species and province.")

# Placeholder literature data with URLs
literature_data = {
    "keywords": ["climate resilience", "species richness", "phenological shifts", "biodiversity indicators"],
    "summary": [
        "Recent studies indicate a strong correlation between climate resilience and biodiversity indicators.",
        "Species richness serves as a critical measure for ecosystem health and resilience.",
        "Phenological shifts are increasingly observed as species respond to climate change."
    ],
    "urls": [
        "https://example.com/climate_resilience_study",
        "https://example.com/species_richness_report",
        "https://example.com/phenological_shifts_research"
    ]
}
st.write("**Key Research Topics**:", ", ".join(literature_data["keywords"]))
for i, (summary, url) in enumerate(zip(literature_data["summary"], literature_data["urls"])):
    st.markdown(f"{i+1}. [{summary}]({url})")

# Final notes
st.write("""
**Note**: This prototype provides a comprehensive view of biodiversity data with provincial and species-specific insights. 
Future updates will integrate more live data and climate models for each species and region.
""")