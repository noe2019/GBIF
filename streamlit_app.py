# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
from pygbif import occurrences as occ
import wbdata
from datetime import datetime
import plotly.express as px
from sklearn.linear_model import LinearRegression
import geopandas as gpd

# Application Title and Description
st.title("Comprehensive Biodiversity and Climate Data Dashboard for Canada")

st.markdown("""
Explore Canada’s biodiversity and climate data across multiple provinces. This dashboard integrates open-source data from:

- **Global Biodiversity Information Facility (GBIF)**: Species occurrence data.
- **World Bank Climate Data**: Climate indicators and economic data.
- **iNaturalist** and **Ocean Biogeographic Information System (OBIS)**: Community-sourced biodiversity observations.
- **Environment and Climate Change Canada (ECCC)**: National climate data.

Get insights at both national and provincial levels to inform research, conservation efforts, and data-driven decisions.
""")

# Helper functions for data fetching and processing
def fetch_gbif_data(species, province):
    """Fetch species occurrence data from GBIF for a specific province in Canada."""
    try:
        occurrences = occ.search(scientificName=species, country="CA", stateProvince=province, limit=100)
        df = pd.json_normalize(occurrences['results'])[['species', 'decimalLatitude', 'decimalLongitude', 'eventDate']]
        df = df.rename(columns={'decimalLatitude': 'latitude', 'decimalLongitude': 'longitude'})
        return df
    except Exception as e:
        st.error(f"Error retrieving data from GBIF: {e}")
        return pd.DataFrame()

# Section 1: Biodiversity Data Analysis by Province
st.header("1. Biodiversity Data by Province")

species = st.text_input("Enter species name (e.g., 'Ursus arctos' for brown bear):")
province = st.selectbox("Select Province", ["Ontario", "Quebec", "British Columbia", "Alberta", "All Provinces"])

if st.button("Fetch Biodiversity Data"):
    gbif_data = fetch_gbif_data(species, province)
    if not gbif_data.empty:
        st.subheader(f"{species} Occurrences in {province}")
        st.map(gbif_data[['latitude', 'longitude']].dropna())
        st.write(gbif_data)
    else:
        st.warning("No data found for this species in the selected province.")

# Section 2: Climate and Environmental Indicators by Province
st.header("2. Climate and Environmental Indicators by Province")

st.markdown("""
Analyze key climate indicators over time and by province:
- **CO₂ Emissions** and **Forest Area** for environmental monitoring.
- **GDP** as an economic factor influencing ecological policies.

Select an indicator and visualize historical data trends for each province.
""")

# Define the indicators of interest with descriptive names for each code
indicators = {
    "AG.LND.FRST.ZS": "Forest area (% of land area)", 
    "NY.GDP.MKTP.CD": "GDP (current US$)",
    "EN.ATM.CO2E.PC": "CO₂ emissions (metric tons per capita)",
    "AG.LND.TOTL.K2": "Land area (sq. km)",
    "AG.PRD.CROP.XD": "Agricultural production index",
    "SP.POP.TOTL": "Total population",
    "EG.ELC.RNEW.ZS": "Renewable electricity output (% of total)",
    "EN.ATM.GHGT.KT.CE": "Total greenhouse gas emissions (kt of CO₂ equivalent)"
}
start_year = st.slider("Select Start Year", 1960, 2020, 1990)

# Fetch World Bank data for a specific indicator
def fetch_wb_data(indicator_code, country="CAN"):
    """Fetch World Bank climate/environmental data for a specified indicator and country."""
    try:
        data = wbdata.get_dataframe({indicator_code: indicators[indicator_code]}, country=country)
        data.reset_index(inplace=True)
        data['date'] = pd.to_datetime(data['date'], errors='coerce')
        data = data[data['date'].dt.year >= start_year]
        return data
    except Exception as e:
        st.error(f"Error retrieving World Bank data: {e}")
        return pd.DataFrame()

# Initialize climate_data to None
climate_data = None

# Select indicator and retrieve data
selected_indicator = st.selectbox("Select Climate Indicator", list(indicators.values()))
indicator_code = [code for code, desc in indicators.items() if desc == selected_indicator][0]

if st.button("Download Climate Data"):
    climate_data = fetch_wb_data(indicator_code)
    if not climate_data.empty:
        st.subheader(f"{selected_indicator} Over Time in Canada")
        st.dataframe(climate_data)

        # Plot the data using the correct column name
        fig = px.line(climate_data, x='date', y=indicators[indicator_code], title=f"{selected_indicator} Over Time")
        st.plotly_chart(fig)

# Section 3: Climate Change Impact Projections
st.header("3. Climate Change Impact Projections")

st.markdown("""
Forecast future climate trends using historical data with a simple linear regression model. 
Select an indicator to project its trend over the next 10 years.
""")

indicator_for_projection = st.selectbox("Select Indicator for Projection", list(indicators.values()))

if st.button("Generate Future Projection"):
    if climate_data is not None and not climate_data.empty:
        try:
            # Prepare data for modeling
            data = climate_data[['date', indicator_code]].dropna().reset_index(drop=True)
            X = np.arange(len(data)).reshape(-1, 1)
            y = data[indicator_code].values.reshape(-1, 1)

            # Train a linear regression model
            model = LinearRegression()
            model.fit(X, y)

            # Predict the next 10 years
            future_years = np.arange(len(X), len(X) + 10).reshape(-1, 1)
            future_pred = model.predict(future_years)

            # Combine past and projected data
            future_dates = pd.date_range(start=data['date'].iloc[-1] + pd.DateOffset(years=1), periods=10, freq='Y')
            future_df = pd.DataFrame({indicator_code: future_pred.flatten()}, index=future_dates)
            combined_df = pd.concat([data.set_index('date'), future_df])

            # Plot historical data and projections
            fig = px.line(combined_df, y=indicator_code, title=f"{selected_indicator} Projection for Next 10 Years")
            fig.add_scatter(x=future_dates, y=future_pred.flatten(), mode='lines', name='Future Projection', line=dict(dash='dash'))
            st.plotly_chart(fig)
        except Exception as e:
            st.error(f"Error generating projection: {e}")
    else:
        st.warning("Please download the climate data first.")

# Section 4: Province-Specific Analysis and Insights
st.header("4. Province-Specific Analysis and Insights")

st.markdown("""
Get customized insights on biodiversity and climate data at the provincial level. Choose a province to explore ecological and climate patterns,
along with customized interpretations to support regional conservation and policy initiatives.
""")

# Additional placeholder for province-specific analyses
selected_province = st.selectbox("Select Province for Analysis", ["Ontario", "Quebec", "British Columbia", "Alberta"])

st.write(f"Additional analysis and insights for {selected_province} will be available in future updates.")