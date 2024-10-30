# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
from pygbif import occurrences as occ
import wbdata
import datetime
import plotly.express as px
from sklearn.linear_model import LinearRegression

# App title and description
st.title("GBIF and World Bank Climate Data Analysis")
st.write("Explore biodiversity data from GBIF and climate data from World Bank for Canada.")

# Section 1: Download and Display GBIF Data for Canada
st.header("1. Download GBIF Data")
species = st.text_input("Enter species name (e.g., 'Ursus arctos' for brown bear):")

if st.button("Download GBIF Data"):
    # Fetch occurrences from GBIF for Canada
    try:
        occurrences = occ.search(scientificName=species, country="CA", limit=100)
        df_gbif = pd.json_normalize(occurrences['results'])[['species', 'decimalLatitude', 'decimalLongitude', 'eventDate']]
        
        # Rename columns for Streamlit map compatibility
        df_gbif = df_gbif.rename(columns={'decimalLatitude': 'latitude', 'decimalLongitude': 'longitude'})
        
        st.write(f"GBIF Data for {species} in Canada")
        st.dataframe(df_gbif)
        
        # Plot the map
        st.map(df_gbif[['latitude', 'longitude']].dropna())
    except Exception as e:
        st.error(f"Error retrieving data: {e}")

# Section 2: World Bank Climate Data Download and Display
st.header("2. Download World Bank Climate Data")

# Define the indicators we want to use
indicators = {
    #"EN.ATM.CO2E.PC": "CO2 emissions (metric tons per capita)", 
    "AG.LND.FRST.ZS": "Forest area (% of land area)", 
    "NY.GDP.MKTP.CD": "GDP (current US$)"
}
start_year = st.slider("Select start year", 1960, 2020, 1990)

# Check for valid indicators by testing each individually
valid_indicators = {}
for code, description in indicators.items():
    try:
        # Try fetching data for each indicator separately to confirm availability
        test_data = wbdata.get_dataframe({code: description}, country="CAN")
        if not test_data.empty:
            valid_indicators[code] = description
        else:
            st.warning(f"Indicator '{description}' ({code}) might be unavailable or empty.")
    except Exception:
        st.warning(f"Indicator '{description}' ({code}) might be unavailable or has been removed.")

if st.button("Download World Bank Climate Data"):
    try:
        # Check if we have valid indicators before proceeding
        if not valid_indicators:
            st.error("No valid indicators are available for the selected time range.")
        else:
            # Fetch data using only valid indicators
            wb_data = wbdata.get_dataframe(valid_indicators, country="CAN")
            if not wb_data.empty:
                wb_data.reset_index(inplace=True)
                
                # Convert 'date' column to datetime format
                wb_data['date'] = pd.to_datetime(wb_data['date'], errors='coerce')
                
                # Filter data by selected start year
                wb_data = wb_data[wb_data['date'].dt.year >= start_year]
                
                st.write("World Bank Climate Indicators for Canada")
                st.dataframe(wb_data)
                
                # Plot the data
                fig = px.line(wb_data, x='date', y=wb_data.columns[1:], title="Climate Indicators Over Time")
                st.plotly_chart(fig)
            else:
                st.warning("No data returned for the selected indicators and country.")
                
    except Exception as e:
        st.error(f"An error occurred while retrieving data: {e}")

# Section 3: Climate Change Impact Scenarios
st.header("3. Project Future Scenarios")
indicator = st.selectbox("Select climate indicator for projection:", list(indicators.values()))

# Simple linear regression model for future projections
if st.button("Generate Climate Impact Scenario"):
    if not wb_data.empty:
        try:
            # Prepare data for modeling
            data = wb_data[[indicator]].dropna().reset_index()
            X = data.index.values.reshape(-1, 1)
            y = data[indicator].values.reshape(-1, 1)

            # Train linear regression model
            model = LinearRegression()
            model.fit(X, y)

            # Predict future values (next 10 years)
            future_years = np.arange(len(X), len(X) + 10).reshape(-1, 1)
            future_pred = model.predict(future_years)

            # Combine past data and future projections for plotting
            future_dates = pd.date_range(start=str(end_year), periods=10, freq='Y')
            future_df = pd.DataFrame(data={indicator: future_pred.flatten()}, index=future_dates)
            combined_df = pd.concat([data.set_index('date'), future_df])

            # Plot past data and projections
            fig = px.line(combined_df, y=indicator, title=f"Projection of {indicator} Over the Next 10 Years")
            fig.add_scatter(x=future_df.index, y=future_pred.flatten(), mode='lines', name='Future Projection')
            st.plotly_chart(fig)
        except Exception as e:
            st.error(f"Error generating projection: {e}")
    else:
        st.warning("Please download World Bank data first.")