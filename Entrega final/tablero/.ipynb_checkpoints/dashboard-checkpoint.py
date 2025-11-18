import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import numpy as np

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
API_URL = "http://3.214.199.14:8001/api/v1/predict"  #

st.set_page_config(page_title="Tablero - house pricing", layout="wide")

# ---------------------------------------------------
# Load cached map data
# ---------------------------------------------------
@st.cache_data
def load_map_data():
    return pd.read_parquet("precio_ciudad_map.parquet")

df = load_map_data()

# ---------------------------------------------------
# LAYOUT
# ---------------------------------------------------
left_col, right_col = st.columns([2, 1])

# ================= LEFT COLUMN — MAP =================
with left_col:
    st.title("🏙️ Precio promedio por sqft por ciudad")

    fig = px.scatter_mapbox(
        df,
        lat='lat',
        lon='lon',
        size='price_per_sqft',
        color='price_per_sqft',
        hover_name='city',
        color_continuous_scale='Viridis',
        mapbox_style='carto-positron',
        zoom=7,
        title='Precio promedio por sqft por ciudad',
        width=900,
        height=700,
    )

    st.plotly_chart(fig, use_container_width=True)

# ================= RIGHT COLUMN — PREDICTION FORM =================
with right_col:
    st.header("🔮 Predict House Price")

    bedrooms = st.number_input("Bedrooms", 1, 10, 3)
    bathrooms = st.number_input("Bathrooms", 1.0, 5.0, 2.0)
    sqft_living = st.number_input("Living area (sqft)", 300, 15000, 1800)
    sqft_lot = st.number_input("Lot size (sqft)", 400, 50000, 4000)
    floors = st.number_input("Floors", 1.0, 4.0, 1.0)

    waterfront = st.selectbox("Waterfront", [0, 1])
    view = st.selectbox("View", [0, 1, 2, 3, 4])
    condition = st.selectbox("Condition", [1, 2, 3, 4, 5])
    sqft_above = st.number_input("Sqft above", 300, 15000, 1500)
    sqft_basement = st.number_input("Sqft basement", 0, 10000, 300)
    yr_built = st.number_input("Year built", 1900, 2025, 1990)
    yr_renovated = st.selectbox("Renovated?", [False, True])

    city = st.selectbox("City", sorted(df["city"].unique()))

    predict_button = st.button("🚀 Predict Price")

    if predict_button:

        # Normalize names
        def normalize_city_name(name):
            return "city_" + name.replace(" ", "_")

        all_cities = df["city"].unique()
        all_city_columns = [normalize_city_name(c) for c in all_cities]

        selected_city_col = normalize_city_name(city)

        city_one_hot = {
            col: (col == selected_city_col) for col in all_city_columns
        }

        # Base numeric payload
        payload = {
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "sqft_living": sqft_living,
            "sqft_lot": sqft_lot,
            "floors": floors,
            "waterfront": waterfront,
            "view": view,
            "condition": condition,
            "sqft_above": sqft_above,
            "sqft_basement": sqft_basement,
            "yr_built": yr_built,
            "yr_renovated": yr_renovated,
        }

        # Add city one-hot columns
        payload.update(city_one_hot)

        st.write("Consultando modelo...")

        try:
            response = requests.post(API_URL, json={"inputs": [payload]})
            result = response.json()
            pred = np.exp(result["predictions"][0])
            st.success(f"🏡 **Predicted Price: USD {pred:,.2f}**")

        except Exception as e:
            st.error(f"Error contacting API: {e}")
