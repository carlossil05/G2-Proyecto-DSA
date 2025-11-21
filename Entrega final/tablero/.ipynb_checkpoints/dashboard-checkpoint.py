import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import numpy as np

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
API_URL = "http://3.214.199.14:8001/api/v1/predict"  #

st.set_page_config(page_title="Tablero - Precio de la vivienda en WA, USA", layout="wide")

# ---------------------------------------------------
# Load cached map data
# ---------------------------------------------------
@st.cache_data
def load_map_data():
    return pd.read_parquet("precio_ciudad_map.parquet")

df = load_map_data()

# ================= MAPA =================
st.title("🏙️ Mapa del precio promedio por sqft")
st.markdown("En este mapa se puede ver cuáles son las ciudades con los precios de vivienda promedio por pies cuadrados más y menos costosos.")

fig = px.scatter_mapbox(
    df,
    lat='lat',
    lon='lon',
    size='price_per_sqft',
    color='price_per_sqft',
    hover_name='city',
    color_continuous_scale='Viridis',
    mapbox_style='carto-positron',
    zoom=10,
    title='Precio promedio por sqft por ciudad',
    width=900,
    height=900,
)

st.plotly_chart(fig, use_container_width=True)

# =================INGRESO DE INFORMACIÓN =================

st.header("🔮 Predicción del precio de la vivienda")
st.markdown("Ingrese los datos de la vivienda y presione **Predecir Precio**.")

col1, col2, col3 = st.columns(3)

with col1:
    
    city = st.selectbox("City", sorted(df["city"].unique()), help="Seleccione la ciudad en que está ubicada la vivienda")
    yr_built = st.number_input("Year built", 1900, 2025, 1990, help="Ingrese el año de construcción")
    yr_renovated = st.selectbox("Renovated?", [False, True], help="Indique si la vivienda ha sido remodelada")

with col2:

    bedrooms = st.number_input("Bedrooms", 1, 10, 3, help="Ingrese la cantidad de habitaciones")
    bathrooms = st.number_input("Bathrooms", 1.0, 5.0, 2.0, help="Ingrese la cantidad de baños")
    floors = st.number_input("Floors", 1.0, 4.0, 1.0, help="Ingrese la cantidad de pisos")
    waterfront = st.selectbox("Waterfront", [0, 1], help="Seleccione'1' si la vivienda está en el borde del agua o '0' de lo contrario")
    view = st.selectbox("View", [0, 1, 2, 3, 4], help="Seleccione '4' si la vista es muy buena o un numero menor de lo contrario")
    condition = st.selectbox("Condition", [1, 2, 3, 4, 5], help="Seleccione '5' si la condición de la vivienda es perfecta o un número menor de lo contrario")
    predict_button = st.button("🚀 Predecir el precio")

with col3:

    sqft_living = st.number_input("Living area (sqft)", 300, 15000, 1800, help="Ingrese el área de la sala en pies cuadrados")
    sqft_lot = st.number_input("Lot size (sqft)", 400, 50000, 4000, help="Ingrese el área del lote en pies cuadrados")
    sqft_above = st.number_input("Sqft above", 300, 15000, 1500, help="Ingrese el área en pies cuadrados sin contar el sótano")
    sqft_basement = st.number_input("Sqft basement", 0, 10000, 300, help="Ingrese el área en pies cuadrados del sótano")
    
# ================= BOTÓN =================
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
        st.success(f"🏡 **Precio estimado de la vivienda: USD {pred:,.2f}**")

    except Exception as e:
        st.error(f"Error contactando la API: {e}")
