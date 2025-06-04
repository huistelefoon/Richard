
import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Titel
st.title("Voorspelmodel Verbruik kWh – REZ Dakveiligheid")

# Dataset inladen
@st.cache_data
def load_data():
    df = pd.read_excel("Dataset REZ Streamlit versie 2.xlsx")

    # Drop overbodige kolommen
    df = df.drop(columns=['Betaalde prijs kWh', 'Energiekosten', 'Marktprijs kWh'], errors='ignore')

    return df

df = load_data()

# Features en target
X = df.drop(columns=['Verbruik_kWh'])
y = df['Verbruik_kWh']

# Model trainen
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# User input
st.subheader("Voer waarden in om verbruik te voorspellen")

user_input = {}
for col in X.columns:
    if df[col].dtype in ['int64', 'float64']:
        user_input[col] = st.number_input(f"{col}", value=float(df[col].mean()))
    elif df[col].dtype == 'object':
        user_input[col] = st.selectbox(f"{col}", options=df[col].unique())
    else:
        user_input[col] = st.text_input(f"{col}")

# Voorspellen
if st.button("Voorspel Verbruik (kWh)"):
    try:
        input_df = pd.DataFrame([user_input])
        prediction = model.predict(input_df)[0]
        st.success(f"Voorspeld verbruik: {prediction:.2f} kWh")
    except Exception as e:
        st.error(f"Er ging iets mis: {e}")
