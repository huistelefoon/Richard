import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

st.title("Voorspelmodel Verbruik (kWh) - REZ Dakveiligheid")

@st.cache_resource
def train_model():
    df = pd.read_excel("Dataset REZ Streamlit.xlsx")
    df.columns = df.columns.str.strip()

    # Encode categorische kolommen
    for col in df.select_dtypes(include=["object", "category"]):
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    X = df.drop(columns=["Verbruik_kWh"])
    y = df["Verbruik_kWh"]

    model = RandomForestRegressor()
    model.fit(X, y)
    return model, X.columns.tolist()

model, feature_names = train_model()

st.header("Voer gegevens in:")

user_input = {}
for feature in feature_names:
    value = st.number_input(f"{feature}", value=0.0)
    user_input[feature] = value

if st.button("Voorspel verbruik"):
    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)[0]
    st.success(f"Voorspeld Verbruik (kWh): {prediction:.2f}")
