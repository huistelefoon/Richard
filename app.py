
import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Verbruik Voorspeller", layout="centered")
st.title("🔌 Verbruik Voorspeller")
st.write("Deze app traint het model en voorspelt het energieverbruik op basis van invoer.")

# === TRAINING ===
st.subheader("📊 Stap 1: Data inladen en model trainen")

try:
    df = pd.read_excel("dataset.xlsx")
    st.success("✅ Dataset succesvol geladen")
    st.write(df.head())

    # Encode categorische kolommen
    label_encoders = {}
    for col in df.select_dtypes(include=["object", "category"]):
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # Features en target
    target = 'Verbruik_kWh'
    X = df.drop(columns=[target])
    y = df[target]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model trainen
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    st.success("✅ Model getraind")

except Exception as e:
    st.error(f"❌ Er ging iets mis tijdens het inladen of trainen: {e}")
    st.stop()

# === INVOER EN VOORSPELLING ===
st.subheader("🔮 Stap 2: Doe een voorspelling")

# Dynamisch invoervelden maken
user_input = {}
for col in X.columns:
    if df[col].dtype == 'int64' or df[col].dtype == 'float64':
        waarde = st.number_input(f"{col}", value=float(df[col].mean()))
    else:
        opties = df[col].unique()
        labels = [k for k, v in label_encoders[col].classes_.items()] if col in label_encoders else opties
        waarde = st.selectbox(f"{col}", opties)
    user_input[col] = waarde

# Data voorbereiden voor voorspelling
input_df = pd.DataFrame([user_input])

# Eventueel encoderen
for col in input_df.columns:
    if col in label_encoders:
        le = label_encoders[col]
        try:
            input_df[col] = le.transform(input_df[col].astype(str))
        except:
            st.error(f"Onbekende waarde voor kolom {col}. Gelieve een geldige waarde te kiezen.")
            st.stop()

if st.button("Voorspel Verbruik (kWh)"):
    try:
        prediction = model.predict(input_df)
        st.success(f"⚡️ Voorspeld verbruik: {prediction[0]:.2f} kWh")
    except Exception as e:
        st.error(f"❌ Fout bij voorspelling: {e}")
