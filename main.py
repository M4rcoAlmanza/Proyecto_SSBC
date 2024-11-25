import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Mapeo
nobeyesdad_mapeo = {
    0: 'Insufficient_Weight',
    1: 'Normal_Weight',
    2: 'Obesity_Type_I',
    3: 'Obesity_Type_II',
    4: 'Obesity_Type_III',
    5: 'Overweight_Level_I',
    6: 'Overweight_Level_II'
}

# Carga el modelo
modelo_svc = joblib.load('Modelos/modelo.pkl')
scaler = joblib.load('Modelos/scaler.pkl')

st.title("Obesity Level")
st.write("Fill out the form below.")

# Valores categoricos
categorical_columns = ['Gender', 'SMOKE', 'family_history_with_overweight', 'CALC', 'FAVC', 'SCC', 'CAEC', 'MTRANS']
categorical_values = {
    'Gender': ['Male', 'Female'],
    'SMOKE': ['No', 'Yes'],
    'family_history_with_overweight': ['No', 'Yes'],
    'CALC': ['No', 'Sometimes', 'Frequently', 'Always'],
    'FAVC': ['No', 'Yes'],
    'SCC': ['No', 'Yes'],
    'CAEC': ['No', 'Sometimes', 'Frequently', 'Always'],
    'MTRANS': ['Car', 'Public Transportation', 'Motorbike', 'Bike', 'Walking'],
}

# Encoders
encoders = {col: LabelEncoder().fit(values) for col, values in categorical_values.items()}
# Formulario
with st.form("Form"):
    age = st.number_input("What is your age?", min_value=0, max_value=120, step=1)
    gender = st.selectbox("What is your gender?", categorical_values['Gender'])
    height = st.number_input("What is your height? (meters)", min_value=0.5, max_value=2.5, step=0.01)
    weight = st.number_input("What is your weight? (kilograms)", min_value=10.0, max_value=200.0, step=0.1)
    calc = st.radio("How often do you consume alcohol?", categorical_values['CALC'])
    favc = st.radio("Do you frequently eat high-calorie foods?", categorical_values['FAVC'])
    fcvc = st.slider("How often do you eat vegetables (1-3)?", min_value=1, max_value=3, step=1)
    ncp = st.slider("How many main meals do you have per day?", min_value=1, max_value=4, step=1)
    scc = st.radio("Do you monitor your daily calorie intake?", categorical_values['SCC'])
    smoke = st.radio("Do you smoke?", categorical_values['SMOKE'])
    ch2o = st.slider("How much water do you drink daily? (1-3)", min_value=1, max_value=3, step=1)
    family_history = st.radio("Do you have a family history of obesity?", categorical_values['family_history_with_overweight'])
    faf = st.slider("How often do you have physical activity?", min_value=0.0, max_value=3.0, step=1.0)
    tue = st.slider("How many hours per day do you use technological devices?", min_value=0, max_value=3, step=1)
    caec = st.radio("Do you consume food between meals?", categorical_values['CAEC'])
    mtrans = st.selectbox("What is your primary mode of transportation?", categorical_values['MTRANS'])
    
    submit = st.form_submit_button("See results")

if submit:
    # Categoricos a númericos
    data = {
        'Gender': encoders['Gender'].transform([gender])[0],
        'Age': age,
        'Height': height,
        'Weight': weight,
        'family_history_with_overweight': encoders['family_history_with_overweight'].transform([family_history])[0],
        'FAVC': encoders['FAVC'].transform([favc])[0],
        'FCVC': fcvc,
        'NCP': ncp,
        'CAEC': encoders['CAEC'].transform([caec])[0],
        'SMOKE': encoders['SMOKE'].transform([smoke])[0],
        'CH2O': ch2o,
        'SCC': encoders['SCC'].transform([scc])[0],
        'FAF': faf,
        'TUE': tue,
        'CALC': encoders['CALC'].transform([calc])[0],
        'MTRANS': encoders['MTRANS'].transform([mtrans])[0],
    }

    # Estandarización
    X_new = pd.DataFrame([data]).values
    X_new = scaler.transform(X_new)

    # Predicción
    result = modelo_svc.predict(X_new)
    st.success(f"**Result:** {nobeyesdad_mapeo[result[0]]}")
