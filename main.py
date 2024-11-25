import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# CSS personalizado para estilizar el formulario
st.markdown("""
    <style>
    body {
        background-color: #f5f5f5;
        font-family: Arial, sans-serif;
    }
    .main-title {
        text-align: center;
        color: #2c3e50;
        font-size: 40px;
        font-weight: bold;
    }
    .form-section {
        background: #ffffff;
        padding: 20px;
        margin: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .form-header {
        font-size: 24px;
        color: #34495e;
        margin-bottom: 20px;
    }
    .stButton > button {
        background-color: #2c3e50;
        color: white;
        font-size: 18px;
        padding: 10px 20px;
        border-radius: 5px;
    }
    .stButton > button:hover {
        background-color: #34495e;
    }
    </style>
""", unsafe_allow_html=True)

# Título principal
st.markdown("<div class='main-title'>Nivel de Obesidad</div>", unsafe_allow_html=True)
st.write("Por favor, rellena el formulario con la información solicitada.")

# Modelo y mapeo
nobeyesdad_mapeo = {
    0: 'Peso bajo',
    1: 'Peso normal',
    2: 'Obesidad tipo I',
    3: 'Obesidad tipo II',
    4: 'Obesidad tipo III',
    5: 'Sobrepeso Nivel I',
    6: 'Sobrepeso Nivel II'
}

modelo_svc = joblib.load('Modelos/modelo.pkl')
scaler = joblib.load('Modelos/scaler.pkl')

# Valores categóricos en español e inglés
categorical_values = {
    'Gender': {'options_es': ['Hombre', 'Mujer'], 'values_en': ['Male', 'Female']},
    'SMOKE': {'options_es': ['No', 'Sí'], 'values_en': ['No', 'Yes']},
    'family_history_with_overweight': {'options_es': ['No', 'Sí'], 'values_en': ['No', 'Yes']},
    'CALC': {'options_es': ['No', 'A veces', 'Frecuentemente', 'Siempre'], 'values_en': ['No', 'Sometimes', 'Frequently', 'Always']},
    'FAVC': {'options_es': ['No', 'Sí'], 'values_en': ['No', 'Yes']},
    'SCC': {'options_es': ['No', 'Sí'], 'values_en': ['No', 'Yes']},
    'CAEC': {'options_es': ['No', 'A veces', 'Frecuentemente', 'Siempre'], 'values_en': ['No', 'Sometimes', 'Frequently', 'Always']},
    'MTRANS': {'options_es': ['Carro', 'Transporte público', 'Motocicleta', 'Bicicleta', 'Caminando'],
               'values_en': ['Car', 'Public Transportation', 'Motorbike', 'Bike', 'Walking']},
}

encoders = {col: LabelEncoder().fit(values['values_en']) for col, values in categorical_values.items()}

# Formulario
with st.form("Form", clear_on_submit=True):
    st.markdown("<div class='form-header'>Información general</div>", unsafe_allow_html=True)
    age = st.number_input("¿Cuál es tu edad?", min_value=0, max_value=120, step=1)
    gender = st.selectbox("¿Cuál es tu género?", categorical_values['Gender']['options_es'])
    height = st.number_input("¿Cuál es tu altura? (metros)", min_value=0.5, max_value=2.5, step=0.01)
    weight = st.number_input("¿Cuál es tu peso? (kilogramos)", min_value=10.0, max_value=200.0, step=0.1)

    st.markdown("<div class='form-header'>Hábitos y Estilo de Vida</div>", unsafe_allow_html=True)
    calc = st.radio("¿Con qué frecuencia consumes alcohol?", categorical_values['CALC']['options_es'])
    favc = st.radio("¿Consumes frecuentemente alimentos calóricos?", categorical_values['FAVC']['options_es'])
    fcvc = st.slider("¿Qué tan seguido consumes vegetales? (0 = Nunca, 1 = Pocas veces, 2 = Frecuentemente, 3 = Siempre)", min_value=1, max_value=3, step=1)
    ncp = st.slider("¿Cuántas comidas principales tienes al día?", min_value=1, max_value=4, step=1)
    scc = st.radio("¿Controlas tu ingesta diaria de calorías?", categorical_values['SCC']['options_es'])
    smoke = st.radio("¿Fumas?", categorical_values['SMOKE']['options_es'])
    ch2o = st.slider("¿Cuánta agua consumes diariamente? (1 = Pocas veces, 2 = Frecuentemente, 3 = Siempre)", min_value=1, max_value=3, step=1)
    family_history = st.radio("¿Tienes antecedentes familiares de obesidad?", categorical_values['family_history_with_overweight']['options_es'])

    st.markdown("<div class='form-header'>Actividad Física y Transporte</div>", unsafe_allow_html=True)
    faf = st.slider("¿Con qué frecuencia realizas actividad física? (0 = Nunca, 1 = Pocas veces, 2 = Frecuentemente, 3 = Siempre)", min_value=0, max_value=3, step=1)
    tue = st.slider("¿Qué tan seguido utilizas dispositivos tecnológicos? (0 = Nunca, 1 = Pocas veces, 2 = Frecuentemente, 3 = Siempre)", min_value=0, max_value=3, step=1)
    caec = st.radio("¿Consumes alimentos entre comidas?", categorical_values['CAEC']['options_es'])
    mtrans = st.selectbox("¿Cuál es tu medio de transporte principal?", categorical_values['MTRANS']['options_es'])

    submit = st.form_submit_button("Ver resultados")

if submit:
    # Conversión de valores y procesamiento
    data = {
        'Gender': encoders['Gender'].transform([categorical_values['Gender']['values_en'][categorical_values['Gender']['options_es'].index(gender)]])[0],
        'Age': age,
        'Height': height,
        'Weight': weight,
        'family_history_with_overweight': encoders['family_history_with_overweight'].transform(
            [categorical_values['family_history_with_overweight']['values_en'][categorical_values['family_history_with_overweight']['options_es'].index(family_history)]])[0],
        'FAVC': encoders['FAVC'].transform([categorical_values['FAVC']['values_en'][categorical_values['FAVC']['options_es'].index(favc)]])[0],
        'FCVC': fcvc,
        'NCP': ncp,
        'CAEC': encoders['CAEC'].transform([categorical_values['CAEC']['values_en'][categorical_values['CAEC']['options_es'].index(caec)]])[0],
        'SMOKE': encoders['SMOKE'].transform([categorical_values['SMOKE']['values_en'][categorical_values['SMOKE']['options_es'].index(smoke)]])[0],
        'CH2O': ch2o,
        'SCC': encoders['SCC'].transform([categorical_values['SCC']['values_en'][categorical_values['SCC']['options_es'].index(scc)]])[0],
        'FAF': faf,
        'TUE': tue,
        'CALC': encoders['CALC'].transform([categorical_values['CALC']['values_en'][categorical_values['CALC']['options_es'].index(calc)]])[0],
        'MTRANS': encoders['MTRANS'].transform([categorical_values['MTRANS']['values_en'][categorical_values['MTRANS']['options_es'].index(mtrans)]])[0],
    }

    X_new = pd.DataFrame([data]).values
    X_new = scaler.transform(X_new)

    # Predicción
    result = modelo_svc.predict(X_new)
    st.success(f"**Resultado:** {nobeyesdad_mapeo[result[0]]}")
