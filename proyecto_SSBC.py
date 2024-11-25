import pandas as pd

from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

import joblib

from metricas import mostrar_metricas

# ------------------------------------------------------------------------------------------------
# CARGAR DATOS
# ------------------------------------------------------------------------------------------------
df = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')

# ------------------------------------------------------------------------------------------------
# LIMPIAR Y TRANSFORMAR
# ------------------------------------------------------------------------------------------------
df = df.drop_duplicates()

columnas_categoricas = ['Gender', 'SMOKE', 'family_history_with_overweight', 'CALC', 'FAVC', 'SCC', 'CAEC', 'MTRANS', 'NObeyesdad']

encoders = {}

for col in columnas_categoricas:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

print("Mapeo de NObeyesdad (valor numérico -> categoría):")
for num, clase in enumerate(encoders['NObeyesdad'].classes_):
    print(f"{num}: {clase}")
    
# ------------------------------------------------------------------------------------------------
# DEFINIR EJES
# ------------------------------------------------------------------------------------------------
y = df['NObeyesdad']  # Clase objetivo
X = df.drop(['NObeyesdad'], axis=1)  # Resto del DF

# ------------------------------------------------------------------------------------------------
# ESTANDARIZAR (UNIFORMIDAD)
# ------------------------------------------------------------------------------------------------
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ------------------------------------------------------------------------------------------------
# SVM CON VALIDACIÓN CRUZADA
# ------------------------------------------------------------------------------------------------
svc = SVC(kernel='linear', C=3, random_state=42)

# Validación cruzada estratificada
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Von validación cruzada
y_clas = cross_val_predict(svc, X, y, cv=cv)
mostrar_metricas("SVM", y, y_clas)

# Entrenar el modelo final con todos los datos
svc.fit(X, y)

# ------------------------------------------------------------------------------------------------
# GUARDAR MODELO Y ESCALADOR
# ------------------------------------------------------------------------------------------------
joblib.dump(svc, 'modelo.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Modelo y escalador guardados exitosamente.")
