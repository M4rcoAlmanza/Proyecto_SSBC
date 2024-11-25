import pandas as pd 
import numpy as np 
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score, classification_report
from metricas import mostrar_metricas

warnings.filterwarnings("ignore")

# Cargar datos
df = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')

# Preprocesamiento
df = df.drop_duplicates()
columnas_categoricas = ['Gender', 'SMOKE', 'family_history_with_overweight', 'CALC', 'FAVC', 'SCC', 'CAEC', 'MTRANS', 'NObeyesdad']
lbl = LabelEncoder()
for col in columnas_categoricas:
    df[col] = lbl.fit_transform(df[col])

# Variables independientes y dependientes
y = df['NObeyesdad']
X = df.drop(['NObeyesdad'], axis=1)

# Estandarizar datos
scaler = StandardScaler()
X = scaler.fit_transform(X)

# División de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)

# Clasificador Random Forest
rf = RandomForestClassifier(n_estimators=400, criterion='entropy', max_depth=8, min_samples_split=30, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
mostrar_metricas("Random Forest", y_test, y_pred_rf)

# Clasificador SVM
# svc = SVC(kernel='linear', C=3, random_state=42)
# svc.fit(X_train, y_train)
# y_pred_svc = svc.predict(X_test)
# mostrar_metricas("SVM", y_test, y_pred_svc)


# ------------------------------------------------------------------------------------------------
# Clasificador KNN
# ------------------------------------------------------------------------------------------------
knn = KNeighborsClassifier(n_neighbors=20)
knn.fit(X_train, y_train)  # Entrenar el modelo KNN
y_pred_knn = knn.predict(X_test)  # Realizar predicciones
mostrar_metricas("KNN", y_test, y_pred_knn)  # Evaluar el modelo

# ------------------------------------------------------------------------------------------------
# Clasificador Gradient Boosting (GBC)
# ------------------------------------------------------------------------------------------------
gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=0.2, max_depth=3, min_samples_split=10, min_samples_leaf=5, random_state=42)
gbc.fit(X_train, y_train)  # Entrenar el modelo GBC
y_pred_gbc = gbc.predict(X_test)  # Realizar predicciones
mostrar_metricas("Gradient Boosting Classifier (GBC)", y_test, y_pred_gbc)  # Evaluar el modelo

# ------------------------------------------------------------------------------------------------
# Clasificador XGBoost (XGB)
# ------------------------------------------------------------------------------------------------
xgb = XGBClassifier(max_depth=4, learning_rate=0.1, n_estimators=50, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=0.1, random_state=42)
xgb.fit(X_train, y_train)  # Entrenar el modelo XGB
y_pred_xgb = xgb.predict(X_test)  # Realizar predicciones
mostrar_metricas("XGBoost Classifier (XGB)", y_test, y_pred_xgb)  # Evaluar el modelo

votacion = VotingClassifier([('KNN', knn), ('GBC', gbc), ('XGB', xgb)])
votacion.fit(X_train, y_train)
y_pred_votacion = votacion.predict(X_test)
mostrar_metricas("Clasificador de Votación (KNN,GBC,XGB)", y_test, y_pred_votacion)
