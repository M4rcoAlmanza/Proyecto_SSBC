from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

def mostrar_metricas(modelo, y_test, y_pred):
    # Calcular
    precision = accuracy_score(y_test, y_pred) * 100
    f1 = f1_score(y_test, y_pred, average='micro') * 100
    recall = recall_score(y_test, y_pred, average='weighted') * 100
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # Crear figura
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [1, 2]})  # Primera columna más pequeña
    
    # Columna 1: Métricas
    ax[0].axis('off')
    text = (
        f"---------------------\n\n"
        f"Modelo: {modelo}\n\n"
        f"---------------------\n\n"
        f"Precisión: {precision:.2f}%\n"
        f"F1-Score: {f1:.2f}%\n"
        f"Recall: {recall:.2f}%\n\n"
        f"Reporte de Clasificación:\n{report}"
    )
    ax[0].text(0.01, 0.99, text, fontsize=11, va='top', ha='left', wrap=True)  # Texto ajustado a la columna

    # Columna 2: Matriz de confusión
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=set(y_test), yticklabels=set(y_test), ax=ax[1])
    ax[1].set_title(f'Matriz de Confusión - {modelo}', fontsize=14, pad=10)
    ax[1].set_xlabel('Resultado', fontsize=12)
    ax[1].set_ylabel('Verdadero', fontsize=12)
    ax[1].tick_params(axis='both', labelsize=10)
    
    plt.tight_layout()
    plt.show()
