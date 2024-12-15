import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.model_selection import cross_validate
from sklearn import metrics
import pandas as pd

# Función para mostrar los classification reports
def mostrar_classification_reports(modelos, X_test, y_test, class_names):
    for nombre, clf in modelos.items():
        predicciones = clf.predict(X_test)
        print(f'\n{nombre.upper()} REPORT')
        print(classification_report(y_test, predicciones, target_names=class_names))

# Función para realizar comparación de modelos con cross_validate
def comparar_modelos(modelos, X, y, cv=5):
    resultados = {}
    for nombre, clf in modelos.items():
        print(f"Evaluando {nombre} con cross_validate...")
        scores = cross_validate(clf, X, y, cv=cv, scoring="accuracy", return_train_score=True)
        # Calcula el promedio (mean) y desviación estándar (std) de los scores de prueba y entrenamiento.
        resultados[nombre] = {
            "mean_test_score": np.mean(scores['test_score']),
            "std_test_score": np.std(scores['test_score']),
            "mean_train_score": np.mean(scores['train_score']),
            "std_train_score": np.std(scores['train_score']),
        }
        print(f"{nombre}: Accuracy = {resultados[nombre]['mean_test_score']:.2f} (+/- {resultados[nombre]['std_test_score']*2:.2f})")
    return resultados

# Función para graficar la curva ROC y calcular el AUC
def graficar_curvas_roc(modelos, X_test, y_test):
    plt.figure(figsize=(8, 6))
    for nombre, clf in modelos.items():
        metrics.RocCurveDisplay.from_estimator(clf, X_test, y_test, ax=plt.gca(), name=f"{nombre}") # Reutilización de un mismo ax: Al pasar ax=plt.gca() en cada iteración, todas las curvas ROC se dibujan en el mismo gráfico.
    plt.title("Curvas ROC")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()
    # for nombre, clf in modelos.items():
    #     metrics.RocCurveDisplay.from_estimator(clf, X_test, y_test)
    #     plt.title(f"Curva ROC para {nombre}")
    #     plt.show()

# Función para seleccionar el mejor modelo según el accuracy
def seleccionar_mejor_modelo(resultados_comparacion):
    # Convertir el diccionario de resultados a un DataFrame para facilitar el manejo y ordenamiento
    df_resultados = pd.DataFrame(resultados_comparacion).T  # Transponer para que los modelos sean las filas
    df_resultados = df_resultados.sort_values(by="mean_test_score", ascending=False)  # Ordenamos por el accuracy
    
    # Imprimir la tabla ordenada
    print("Comparación de Modelos (Ordenados por Accuracy):")
    print(df_resultados)
    
    # Seleccionar el mejor modelo (el primero después de ordenar)
    mejor_modelo = df_resultados.index[0]
    print(f"\nEl mejor modelo según el accuracy es: {mejor_modelo}")
    
    # Devolver el mejor modelo
    return mejor_modelo