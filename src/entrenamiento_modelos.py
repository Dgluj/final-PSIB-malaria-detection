import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# División de los datos en conjuntos de entrenamiento y prueba
def dividir_datos(X, y, test_size=0.3, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# Función genérica para entrenar un modelo y devolver un DataFrame con Reales y Predichas
def entrenar_modelo(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)  # Entrenamiento del modelo
    predichas = clf.predict(X_test)  # Predicciones
    return pd.DataFrame({"Reales": y_test, "Predichas": predichas})

# Entrenar y evaluar con varios modelos juntos
def evaluar_modelos(X, y):
    # Dividimos los datos
    X_train, X_test, y_train, y_test = dividir_datos(X, y)
    
    # Modelos a entrenar
    modelos = {
        "RandomForestClassifier": RandomForestClassifier(max_depth=3, random_state=0),
        "SVM": SVC(random_state=0)
    }
    
    resultados = {}  # Guardaremos los resultados de cada modelo
    for nombre, clf in modelos.items():
        print(f"Entrenando {nombre}...")
        resultados[nombre] = entrenar_modelo(clf, X_train, X_test, y_train, y_test)
    
    return modelos, resultados, X_test, y_test

# Visualización de las matrices de confusión
def mostrar_matrices_confusion(modelos, X_test, y_test, class_names):
    for nombre, clf in modelos.items():
        predicciones = clf.predict(X_test)
        cm = confusion_matrix(y_test, predicciones)
        disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"Matriz de confusión para {nombre}")
        plt.show()