import cv2
import numpy as np
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from src.carga_imagenes import cargar_imagenes
from src.preprocesamiento import (
    reducir_ruido, separar_canales, seleccionar_canal_mayor_contraste, 
    binarizar_con_kmeans, aplicar_filtro_mediana, aplicar_operaciones_morfologicas, 
    rellenar_celulas
)
from src.segmentacion import aplicar_watershed
from src.extraccion_de_caracteristicas import construir_base_datos

def obtener_label_para_contorno(nombre_imagen, contorno):
    # Placeholder: aquí se debe implementar la lógica para asignar una etiqueta (0 = sano, 1 = infectado)
    # Podría ser manual o provenir de anotaciones. Por ahora devolvemos aleatorio (no ideal).
    # En un caso real, esto debe reemplazarse por la lógica real.
    return np.random.randint(0, 2)

def generar_dataset_y_entrenar():
    imagenes = cargar_imagenes()
    lista_df = []
    lista_labels = []

    # Procesar cada imagen y extraer características
    for nombre, img in imagenes.items():
        # Preprocesamiento
        img_filtered, img_rgb = reducir_ruido(img)
        canal_rojo, canal_verde, canal_azul = separar_canales(img_rgb)
        canal_seleccionado = seleccionar_canal_mayor_contraste(canal_rojo, canal_verde, canal_azul)
        img_binarizada = binarizar_con_kmeans(canal_seleccionado)
        img_mediana = aplicar_filtro_mediana(img_binarizada)
        img_morfo = aplicar_operaciones_morfologicas(img_mediana)
        img_rellena = rellenar_celulas(img_morfo)

        # Watershed
        img_ws, _ = aplicar_watershed(img_rellena, level=40)

        # Contornos
        contornos, _ = cv2.findContours(img_ws, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Extraer características
        df = construir_base_datos(canal_seleccionado, contornos)

        # Obtener labels
        labels_imagen = []
        for i, cont in enumerate(contornos):
            label = obtener_label_para_contorno(nombre, cont)
            labels_imagen.append(label)
        if len(df) > 0:
            df["Label"] = labels_imagen
            lista_df.append(df)

    if len(lista_df) == 0:
        print("No se generó ningún dataset, revisar el pipeline.")
        return

    dataset = pd.concat(lista_df, ignore_index=True)
    # Guardamos el dataset si se desea
    dataset.to_csv("dataset_celulas.csv", index=False)

    # Entrenar modelos (ejemplo simple)
    X = dataset[["Contraste", "Energía", "Homogeneidad", "Area"]].values
    y = dataset["Label"].values

    modelo1 = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo2 = SVC(probability=True, random_state=42)

    modelo1.fit(X, y)
    modelo2.fit(X, y)

    joblib.dump(modelo1, "modelos/modelo1.pkl")
    joblib.dump(modelo2, "modelos/modelo2.pkl")
    print("Modelos entrenados y guardados exitosamente.")

# División de los datos en conjuntos de entrenamiento y prueba
def dividir_datos(X, y, test_size=0.5, random_state=42):
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
        "RandomForest": RandomForestClassifier(max_depth=3, random_state=0),
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
