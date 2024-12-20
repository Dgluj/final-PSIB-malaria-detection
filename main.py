import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib

from src.carga_imagenes import cargar_imagenes
from src.preprocesamiento import reducir_ruido, separar_canales, seleccionar_canal_mayor_contraste, binarizar_con_kmeans, aplicar_filtro_mediana, aplicar_operaciones_morfologicas, rellenar_celulas 
from src.segmentacion import aplicar_watershed
from src.extraccion_de_caracteristicas import construir_base_datos, clasificacion_final
from src.entrenamiento_modelos import dividir_datos, evaluar_modelos, mostrar_matrices_confusion
from src.seleccion_modelo import mostrar_classification_reports, comparar_modelos, graficar_curvas_roc, seleccionar_mejor_modelo

def main():
    imagenes = cargar_imagenes()
    print(f"Se cargaron {len(imagenes)} imágenes:")
    for nombre in imagenes.keys():
        print(f"- {nombre}")
    
    # Inicializo el DataFrame final
    df_final = pd.DataFrame()

    for nombre, img in imagenes.items(): 
        img_filtered, img_rgb = reducir_ruido(img) # Reducir el ruido y convertir la imagen a RGB
        canal_rojo, canal_verde, canal_azul = separar_canales(img_rgb) # Separar los canales de la imagen filtrada
        canal_seleccionado = seleccionar_canal_mayor_contraste(canal_rojo, canal_verde, canal_azul)

        img_binarizada = binarizar_con_kmeans(canal_seleccionado) # Binarizar la imagen del canal de mayor contraste usando KMeans
        img_mediana = aplicar_filtro_mediana(img_binarizada) # Aplicar filtro de mediana para suavizar la imagen binarizada
        img_morfo = aplicar_operaciones_morfologicas(img_mediana) # Aplicar operaciones morfológicas (dilatación y erosión)
        img_rellena = rellenar_celulas(img_morfo) # Rellenar las células

        # Aplicar la transformada de la distancia y Watershed a la imagen completa binaria (img_binarizada)
        img_ws, resultados_intermedios = aplicar_watershed(img_rellena, level=40) # No achicar mas xq se caga

        # Detectar contornos
        contornos, _ = cv2.findContours(img_ws, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Construir base de datos completa para cada imagen
        df = construir_base_datos(canal_seleccionado, contornos, 4500)

        df["Infectada"] = (
            (df["Área"] > 9940) & (df["Área"] < 17300) & 
            (df["Contraste"] > 20) & (df["Contraste"] < 73) & 
            (df["Homogeneidad"] < 0.527) & 
            (df["Entropía"] > 9.22) & 
            (df["Energía"] < 0.27) &
            (df["Correlación Haralick"] > 0.995) &
            (df["Circularidad"] > 0.735)
        ).astype(int)  # Convertir a 0 o 1
        
        # Agregar la columna "Imagen" con el nombre de la imagen
        df["Imagen"] = nombre

        # Reorganizar las columnas para que "Imagen" sea la primera
        columnas = ["Imagen"] + [col for col in df.columns if col != "Imagen"]
        df = df[columnas]

        # Para que se visualicen todas las columnas de la Tabla
        pd.set_option('display.max_columns', None)

        df_desbalanceado = clasificacion_final(df)

        # Concatenar el DataFrame de infectadas y sanas al DataFrame final
        df_final = pd.concat([df_final, df_desbalanceado], ignore_index=True)
    
    # Imprimir el DataFrame final
    print("DataFrame Final Desbalanceado:")
    print(df_final)
    
    # ENTRENAMIENTO
    # Mantén solo las características relevantes
    X = df_final.drop(columns=["Imagen","ID","Infectada"]).copy() # No se si sacar las coordenadas o dejarlas para algun bb final
    y = df_final["Infectada"]

    type(X) # <class 'pandas.core.frame.DataFrame'>
    type(y) # <class 'pandas.core.series.Series'>

    # Opcional: convierte a numpy array si no necesitas trabajar con DataFrame
    X = X.values  # Solo los valores, el modelo no necesita el encabezado ni índices
    y = y.values  # Convierte la columna objetivo
    
    # División, entrenamiento y evaluación
    modelos, resultados, X_test, y_test = evaluar_modelos(X, y) 
    """
    El type de modelos es <class 'dict'>
    Los modelos de evaluar_modelos son: {'RandomForest': RandomForestClassifier(max_depth=3, random_state=0), 'SVM': SVC(random_state=0)}     
    El type de resultados es <class 'dict'>
    Los resultados de evaluar_modelos son: {'RandomForest':     Reales  Predichas..... 'SVM':     Reales  Predichas.....}
    """
    print("El type de modelos es", type(modelos))
    print("Los modelos de evaluar_modelos son:", modelos)

    print("El type de resultados es", type(resultados))
    print("Los resultados de evaluar_modelos son:",resultados)

    # Mostrar matrices de confusión
    mostrar_matrices_confusion(modelos, X_test, y_test, ["No infectada", "Infectada"])      
    # Mostrar los classification report
    mostrar_classification_reports(modelos, X_test, y_test, ["No infectada", "Infectada"])

    # SELECCIÓN DEL MODELO
    # Comparar modelos
    resultados_comparacion = comparar_modelos(modelos, X, y, cv=5)
    
    # Convertir los resultados de comparación a DataFrame y mostrarlos de forma más visual
    df_comparacion = pd.DataFrame(resultados_comparacion).T
    print("\nResultados de comparación de modelos:")
    # print(df_comparacion)

    # Graficar curvas ROC
    graficar_curvas_roc(modelos, X_test, y_test)

    # Seleccionar el mejor modelo según el accuracy (o cualquier otra métrica relevante)
    nombre_mejor_modelo = seleccionar_mejor_modelo(resultados_comparacion)
    
    # Acceder directamente al modelo desde el diccionario 'modelos'
    mejor_modelo = modelos[nombre_mejor_modelo]

    print(mejor_modelo)
    print("El type del mejor modelo es: ", type(mejor_modelo))
    
    # Guardar el mejor modelo usando joblib
    joblib.dump(mejor_modelo, "mejor_modelo.pkl") #El type del mejor modelo es <class 'sklearn.ensemble._forest.RandomForestClassifier'>

if __name__ == "__main__":
    main()