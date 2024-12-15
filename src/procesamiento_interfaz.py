import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib

import sys
import os

# Agregar la ruta raíz del proyecto al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocesamiento import reducir_ruido, separar_canales, seleccionar_canal_mayor_contraste, binarizar_con_kmeans, aplicar_filtro_mediana, aplicar_operaciones_morfologicas, rellenar_celulas 
from src.segmentacion import aplicar_watershed, dibujar_bounding_boxes, procesar_recortes_y_watershed, segmentar_recortes
from src.extraccion_de_caracteristicas import construir_base_datos, clasificacion_final
from src.utils import dibujar_bounding_boxes_en_identificadas
from src.entrenamiento_modelos import dividir_datos, evaluar_modelos, mostrar_matrices_confusion
from src.seleccion_modelo import mostrar_classification_reports, comparar_modelos, graficar_curvas_roc, seleccionar_mejor_modelo

# Funciones principales:
# Esta es una función que hará todo el procesamiento de la imagen
def procesar_imagen_con_modelo(self, img):
    if self.mejor_modelo is None:
        print("No se cargó ningún modelo. Por favor, revisa la carga del modelo.")
        return None, None
    
    img_filtered = reducir_ruido(img)  
    img_rgb = cv2.cvtColor(img_filtered, cv2.COLOR_BGR2RGB)
    canal_rojo, canal_verde, canal_azul = separar_canales(img_rgb)  # Función que separa los canales de color
    canal_seleccionado = seleccionar_canal_mayor_contraste(canal_rojo, canal_verde, canal_azul)
    img_binarizada = binarizar_con_kmeans(canal_seleccionado)
    img_mediana = aplicar_filtro_mediana(img_binarizada)
    img_morfo = aplicar_operaciones_morfologicas(img_mediana)
    img_rellena = rellenar_celulas(img_morfo)
    img_ws, resultados_intermedios = aplicar_watershed(img_rellena, level=40)
    contornos, _ = cv2.findContours(img_ws, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_bounding_boxes = dibujar_bounding_boxes(canal_seleccionado, contornos, color=(0, 255, 0), grosor=1, umbral_area_min=5000)
    df = construir_base_datos(canal_seleccionado, contornos)
    X = df.drop(columns=["Imagen","ID"]).copy() 
    X = X.values
    predicciones = self.mejor_modelo.predict(X)

    return img_bounding_boxes, predicciones  # Devuelve la imagen con las cajas y el DataFrame