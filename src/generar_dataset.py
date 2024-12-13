# src/generar_dataset.py
import os
import glob
import cv2
import pandas as pd
import numpy as np

from src.carga_imagenes import cargar_imagenes
from src.preprocesamiento import (reducir_ruido, separar_canales, seleccionar_canal_mayor_contraste, 
                                  binarizar_con_kmeans, aplicar_filtro_mediana, aplicar_operaciones_morfologicas, rellenar_celulas)
from src.segmentacion import aplicar_watershed
from src.extraccion_de_caracteristicas import construir_base_datos

def asignar_etiquetas(df_imagen):
    """
    Asigna etiquetas a las células en el DataFrame df_imagen.
    Por ahora, las marcamos todas como 0 (sanas).
    Cuando definas un criterio o tengas datos para etiquetar infectadas, edita esta función.
    """
    df_imagen['label'] = 0  # Todo sano por defecto
    return df_imagen

def generar_dataset(ruta_imagenes, archivo_salida):
    # Suponiendo que en la carpeta data/ tienes las 14 imágenes
    # Cargamos todas las imágenes con la función ya existente
    imagenes = {}
    for archivo in os.listdir(ruta_imagenes):
        if archivo.lower().endswith(('.png', '.jpg', '.jpeg')):
            ruta = os.path.join(ruta_imagenes, archivo)
            img = cv2.imread(ruta)
            # Las funciones esperan imágenes RGB, convertimos
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imagenes[archivo] = img_rgb

    # DataFrame total
    df_total = pd.DataFrame()

    # Procesar cada imagen
    for nombre, img in imagenes.items():
        # Pipeline de preprocesamiento (del main original)
        # 1. Reducir ruido
        img_denoised, img_rgb_denoised = reducir_ruido(img)

        # 2. Separar canales
        canal_rojo, canal_verde, canal_azul = separar_canales(img_rgb_denoised)

        # 3. Seleccionar canal mayor contraste
        canal_seleccionado = seleccionar_canal_mayor_contraste(canal_rojo, canal_verde, canal_azul)

        # 4. Binarizar con k-means
        img_binarizada = binarizar_con_kmeans(canal_seleccionado)

        # 5. Filtrar mediana
        img_mediana = aplicar_filtro_mediana(img_binarizada)

        # 6. Operaciones morfológicas
        img_morfo = aplicar_operaciones_morfologicas(img_mediana)

        # 7. Rellenar células
        img_rellena = rellenar_celulas(img_morfo)

        # 8. Aplicar Watershed
        img_ws, _ = aplicar_watershed(img_rellena, level=40)

        # 9. Encontrar contornos
        contornos, _ = cv2.findContours(img_ws, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 10. Construir base de datos de características (GLCM) para esta imagen
        df_imagen = construir_base_datos(canal_seleccionado, contornos)

        # Agregar columna con nombre de la imagen
        df_imagen['imagen'] = nombre

        # Asignar etiquetas (por ahora dummy)
        df_imagen = asignar_etiquetas(df_imagen)

        # Unir con el total
        df_total = pd.concat([df_total, df_imagen], ignore_index=True)

    # Guardar el DataFrame completo
    df_total.to_csv(archivo_salida, index=False)
    print(f"Dataset guardado en {archivo_salida}")