import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops 

def calcular_glcm(imagen):
    """
    Calcula la matriz GLCM y sus propiedades (contraste, energía, homogeneidad).
    Args:
        imagen (numpy.ndarray): Imagen en escala de grises.
    Returns:
        dict: Diccionario con las propiedades de la GLCM.
    """

    if imagen.size == 0 or np.all(imagen == 0):
        return {"Contraste": 0, "Energía": 0, "Homogeneidad": 0}

    glcm = graycomatrix(imagen, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)

    propiedades = {
        "Contraste": graycoprops(glcm, 'contrast')[0, 0],
        "Energía": graycoprops(glcm, 'energy')[0, 0],
        "Homogeneidad": graycoprops(glcm, 'homogeneity')[0, 0]
    }
    return propiedades

def construir_base_datos(canal_seleccionado, contornos):
    """
    Construye una base de datos con las características GLCM de cada célula detectada.

    Args:
        canal_seleccionado (numpy.ndarray): Imagen en escala de grises.
        contornos (list): Lista de contornos detectados.

    Returns:
        pandas.DataFrame: DataFrame con las características y coordenadas de las células.
    """
    columnas = ["ID", "X", "Y", "Width", "Height", "Contraste", "Energía", "Homogeneidad"]
    data = []

    for i, contorno in enumerate(contornos):
        x, y, w, h = cv2.boundingRect(contorno)
        recorte = canal_seleccionado[y:y+h, x:x+w]

        if recorte.size > 0 and np.any(recorte):
            caracteristicas = calcular_glcm(recorte)
            data.append([
                i, x, y, w, h,
                caracteristicas["Contraste"],
                caracteristicas["Energía"],
                caracteristicas["Homogeneidad"]
            ])

    df = pd.DataFrame(data, columns=columnas)
    return df

# Seleccion
# Quedarnos