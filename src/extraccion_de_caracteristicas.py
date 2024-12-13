import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops 
from pyfeats import glcm_features

# Plan A
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

def construir_base_datos_plan_A(canal_seleccionado, contornos):
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

#Plan B
def calcular_glcm_manual(imagen, distancia=1):
    """
    Calcula la matriz GLCM manualmente para una imagen en escala de grises.

    Args:
        imagen (numpy.ndarray): Imagen en escala de grises.
        distancia (int): Distancia entre píxeles para calcular la co-ocurrencia.

    Returns:
        numpy.ndarray: Matriz GLCM normalizada.
    """
    niveles = 256  # Suponemos 8 bits (0-255)
    glcm = np.zeros((niveles, niveles), dtype=np.uint32)

    # Calcular la co-ocurrencia
    for i in range(imagen.shape[0] - distancia):
        for j in range(imagen.shape[1] - distancia):
            fila = imagen[i, j]
            columna = imagen[i + distancia, j]  # Direccion horizontal por defecto
            glcm[fila, columna] += 1

    # Normalización para obtener probabilidades
    if glcm.sum() > 0:
        glcm = glcm / glcm.sum()
    return glcm

def calcular_caracteristicas_glcm_pyfeats(imagen):
    """
    Calcula las características GLCM usando pyfeats.
    Args:
        imagen (numpy.ndarray): Imagen en escala de grises.
    Returns:
        dict: Diccionario con las características de GLCM.
    """
    if imagen.size == 0 or np.all(imagen == 0):
        return {"Contrast": 0, "Energy": 0, "Homogeneity": 0}
    
    glcm = calcular_glcm_manual(imagen)  # Crear la matriz GLCM
    if glcm.size == 0 or not np.any(glcm):
        return {"Contrast": 0, "Energy": 0, "Homogeneity": 0}
    
    try:
        features = glcm_features(glcm)  # Obtener todas las características disponibles
    except ValueError as e:
        print(f"Error al calcular las características GLCM: {e}")
        return {}
    
    return features

def construir_base_datos_plan_B(canal_seleccionado, contornos):
    columnas = ["ID", "X", "Y", "Width", "Height", "Contraste", "Energía", "Homogeneidad"]
    data = []

    for i, contorno in enumerate(contornos):
        x, y, w, h = cv2.boundingRect(contorno)
        recorte = canal_seleccionado[y:y + h, x:x + w]

        if recorte.size > 0:
            caracteristicas = calcular_caracteristicas_glcm_pyfeats(recorte)
            contraste = caracteristicas.get("Contrast", 0)
            energia = caracteristicas.get("Energy", 0)
            homogeneidad = caracteristicas.get("Homogeneity", 0)

            data.append([i, x, y, w, h, contraste, energia, homogeneidad])

    df = pd.DataFrame(data, columns=columnas)
    return df