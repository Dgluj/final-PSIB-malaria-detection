import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops, canny, blob_log 
from skimage.measure import label, regionprops

def calcular_glcm(imagen):
    """
    Calcula la matriz GLCM y sus propiedades (contraste, energía, homogeneidad, cluster shade,
    cluster prominencia, correlación de Haralick y entropía).
    Args:
        imagen (numpy.ndarray): Imagen en escala de grises.
    Returns:
        dict: Diccionario con las propiedades de la GLCM.
    """

    if imagen.size == 0 or np.all(imagen == 0):
        return {
            "Contraste": 0,
            "Energía": 0,
            "Homogeneidad": 0,
            "Cluster Shade": 0,
            "Cluster Prominencia": 0,
            "Correlación Haralick": 0,
            "Entropía": 0
        }
    glcm = graycomatrix(imagen, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    
    # Cluster Shade y Cluster Prominencia (cálculos manuales)
    glcm_normalizado = glcm[:, :, 0, 0]
    niveles = np.arange(256)
    i, j = np.meshgrid(niveles, niveles, indexing='ij')
    media_i = np.sum(i * glcm_normalizado)
    media_j = np.sum(j * glcm_normalizado)
    
    cluster_shade = np.sum(((i + j - media_i - media_j) ** 3) * glcm_normalizado)
    cluster_prominencia = np.sum(((i + j - media_i - media_j) ** 4) * glcm_normalizado)

    # Entropía
    entropia = -np.sum(glcm_normalizado * np.log2(glcm_normalizado + 1e-10))
    
    propiedades = {
        "Contraste": graycoprops(glcm, 'contrast')[0, 0],
        "Energía": graycoprops(glcm, 'energy')[0, 0],
        "Homogeneidad": graycoprops(glcm, 'homogeneity')[0, 0],
        "Correlación Haralick": graycoprops(glcm, 'correlation')[0, 0],
        "Cluster Shade": cluster_shade,
        "Cluster Prominencia": cluster_prominencia,
        "Entropía": entropia
    }
    return propiedades

def calcular_nuevas_caracteristicas(imagen, contorno):
    """
    Calcula características adicionales de forma y textura de las células.

    Args:
        imagen (numpy.ndarray): Imagen en escala de grises.
        contorno (list): Contorno de una célula.

    Returns:
        dict: Diccionario con las nuevas características.
    """
    # Área y perímetro de la célula
    area = cv2.contourArea(contorno)
    perimetro = cv2.arcLength(contorno, True)

    # Circularidad
    if perimetro == 0:
        circularidad = 0
    else:
        circularidad = 4 * np.pi * area / (perimetro ** 2)

    return {
        "Área": area,
        "Perímetro": perimetro,
        "Circularidad": circularidad,
    }

def construir_base_datos(canal_seleccionado, contornos):
    """
    Construye una base de datos con las características GLCM y las características adicionales de cada célula detectada.

    Args:
        canal_seleccionado (numpy.ndarray): Imagen en escala de grises.
        contornos (list): Lista de contornos detectados.

    Returns:
        pandas.DataFrame: DataFrame con las características y coordenadas de las células.
    """
    columnas = [
        "ID", "X", "Y", "Width", "Height", 
        "Área", "Perímetro", "Circularidad", 
        "Contraste", "Energía", "Homogeneidad", "Cluster Shade", 
        "Cluster Prominencia", "Correlación Haralick", "Entropía"
    ]
    data = []

    for i, contorno in enumerate(contornos):
        x, y, w, h = cv2.boundingRect(contorno)
        recorte = canal_seleccionado[y:y+h, x:x+w]

        # Solo procesar si el recorte no está vacío y tiene valores no nulos
        if recorte.size > 0 and np.any(recorte):
            # Calcular características de GLCM
            caracteristicas_glcm = calcular_glcm(recorte)

            # Calcular características adicionales de forma y textura
            caracteristicas_forma = calcular_nuevas_caracteristicas(canal_seleccionado, contorno)

            # Agregar las características al dataset
            data.append([
                i, x, y, w, h, 
                caracteristicas_forma["Área"], 
                caracteristicas_forma["Perímetro"], 
                caracteristicas_forma["Circularidad"], 
                caracteristicas_glcm["Contraste"], 
                caracteristicas_glcm["Energía"], 
                caracteristicas_glcm["Homogeneidad"], 
                caracteristicas_glcm["Cluster Shade"], 
                caracteristicas_glcm["Cluster Prominencia"], 
                caracteristicas_glcm["Correlación Haralick"], 
                caracteristicas_glcm["Entropía"]
            ])

    df = pd.DataFrame(data, columns=columnas)
    return df

def clasificar_celulas(dataframe, umbrales):
    """
    Clasifica las células en el DataFrame como infectadas o no, basado en los umbrales.

    Args:
        dataframe (pandas.DataFrame): DataFrame con las características de las células.
        umbrales (dict): Diccionario con los umbrales de las características.

    Returns:
        pandas.DataFrame: DataFrame actualizado con una columna adicional "Infectada".
    """
    def clasificar_infectada(fila):
        """
        Clasifica una célula como infectada o no, basado en umbrales.

        Args:
            fila (pandas.Series): Fila del DataFrame con las características de la célula.

        Returns:
            int: 1 si está infectada, 0 si no.
        """
        # Verificar las condiciones para cada característica, considerando si está definida en el diccionario de umbrales.
        condiciones = [
            fila["Área"] > umbrales.get("Área", 0),
            fila["Perímetro"] > umbrales.get("Perímetro", 0),
            fila["Circularidad"] < umbrales.get("Circularidad", float('inf')),
            fila["Contraste"] > umbrales.get("Contraste", 0),
            fila["Energía"] > umbrales.get("Energía", 0),
            fila["Homogeneidad"] > umbrales.get("Homogeneidad", 0),
            fila["Cluster Shade"] < umbrales.get("Cluster Shade", float('inf')),
            fila["Cluster Prominencia"] < umbrales.get("Cluster Prominencia", float('inf')),
            fila["Correlación Haralick"] > umbrales.get("Correlación Haralick", 0),
            fila["Entropía"] < umbrales.get("Entropía", float('inf'))
        ]

        # Si solo la condición de área se cumple, la célula está infectada.
        return 1 if condiciones[0] else 0

    # Aplicar la clasificación a cada fila del DataFrame
    dataframe["Infectada"] = dataframe.apply(clasificar_infectada, axis=1)
    return dataframe

def clasificacion_final(dataframe, umbrales):
    """
    Realiza la clasificación final de las células y genera un DataFrame balanceado con células infectadas y no infectadas.

    Args:
        dataframe (pandas.DataFrame): DataFrame con las características de las células.
        umbrales (dict): Diccionario con los umbrales de las características.

    Returns:
        pandas.DataFrame: DataFrame final balanceado con células infectadas y no infectadas.
    """
    # Clasificar las células
    dataframe = clasificar_celulas(dataframe, umbrales)

    # Filtrar células infectadas y no infectadas
    infectadas = dataframe[dataframe["Infectada"] == 1]
    no_infectadas = dataframe[dataframe["Infectada"] == 0]

    # Seleccionar una cantidad aleatoria de no infectadas igual al número de infectadas
    no_infectadas_sample = no_infectadas.sample(n=len(infectadas), random_state=42)

    # Concatenar los dos grupos
    dataframe_final = pd.concat([infectadas, no_infectadas_sample]).reset_index(drop=True)

    return dataframe_final

