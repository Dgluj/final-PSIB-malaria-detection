import numpy as np
import cv2
import matplotlib.pyplot as plt
import SimpleITK as sitk

def dibujar_bounding_boxes_en_identificadas(img, dataframe):
    """
    Dibuja los bounding boxes de las células infectadas y no infectadas en la imagen original.
    
    Args:
        img (numpy.ndarray): Imagen original en color.
        dataframe (pandas.DataFrame): DataFrame con las características de las células, incluyendo la clasificación.
    
    Returns:
        numpy.ndarray: Imagen con los bounding boxes dibujados.
    """
    # Recorrer cada fila del DataFrame
    for _, fila in dataframe.iterrows():
        x, y, w, h = fila["X"], fila["Y"], fila["Width"], fila["Height"]
        # Determinar color y texto según la clasificación
        if fila["Infectada"] == 1:
            color = (0, 0, 255)  # Rojo para infectada
            texto = "Infectada"
        else:
            color = (0, 255, 0)  # Verde para sana
            texto = "Sana"
        
        # Dibujar el rectángulo del bounding box
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

        # Colocar el texto sobre el bounding box
        cv2.putText(img, texto, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    
    return img
