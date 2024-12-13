import cv2
import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np

def binarizar(imagen, umbral):
    """
    Procesa una imagen aplicando binarización e inversión, y muestra los resultados.

    Parámetros:
        imagen (numpy.ndarray): Imagen en escala de grises a procesar.
        umbral (int): Valor umbral para la binarización (0-255).
    
    Returns:
        img_invertida (numpy.darray): Imagen binaria
    """
    # Binarización: los valores >= umbral se hacen blancos (255), los menores negros (0)
    _, img_binarizada = cv2.threshold(imagen, umbral, 255, cv2.THRESH_BINARY)

    # Invertir la imagen binarizada
    img_invertida = cv2.bitwise_not(img_binarizada)

    # # Mostrar la imagen original, binarizada e invertida
    # plt.figure(figsize=(15, 5))

    # # Imagen original
    # plt.subplot(1, 3, 1)
    # plt.imshow(imagen_wavelet, cmap="gray")
    # plt.title("Imagen Wavelet")
    # plt.axis("off")

    # # Imagen binarizada
    # plt.subplot(1, 3, 2)
    # plt.imshow(img_binarizada, cmap="gray")
    # plt.title("Imagen Binarizada")
    # plt.axis("off")

    # # Imagen invertida
    # plt.subplot(1, 3, 3)
    # plt.imshow(img_invertida, cmap="gray")
    # plt.title("Imagen Invertida")
    # plt.axis("off")

    # plt.tight_layout()
    # plt.show()

    return img_invertida

def aplicar_watershed(imagen, level=80):
    """
    Aplica la transformada de la distancia y el algoritmo Watershed a una imagen dada.

    Args:
        imagen (numpy.ndarray): Imagen de entrada en formato binario (0 y 255).
        level (int, optional): El nivel de umbralización para el algoritmo Watershed. Default es 80.

    Returns:
        img_ws (numpy.ndarray): Imagen binaria resultante del algoritmo Watershed.
    """
    # Convertir la imagen binaria a formato SimpleITK
    umbral = sitk.GetImageFromArray(imagen)

    # Aplicar transformada de la distancia
    dist_transformada = sitk.SignedMaurerDistanceMap(umbral, insideIsPositive=True, useImageSpacing=True)

    # Invertir la transformada de distancia
    dist_transformada = sitk.InvertIntensity(dist_transformada)

    # Aplicar el algoritmo Watershed con el nivel proporcionado
    etiquetas = sitk.MorphologicalWatershed(dist_transformada, markWatershedLine=True, level=level, fullyConnected=True)

    # Crear una máscara para regiones etiquetadas
    ws = sitk.Mask(umbral, etiquetas)

    # Convertir las imágenes a arrays de numpy para visualización
    umbral_array = sitk.GetArrayFromImage(umbral)
    dist_array = sitk.GetArrayFromImage(dist_transformada)
    etiquetas_array = sitk.GetArrayFromImage(etiquetas)
    img_ws = sitk.GetArrayFromImage(ws)

    # Visualizar los resultados
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))

    ax[0, 0].imshow(dist_array, cmap='gray')
    ax[0, 0].set_title('Transformada de la distancia', fontsize=15)

    ax[0, 1].imshow(umbral_array, cmap='gray')
    ax[0, 1].set_title('Imagen binaria', fontsize=15)

    ax[1, 0].imshow(etiquetas_array, cmap='gray')
    ax[1, 0].set_title('Watershed', fontsize=15)

    ax[1, 1].imshow(img_ws, cmap='gray')
    ax[1, 1].set_title('Watershed imagen binaria', fontsize=15)

    plt.tight_layout()
    plt.show()

    return img_ws

def aplicar_dilatacion_y_erosion(img_binarizada, kernel_size=(3, 3), iterations=1, mostrar_resultados=True):
    """
    Aplica una dilatación seguida de una erosión a una imagen binarizada.

    Args:
        img_binarizada (numpy.ndarray): Imagen binarizada (en escala de grises o binaria).
        kernel_size (tuple): Tamaño del kernel estructural (por defecto (3, 3)).
        iterations (int): Número de iteraciones para las operaciones morfológicas (por defecto 1).
        mostrar_resultados (bool): Si es True, muestra las imágenes procesadas (por defecto True).

    Returns:
        numpy.ndarray: Imagen procesada después de la dilatación y la erosión.
    """
    # Crear el kernel estructural
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

    # Aplicar dilatación
    img_dilated = cv2.dilate(img_binarizada, kernel, iterations=iterations)

    # Aplicar erosión
    img_eroded = cv2.erode(img_dilated, kernel, iterations=iterations)

    if mostrar_resultados:
        # Mostrar las imágenes procesadas
        plt.figure(figsize=(15, 5))

        # Imagen binarizada original
        plt.subplot(1, 3, 1)
        plt.imshow(img_binarizada, cmap="gray")
        plt.title("Imagen Binarizada")
        plt.axis("off")

        # Imagen dilatada
        plt.subplot(1, 3, 2)
        plt.imshow(img_dilated, cmap="gray")
        plt.title("Imagen Dilatada")
        plt.axis("off")

        # Imagen erosionada
        plt.subplot(1, 3, 3)
        plt.imshow(img_eroded, cmap="gray")
        plt.title("Imagen Erosionada")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    return img_eroded

def dibujar_bounding_boxes(imagen, contornos, color=(0, 255, 0), grosor=1):
    """
    Dibuja bounding boxes alrededor de los contornos especificados en una imagen.

    Parámetros:
        imagen (numpy.ndarray): Imagen de entrada (en escala de grises o BGR).
        contornos (list): Lista de contornos detectados.
        color (tuple): Color del rectángulo en formato BGR (por defecto: verde).
        grosor (int): Grosor de las líneas del rectángulo (por defecto: 1).

    Retorna:
        numpy.ndarray: Imagen con los bounding boxes dibujados.
    """
    # Convertir la imagen a BGR si está en escala de grises
    if len(imagen.shape) == 2:  # Verifica si la imagen tiene un solo canal (grises)
        img_bboxes = cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)
    else:
        img_bboxes = imagen.copy()  # Copiar la imagen si ya está en BGR

    # Dibujar los bounding boxes en la imagen
    for contorno in contornos:
        x, y, w, h = cv2.boundingRect(contorno)  # Calcular bounding box
        cv2.rectangle(img_bboxes, (x, y), (x + w, y + h), color, grosor)  # Dibujar rectángulo

    # Mostrar la imagen con los bounding boxes
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(img_bboxes, cv2.COLOR_BGR2RGB))  # Convertir a RGB para Matplotlib
    plt.title("Bounding Boxes")
    plt.axis('off')
    plt.show()

    return img_bboxes

def procesar_recortes_y_watershed(imagen_ws, contornos, img_original_bgr, level=30, umbral_area_minima=800, filas=5, columnas=5):
    """
    Procesa cada recorte generado por los bounding boxes, aplica watershed y dibuja los resultados.
    
    Args:
        imagen_ws (numpy.ndarray): Imagen binaria procesada por watershed.
        contornos (list): Contornos detectados en la imagen original.
        img_original_bgr (numpy.ndarray): Imagen original en formato BGR.
        level (int): Nivel para el algoritmo Watershed.
        umbral_area_minima (int): Área mínima para considerar un contorno válido.
        filas (int): Número de filas para la cuadrícula de visualización.
        columnas (int): Número de columnas para la cuadrícula de visualización.
        
    Returns:
        img_original_bgr (numpy.ndarray): Imagen original con los nuevos bounding boxes dibujados.
    """
    resultados_watershed = []

    # Iterar sobre los contornos y procesar los recortes
    for i, contorno in enumerate(contornos):
        x, y, w, h = cv2.boundingRect(contorno)  # Obtener bounding box
        recorte = imagen_ws[y:y+h, x:x+w]       # Recortar región de interés

        # Aplicar watershed al recorte
        watershed_array = aplicar_watershed(recorte, level=level)

        # Detectar contornos en el resultado del watershed
        contornos_recorte, _ = cv2.findContours(watershed_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contornos_filtrados = [c for c in contornos_recorte if cv2.contourArea(c) > umbral_area_minima]

        # Dibujar bounding boxes sobre la imagen original
        for c in contornos_filtrados:
            x_recorte, y_recorte, w_recorte, h_recorte = cv2.boundingRect(c)
            cv2.rectangle(img_original_bgr, (x + x_recorte, y + y_recorte), 
                          (x + x_recorte + w_recorte, y + y_recorte + h_recorte), (0, 255, 0), 1)

        # Almacenar el resultado del watershed
        resultados_watershed.append(watershed_array)

        # Mostrar en cuadrícula cada recorte procesado
        if i % (filas * columnas) == 0:
            if i > 0:
                plt.tight_layout()
                plt.show()
            fig, axs = plt.subplots(filas, columnas, figsize=(15, 15))
            axs = axs.ravel()

        axs[i % (filas * columnas)].imshow(watershed_array, cmap='gray')
        axs[i % (filas * columnas)].set_title(f'Recorte {i+1}')
        axs[i % (filas * columnas)].axis('off')

    # Mostrar la última figura
    plt.tight_layout()
    plt.show()

    return img_original_bgr

# Uso del pipeline
def segmentar_recortes(imagen_wavelet, imagen_ws, contornos, level=30, umbral_area_minima=800, filas=5, columnas=5):
    """
    Ejecuta el pipeline completo de procesamiento: watershed, contornos y bounding boxes.
    
    Args:
        imagen_wavelet (numpy.ndarray): Imagen original en escala de grises.
        imagen_ws (numpy.ndarray): Imagen binaria procesada por watershed.
        contornos (list): Contornos detectados en la imagen.
        level (int): Nivel para watershed.
        umbral_area_minima (int): Área mínima para contornos válidos.
        filas (int): Número de filas para cuadrícula.
        columnas (int): Número de columnas para cuadrícula.
    """
    # Convertir la imagen original a formato BGR
    img_original_bgr = cv2.cvtColor(imagen_wavelet, cv2.COLOR_GRAY2BGR)

    # Dibujar los bounding boxes iniciales
    img_original_bgr = dibujar_bounding_boxes(img_original_bgr, contornos)

    # Procesar recortes y watershed
    img_con_boxes = procesar_recortes_y_watershed(imagen_ws, contornos, img_original_bgr, 
                                                  level=level, umbral_area_minima=umbral_area_minima, 
                                                  filas=filas, columnas=columnas)

    # Mostrar la imagen final con nuevos bounding boxes
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(img_con_boxes, cv2.COLOR_BGR2RGB))
    plt.title("Imagen Original con Nuevos Bounding Boxes")
    plt.axis('off')
    plt.show()
    return img_con_boxes