from sklearn.cluster import KMeans
import numpy as np
import cv2

def segmentar_kmeans_y_umbral(img, n_clusters=2, umbral=70):
    """
    Segmenta la imagen utilizando KMeans y aplica un umbral adicional para identificar células infectadas.

    Args:
        img (numpy.ndarray): Imagen preprocesada (por ejemplo, transformada Wavelet).
        n_clusters (int): Número de clusters para KMeans (por defecto 2).
        umbral (int): Umbral para detectar núcleos de células infectadas.

    Returns:
        numpy.ndarray: Imagen segmentada con fondo negro (0), células grises (128),
                       y núcleos infectados blancos (255).
    """
    # Aplicar KMeans
    img_reshape = img.reshape((-1, 1))
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(img_reshape)

    # Reconstruir la imagen segmentada
    etiquetas = kmeans.labels_.reshape(img.shape)

    # Identificar la etiqueta del fondo (la clase más frecuente)
    etiqueta_fondo = np.argmax(np.bincount(etiquetas.flatten()))

    # Reasignar valores de intensidad
    img_segmentada = np.zeros_like(img, dtype=np.uint8)
    img_segmentada[etiquetas == etiqueta_fondo] = 0  # Fondo negro
    img_segmentada[etiquetas != etiqueta_fondo] = 128  # Células grises
    img_segmentada[(etiquetas != etiqueta_fondo) & (img < umbral)] = 255  # Núcleos blancos

    return img_segmentada

def aplicar_floodfill(img_segmentada):
    """
    Aplica Flood Fill para rellenar las áreas conectadas a los píxeles blancos rodeados de grises.

    Args:
        img_segmentada (numpy.ndarray): Imagen segmentada con fondo negro, células grises y núcleos blancos.

    Returns:
        numpy.ndarray: Imagen con las células infectadas completamente rellenadas.
    """
    # Crear una copia de la imagen segmentada
    img_filled = img_segmentada.copy()

    # Crear máscara para Flood Fill
    mask = np.zeros((img_segmentada.shape[0] + 2, img_segmentada.shape[1] + 2), dtype=np.uint8)

    # Detectar contornos
    contours, _ = cv2.findContours((img_segmentada == 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No se encontraron contornos.")
        return img_filled

    # Aplicar Flood Fill desde el centro de cada contorno
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cx, cy = x + w // 2, y + h // 2  # Centro del contorno
        if img_segmentada[cy, cx] == 255:  # Validar que sea blanco
            cv2.floodFill(img_filled, mask, (cx, cy), 255)

    return img_filled

def filtrar_celulas_infectadas(img_floodfilled, area_minima=2000):
    """
    Filtra las células infectadas basándose en el área mínima de los objetos.

    Args:
        img_floodfilled (numpy.ndarray): Imagen procesada con Flood Fill aplicado.
        area_minima (int): Área mínima para conservar un objeto.

    Returns:
        numpy.ndarray: Imagen binaria con las células infectadas filtradas.
    """
    # Binarizar la imagen
    _, img_binarizada = cv2.threshold(img_floodfilled, 129, 255, cv2.THRESH_BINARY)

    # Detectar contornos
    cnts, _ = cv2.findContours(img_binarizada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Crear una imagen en blanco para los contornos filtrados
    img_filtrada = np.zeros_like(img_binarizada)

    for c in cnts:
        if cv2.contourArea(c) > area_minima:  # Filtrar por área mínima
            cv2.drawContours(img_filtrada, [c], -1, 255, thickness=cv2.FILLED)

    return img_filtrada