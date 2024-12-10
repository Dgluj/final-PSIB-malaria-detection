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
