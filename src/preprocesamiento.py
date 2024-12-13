import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt
import os
from sklearn.cluster import KMeans

def reducir_ruido(img):
    """
    Aplica el filtro Non-Local Means Denoising para reducir ruido.

    Args:
        img (numpy.ndarray): Imagen en formato RGB.

    Returns:
        numpy.ndarray: Imagen filtrada.
    """
    imagen_suavizada = cv2.GaussianBlur(img, (5, 5), 0)
    img_bgr = cv2.cvtColor(imagen_suavizada, cv2.COLOR_RGB2BGR)
    img_denoised = cv2.fastNlMeansDenoisingColored(img_bgr, None, 10, 10, 7, 21)
    img_rgb_denoised = cv2.cvtColor(img_denoised, cv2.COLOR_BGR2RGB)
    return img_denoised, img_rgb_denoised

def separar_canales(img):
    """
    Separa los canales RGB de la imagen.

    Args:
        img (numpy.ndarray): Imagen en formato RGB.

    Returns:
        tuple: Canales rojo, verde y azul.
    """
    canal_rojo = img[:, :, 0].astype('uint8')
    canal_verde = img[:, :, 1].astype('uint8')
    canal_azul = img[:, :, 2].astype('uint8')
    return canal_rojo, canal_verde, canal_azul

def seleccionar_canal_mayor_contraste(canal_rojo, canal_verde, canal_azul):
    """
    Selecciona el canal con mayor contraste (desviación estándar).

    Args:
        canal_rojo, canal_verde, canal_azul (numpy.ndarray): Canales de la imagen.

    Returns:
        numpy.ndarray: Canal seleccionado.
    """
    contraste_rojo = np.std(canal_rojo)
    contraste_verde = np.std(canal_verde)
    contraste_azul = np.std(canal_azul)

    if contraste_azul >= max(contraste_rojo, contraste_verde):
        print("Canal Azul seleccionado (mayor contraste).")
        return canal_azul
    elif contraste_verde >= max(contraste_rojo, contraste_azul):
        print("Canal Verde seleccionado (mayor contraste).")
        return canal_verde
    else:
        print("Canal Rojo seleccionado (mayor contraste).")
        return canal_rojo
    
def aplicar_fft(canal):
    """
    Aplica la Transformada Rápida de Fourier (FFT) al canal seleccionado y visualiza 
    tanto el canal como su espectro de frecuencias.

    Args:
        canal (numpy.ndarray): Canal seleccionado de la imagen en escala de grises.
    """
    f_transformada = np.fft.fft2(canal)
    f_centrada = np.fft.fftshift(f_transformada)  # Centrar frecuencias bajas en el centro del espectro
    espectro = np.log(1 + np.abs(f_centrada))

    # # Graficar el canal original y el espectro de frecuencias
    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.title("Canal Seleccionado")
    # plt.imshow(canal, cmap='gray')
    # plt.axis('off')
    # plt.subplot(1, 2, 2)
    # plt.title("Espectro de Frecuencias (FFT)")
    # plt.imshow(espectro, cmap='gray')
    # plt.axis('off')
    # plt.tight_layout()
    # plt.show() 

def aplicar_wavelet(canal):
    """
    Aplica la Transformada Wavelet (DWT) y retorna la componente LL normalizada.

    Args:
        canal (numpy.ndarray): Canal seleccionado.

    Returns:
        numpy.ndarray: Componente LL de la Transformada Wavelet.
    """
    coeficientes = pywt.dwt2(canal, 'haar')  # Usar wavelet 'haar'
    LL, (LH, HL, HH) = coeficientes
    imagen_wavelet = cv2.normalize(LL, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return imagen_wavelet

def aplicar_ecualizado(imagen, nombre="Canal", mostrar_resultados=True):
    """
    Aplica la ecualización de histograma a una imagen y muestra los resultados junto con los histogramas.

    Args:
        imagen (numpy.ndarray): Imagen de entrada en escala de grises.
        nombre (str): Nombre del canal para identificarlo en los resultados.
        mostrar_resultados (bool): Si es True, muestra las imágenes y los histogramas (por defecto True).

    Returns:
        numpy.ndarray: Imagen ecualizada.
    """
    # Ecualización del canal seleccionado
    canal_ecualizado = cv2.equalizeHist(imagen.astype('uint8'))

    if mostrar_resultados:
        # Gráficos
        plt.figure(figsize=(15, 10))

        # Imagen del canal seleccionado
        plt.subplot(2, 2, 1)
        plt.imshow(imagen, cmap='gray')
        plt.title(f"Canal Seleccionado {nombre} - Original")
        plt.axis('off')

        # Histograma de la imagen original
        plt.subplot(2, 2, 2)
        plt.hist(imagen.ravel(), bins=256, range=(0, 255), color='blue')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.title(f"Histograma - Canal {nombre} Original")
        plt.xlabel("Intensidad de los píxeles")
        plt.ylabel("Cantidad de píxeles")

        # Imagen del canal ecualizado
        plt.subplot(2, 2, 3)
        plt.imshow(canal_ecualizado, cmap='gray')
        plt.title(f"Canal Seleccionado {nombre} - Ecualizado")
        plt.axis('off')

        # Histograma de la imagen ecualizada
        plt.subplot(2, 2, 4)
        plt.hist(canal_ecualizado.ravel(), bins=256, range=(0, 255), color='blue')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.title(f"Histograma - Canal {nombre} Ecualizado")
        plt.xlabel("Intensidad de los píxeles")
        plt.ylabel("Cantidad de píxeles")

        plt.tight_layout()
        plt.show()

    return canal_ecualizado

def escalar_con_clipping(imagen, low=0, high=255):
    """
    Escala las intensidades de una imagen a un rango definido y aplica clipping.

    Args:
        imagen (numpy.ndarray): Imagen de entrada en escala de grises.
        low (int): Valor mínimo del rango deseado.
        high (int): Valor máximo del rango deseado.

    Returns:
        numpy.ndarray: Imagen escalada y ajustada al rango especificado.
    """
    # Convertir a float para evitar pérdida de precisión durante los cálculos
    imagen_float = imagen.astype(np.float32)

    # Normalizar las intensidades a [0, 1]
    imagen_normalizada = (imagen_float - imagen_float.min()) / (imagen_float.max() - imagen_float.min())

    # Escalar al rango deseado
    imagen_escalada = imagen_normalizada * (high - low) + low

    # Aplicar clipping para asegurar que esté dentro del rango [low, high]
    imagen_clipped = np.clip(imagen_escalada, low, high)

    return imagen_clipped.astype(np.uint8)


def seleccionar_y_preprocesar(imagen, mostrar_resultados=True):
    """
    Analiza las características de la imagen y selecciona el mejor método de preprocesamiento.

    Args:
        imagen (numpy.ndarray): Imagen en escala de grises.
        mostrar_resultados (bool): Si es True, muestra los gráficos del proceso (por defecto True).

    Returns:
        str: Método seleccionado ("original", "ecualizado", "escalado").
        numpy.ndarray: Imagen preprocesada.
    """
    # Métricas de la imagen
    std_dev = np.std(imagen)  # Desviación estándar
    rango_dinamico = np.max(imagen) - np.min(imagen)  # Rango dinámico

    # Decisiones basadas en las métricas
    if rango_dinamico < 150:  # Rango estrecho
        metodo = "escalado"
        imagen_preprocesada = cv2.normalize(imagen, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    elif 150 < rango_dinamico < 155:  # Baja variabilidad
        metodo = "ecualizado"
        imagen_preprocesada = cv2.equalizeHist(imagen.astype('uint8'))
    else:  # Imagen con buen contraste ya
        metodo = "original"
        imagen_preprocesada = imagen

    # Mostrar resultados si se requiere
    if mostrar_resultados:
        plt.figure(figsize=(15, 10))

        # Imagen original
        plt.subplot(2, 2, 1)
        plt.imshow(imagen, cmap='gray')
        plt.title("Imagen Original")
        plt.axis('off')

        # Histograma de la imagen original
        plt.subplot(2, 2, 2)
        plt.hist(imagen.ravel(), bins=256, range=(0, 255), color='blue')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.title("Histograma - Imagen Original")
        plt.xlabel("Intensidad de los píxeles")
        plt.ylabel("Cantidad de píxeles")

        # Imagen preprocesada
        plt.subplot(2, 2, 3)
        plt.imshow(imagen_preprocesada, cmap='gray')
        plt.title(f"Imagen Preprocesada ({metodo.capitalize()})")
        plt.axis('off')

        # Histograma de la imagen preprocesada
        plt.subplot(2, 2, 4)
        plt.hist(imagen_preprocesada.ravel(), bins=256, range=(0, 255), color='blue')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.title(f"Histograma - Imagen Preprocesada ({metodo.capitalize()})")
        plt.xlabel("Intensidad de los píxeles")
        plt.ylabel("Cantidad de píxeles")

        plt.tight_layout()
        plt.show()

    return metodo, imagen_preprocesada

def binarizar_con_kmeans(img, n_clusters=2):
    """
    Binariza la imagen utilizando KMeans con 2 clusters y la invierte
    para que el fondo sea negro y las células blancas.

    Args:
        img (numpy.ndarray): Imagen preprocesada (por ejemplo, en escala de grises).
        n_clusters (int): Número de clusters para KMeans (por defecto 2).

    Returns:
        numpy.ndarray: Imagen binarizada con fondo negro (0) y primer cluster blanco (255).
    """
    # Convertir la imagen a un formato adecuado para KMeans (un solo canal)
    img_reshape = img.reshape((-1, 1))

    # Aplicar KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(img_reshape)

    # Obtener las etiquetas de los clusters y reconstruir la imagen segmentada
    etiquetas = kmeans.labels_.reshape(img.shape)

    # Crear una imagen binarizada con fondo blanco (255) y el primer cluster negro (0)
    img_binarizada = np.zeros_like(img, dtype=np.uint8)
    img_binarizada[etiquetas != np.argmax(np.bincount(etiquetas.flatten()))] = 255  # Células blancas (invertido)

    return img_binarizada

def aplicar_filtro_mediana(img_binarizada, ksize=15):
    """
    Aplica un filtro de mediana a la imagen binarizada para reducir el ruido de tipo 'sal y pimienta'.

    Args:
        img_binarizada (numpy.ndarray): Imagen binarizada.
        ksize (int): Tamaño del kernel para el filtro de mediana (por defecto 3).

    Returns:
        numpy.ndarray: Imagen binarizada después de aplicar el filtro de mediana.
    """
    return cv2.medianBlur(img_binarizada, ksize)

def aplicar_operaciones_morfologicas(img_binarizada, kernel_dil=7, kernel_ero=3):
    """
    Aplica dilatación y erosión a la imagen binarizada para cerrar círculos dentro de las células.

    Args:
        img_binarizada (numpy.ndarray): Imagen binarizada.
        kernel_dil (int): Tamaño del kernel para la dilatación (por defecto 7).
        kernel_ero (int): Tamaño del kernel para la erosión (por defecto 3).

    Returns:
        numpy.ndarray: Imagen después de las operaciones morfológicas.
    """
    kernel_dilatacion = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_dil, kernel_dil))
    kernel_erosion = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_ero, kernel_ero))

    # Aplicar dilatación
    img_dilatada = cv2.dilate(img_binarizada, kernel_dilatacion, iterations=1)

    # Aplicar erosión
    img_final = cv2.erode(img_dilatada, kernel_erosion, iterations=1)

    return img_final


def rellenar_celulas(img_binaria):
    """
    Rellena las células detectadas en la imagen binaria para eliminar huecos internos.

    Args:
        img_binaria (numpy.ndarray): Imagen binaria (fondo negro, objetos blancos).

    Returns:
        numpy.ndarray: Imagen binaria con las células rellenadas.
    """
    # Crear una copia de la imagen binaria
    img_rellena = img_binaria.copy()

    # Detectar contornos
    contornos, _ = cv2.findContours(img_rellena, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Rellenar cada contorno
    cv2.drawContours(img_rellena, contornos, -1, 255, thickness=cv2.FILLED)

    return img_rellena
