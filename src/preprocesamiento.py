import cv2
import numpy as np
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
        #print("Canal Azul seleccionado (mayor contraste).")
        return canal_azul
    elif contraste_verde >= max(contraste_rojo, contraste_azul):
        #print("Canal Verde seleccionado (mayor contraste).")
        return canal_verde
    else:
        #print("Canal Rojo seleccionado (mayor contraste).")
        return canal_rojo

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