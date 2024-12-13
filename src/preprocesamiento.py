import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt

def reducir_ruido(img):
    """
    Aplica el filtro Non-Local Means Denoising para reducir ruido.

    Args:
        img (numpy.ndarray): Imagen en formato RGB.

    Returns:
        numpy.ndarray: Imagen filtrada.
    """
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_denoised = cv2.fastNlMeansDenoisingColored(img_bgr, None, 10, 10, 5, 21)
    img_rgb_denoised = cv2.cvtColor(img_denoised, cv2.COLOR_BGR2RGB)
    return img_rgb_denoised

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

def aplicar_ecualizado(imagen_wavelet, mostrar_resultados=True):
    """
    Aplica la ecualización de histograma a una imagen y muestra los resultados junto con los histogramas.

    Args:
        imagen_wavelet (numpy.ndarray): Imagen de entrada en escala de grises.
        mostrar_resultados (bool): Si es True, muestra las imágenes y los histogramas (por defecto True).

    Returns:
        numpy.ndarray: Imagen ecualizada.
    """
    # Ecualización del canal seleccionado
    canal_ecualizado = cv2.equalizeHist(imagen_wavelet.astype('uint8'))

    if mostrar_resultados:
        # Gráficos
        plt.figure(figsize=(15, 10))

        # Imagen del canal seleccionado
        plt.subplot(2, 2, 1)
        plt.imshow(imagen_wavelet, cmap='gray')
        plt.title("Canal Seleccionado - Original")
        plt.axis('off')

        # Histograma de la imagen original
        plt.subplot(2, 2, 2)
        plt.hist(imagen_wavelet.ravel(), bins=256, range=(0, 255), color='blue')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.title("Histograma - Canal Original")
        plt.xlabel("Intensidad de los píxeles")
        plt.ylabel("Cantidad de píxeles")

        # Imagen del canal ecualizado
        plt.subplot(2, 2, 3)
        plt.imshow(canal_ecualizado, cmap='gray')
        plt.title("Canal Seleccionado - Ecualizado")
        plt.axis('off')

        # Histograma de la imagen ecualizada
        plt.subplot(2, 2, 4)
        plt.hist(canal_ecualizado.ravel(), bins=256, range=(0, 255), color='blue')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.title("Histograma - Canal Ecualizado")
        plt.xlabel("Intensidad de los píxeles")
        plt.ylabel("Cantidad de píxeles")

        plt.tight_layout()
        plt.show()

    return canal_ecualizado