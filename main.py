import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.carga_imagenes import cargar_imagenes
from src.preprocesamiento import reducir_ruido, separar_canales, seleccionar_canal_menor_contraste, aplicar_fft, aplicar_wavelet
from src.segmentacion import segmentar_kmeans_y_umbral, aplicar_floodfill, filtrar_celulas_infectadas 
from src.watershed import binarizar, aplicar_watershed

def main():
    imagenes = cargar_imagenes()
#    print(f"Se cargaron {len(imagenes)} imágenes:")
#    for nombre in imagenes.keys():
#        print(f"- {nombre}")

    for nombre, img in imagenes.items():
        # Reducir ruido
        img_denoised = reducir_ruido(img)
        
        # Separar canales y seleccionar el de menor contraste
        canal_rojo, canal_verde, canal_azul = separar_canales(img_denoised)
        canal_seleccionado = seleccionar_canal_menor_contraste(canal_rojo, canal_verde, canal_azul)
        
        # Analizar el espectro de frecuencias del canal seleccionado
        aplicar_fft(canal_seleccionado)

        # Aplicar la Transformada Wavelet
        imagen_wavelet = aplicar_wavelet(canal_seleccionado)

        # # Mostrar el resultado
        # plt.figure(figsize=(15, 5))
        # plt.subplot(1, 3, 1)
        # plt.imshow(img)
        # plt.title(f"Original: {nombre}")
        # plt.axis("off")

        # plt.subplot(1, 3, 2)
        # plt.imshow(canal_seleccionado, cmap="gray")
        # plt.title("Canal Seleccionado")
        # plt.axis("off")

        # plt.subplot(1, 3, 3)
        # plt.imshow(imagen_wavelet, cmap="gray")
        # plt.title("Wavelet (LL)")
        # plt.axis("off")

        # plt.tight_layout()
        # plt.show()

        # Aplicar KMeans y Umbralización
        img_segmentada = segmentar_kmeans_y_umbral(imagen_wavelet, n_clusters=2, umbral=80)

        # *** Flood Fill directo en el main ***
        # Crear máscara
        mask = np.zeros((img_segmentada.shape[0] + 2, img_segmentada.shape[1] + 2), dtype=np.uint8)

        # Píxeles blancos (células enfermas) a 1
        mask[1:-1, 1:-1][img_segmentada == 255] = 1

        # Encontrar contornos de las células enfermas
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Rellenar las áreas conectadas con floodFill
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.floodFill(img_segmentada, mask, (x + w // 2, y + h // 2), 255)  # Rellenar con blanco (255)

        # Binarizar la imagen
        _, img_infectadas = cv2.threshold(img_segmentada, 129, 255, cv2.THRESH_BINARY)

        # Encontrar los contornos en la imagen binarizada
        cnts = cv2.findContours(img_infectadas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        # Crear una imagen en blanco para dibujar los contornos filtrados
        img_infectadas = np.zeros_like(img_infectadas)

        # Iterar sobre cada contorno y verificar su área
        for c in cnts:
            area = cv2.contourArea(c)
            if area > 2000:  # Mantener solo los objetos cuya área es mayor a 3000 píxeles
                # Dibujar los contornos en la imagen filtrada
                cv2.drawContours(img_infectadas, [c], -1, 255, thickness=cv2.FILLED)

        # # Crear máscara
        # mask = np.zeros((img_segmentada.shape[0] + 2, img_segmentada.shape[1] + 2), dtype=np.uint8)

        # # Detectar contornos de las áreas blancas
        # contours, _ = cv2.findContours((img_segmentada == 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # if contours:
        #     for contour in contours:
        #         x, y, w, h = cv2.boundingRect(contour)
        #         cx, cy = x + w // 2, y + h // 2  # Centro aproximado
        #         if img_segmentada[cy, cx] == 255:  # Validar que sea blanco
        #             cv2.floodFill(img_segmentada, mask, (cx, cy), 255)

        # # Filtrar las células infectadas basándonos en el área mínima
        # img_filtrada = filtrar_celulas_infectadas(img_segmentada, area_minima=2000)

        # Mostrar resultados
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(img, cmap="gray")
        plt.title(f"Original: {nombre}")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(img_segmentada, cmap="gray")
        plt.title("Post-KMeans y Umbral")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(img_infectadas, cmap="gray")
        plt.title("Células Infectadas Filtradas")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

        # Binarizar el canal seleccionado preprocesado (imagen_wavelet)
        img_binarizada = binarizar(imagen_wavelet, 215)

        # Aplicar la transformada de la distancia y Watershed a la imagen completa binaria (img_binarizada)
        img_completa_ws = aplicar_watershed(img_binarizada,80)

        # Aplicar la transformada de la distancia y Watershed a la imagen de potenciales células infectadas (img_infectadas)
        img_infectadas_ws = aplicar_watershed(img_infectadas, 20)

if __name__ == "__main__":
    main()