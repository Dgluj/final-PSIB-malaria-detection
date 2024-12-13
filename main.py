import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.carga_imagenes import cargar_imagenes
from src.preprocesamiento import reducir_ruido, separar_canales, seleccionar_canal_mayor_contraste, aplicar_fft, aplicar_wavelet, aplicar_ecualizado
from src.segmentacion import segmentar_kmeans_y_umbral, aplicar_floodfill, filtrar_celulas_infectadas 
from src.watershed import binarizar, aplicar_watershed, aplicar_dilatacion_y_erosion, dibujar_bounding_boxes, procesar_recortes_y_watershed, segmentar_recortes

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
        canal_seleccionado = seleccionar_canal_mayor_contraste(canal_rojo, canal_verde, canal_azul)
        
        # Analizar el espectro de frecuencias del canal seleccionado
        aplicar_fft(canal_seleccionado)

        # Aplicar la Transformada Wavelet
        imagen_wavelet = aplicar_wavelet(canal_seleccionado)

        # Aplicar ecualizado
        canal_ecualizado = aplicar_ecualizado(imagen_wavelet)

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

        # Binarizar el canal seleccionado preprocesado (imagen_wavelet)
        img_binarizada = binarizar(canal_ecualizado, 100)

        img_cerrada = aplicar_dilatacion_y_erosion(img_binarizada)

        # Aplicar la transformada de la distancia y Watershed a la imagen completa binaria (img_binarizada)
        img_ws = aplicar_watershed(img_cerrada,150)

        # Detectar contornos
        contornos, _ = cv2.findContours(img_ws, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Llamar a la función para dibujar los bounding boxes
        img_bounding_boxes = dibujar_bounding_boxes(imagen_wavelet, contornos, color=(0, 255, 0), grosor=1)

        # Llamar al pipeline
        img_bounding_boxes_final = segmentar_recortes(imagen_wavelet, img_ws, contornos, level=30, umbral_area_minima=800)

        # Mostrar la comparación entre los bounding boxes previos y los nuevos
        fig, axs = plt.subplots(1, 2, figsize=(15, 7))

        # Mostrar imagen con bounding boxes previos
        axs[0].imshow(cv2.cvtColor(cv2.cvtColor(img_bounding_boxes, cv2.COLOR_BGR2RGB), cv2.COLOR_BGR2RGB))
        axs[0].set_title("Bounding Boxes Previos")
        axs[0].axis('off')

        # Mostrar imagen con nuevos bounding boxes
        axs[1].imshow(cv2.cvtColor(img_bounding_boxes_final, cv2.COLOR_BGR2RGB))
        axs[1].set_title("Nuevos Bounding Boxes")
        axs[1].axis('off')

        # Ajustar el layout y mostrar
        plt.tight_layout()
        plt.show()
if __name__ == "__main__":
    main()