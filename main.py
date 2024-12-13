import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.carga_imagenes import cargar_imagenes
from src.preprocesamiento import reducir_ruido, separar_canales, seleccionar_canal_mayor_contraste, aplicar_fft, aplicar_wavelet, aplicar_ecualizado
from src.segmentacion import segmentar_kmeans_y_umbral, aplicar_floodfill, filtrar_celulas_infectadas 
from src.watershed import binarizar, binarizar_auto, aplicar_watershed, aplicar_dilatacion_y_erosion, dibujar_bounding_boxes, procesar_recortes_y_watershed, segmentar_recortes
from src.extraccion_de_caracteristicas import construir_base_datos

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
        
        # # Analizar el espectro de frecuencias del canal seleccionado
        # aplicar_fft(canal_seleccionado)

        # Aplicar ecualizado
        canal_ecualizado = aplicar_ecualizado(canal_seleccionado)

        #  # Aplicar la Transformada Wavelet
        # canal_ecualizado_wavelet = aplicar_wavelet(canal_ecualizado)

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
        #img_binarizada = binarizar(canal_ecualizado, 100)
        img_binarizada = binarizar_auto(canal_ecualizado) #checkinggggggggggg
        img_binarizada = img_binarizada.astype(np.uint8)

        img_dilatada, img_cerrada = aplicar_dilatacion_y_erosion(img_binarizada)

        # Aplicar la transformada de la distancia y Watershed a la imagen completa binaria (img_binarizada)
        img_ws, resultados_intermedios = aplicar_watershed(img_cerrada, level=40) # No achicar mas xq se caga

        # Mostrar los resultados intermedios
        fig, ax = plt.subplots(2, 2, figsize=(10, 10))

        ax[0, 0].imshow(resultados_intermedios['dist_array'], cmap='gray')
        ax[0, 0].set_title('Transformada de la distancia', fontsize=15)

        ax[0, 1].imshow(resultados_intermedios['umbral_array'], cmap='gray')
        ax[0, 1].set_title('Imagen binaria', fontsize=15)

        ax[1, 0].imshow(resultados_intermedios['etiquetas_array'], cmap='gray')
        ax[1, 0].set_title('Etiquetas Watershed', fontsize=15)

        ax[1, 1].imshow(img_ws, cmap='gray')
        ax[1, 1].set_title('Resultado Watershed', fontsize=15)

        plt.tight_layout()
        plt.show()

        # Detectar contornos
        contornos, _ = cv2.findContours(img_ws, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Llamar a la función para dibujar los bounding boxes
        img_bounding_boxes = dibujar_bounding_boxes(canal_seleccionado, contornos, color=(0, 255, 0), grosor=1, umbral_area_minima=500)

        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(img_bounding_boxes, cv2.COLOR_BGR2RGB))
        plt.title("Bounding Boxes")
        plt.axis("off")
        plt.show()

        # # Llamar al pipeline
        # img_bounding_boxes_final = segmentar_recortes(canal_seleccionado, img_ws, contornos, level=30, umbral_area_minima=1000)

        # # Mostrar la comparación entre los bounding boxes previos y los nuevos
        # fig, axs = plt.subplots(1, 2, figsize=(15, 7))

        # # Mostrar imagen con bounding boxes previos
        # axs[0].imshow(cv2.cvtColor(cv2.cvtColor(img_bounding_boxes, cv2.COLOR_BGR2RGB), cv2.COLOR_BGR2RGB))
        # axs[0].set_title("Bounding Boxes Previos")
        # axs[0].axis('off')

        # # Mostrar imagen con nuevos bounding boxes
        # axs[1].imshow(cv2.cvtColor(img_bounding_boxes_final, cv2.COLOR_BGR2RGB))
        # axs[1].set_title("Nuevos Bounding Boxes")
        # axs[1].axis('off')

        # # Ajustar el layout y mostrar
        # plt.tight_layout()
        # plt.show()

        # Construir base de datos
        df = construir_base_datos(canal_seleccionado, contornos)

        # Mostrar imagen con bounding boxes
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(img_bounding_boxes, cv2.COLOR_BGR2RGB))
        plt.title("Bounding Boxes")
        plt.axis("off")
        plt.show()

        # Mostrar DataFrame
        print("Base de datos de características:")
        print(df)

        # # Construir base de datos
        # df = construir_base_datos(canal_seleccionado, contornos)

        # # Mostrar imagen con bounding boxes
        # plt.figure(figsize=(10, 10))
        # plt.imshow(cv2.cvtColor(img_bounding_boxes, cv2.COLOR_BGR2RGB))
        # plt.title("Bounding Boxes")
        # plt.axis("off")
        # plt.show()

        # # Mostrar DataFrame
        # print("Base de datos de características:")
        # print(df)

        # Detenerse después de procesar la primera imagen
        break

if __name__ == "__main__":
    main()