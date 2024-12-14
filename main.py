import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from src.carga_imagenes import cargar_imagenes
from src.preprocesamiento import reducir_ruido, separar_canales, seleccionar_canal_mayor_contraste, aplicar_fft, aplicar_wavelet, aplicar_ecualizado, binarizar_con_kmeans, aplicar_filtro_mediana, aplicar_operaciones_morfologicas, rellenar_celulas 
from src.segmentacion import segmentar_kmeans_y_umbral, aplicar_floodfill, filtrar_celulas_infectadas, binarizar, binarizar_auto, aplicar_watershed, aplicar_dilatacion_y_erosion, dibujar_bounding_boxes, procesar_recortes_y_watershed, segmentar_recortes
from src.extraccion_de_caracteristicas import construir_base_datos, clasificacion_final
from src.utils import dibujar_bounding_boxes_en_identificadas

def main():
    # Cargar una imagen a través de la futura interfaz gráfica (placeholder):
    # Por ahora, llamamos a una función que podría simular la carga de una única imagen desde el usuario.
    # En caso de no tener GUI implementada aún, podemos cargar una imagen específica de data/.
    # img, nombre = cargar_imagen_desde_GUI()  # Cuando esté la GUI
    # Por ahora, cargamos una imagen directamente:
    imagenes = cargar_imagenes()
    print(f"Se cargaron {len(imagenes)} imágenes:")
    for nombre in imagenes.keys():
        print(f"- {nombre}")
    df_final = pd.DataFrame()

    for nombre, img in imagenes.items(): 
        img_filtered, img_rgb = reducir_ruido(img) # Reducir el ruido y convertir la imagen a RGB
        canal_rojo, canal_verde, canal_azul = separar_canales(img_rgb) # Separar los canales de la imagen filtrada
        canal_seleccionado = seleccionar_canal_mayor_contraste(canal_rojo, canal_verde, canal_azul)

        img_binarizada = binarizar_con_kmeans(canal_seleccionado) # Binarizar la imagen del canal de mayor contraste usando KMeans
        img_mediana = aplicar_filtro_mediana(img_binarizada) # Aplicar filtro de mediana para suavizar la imagen binarizada
        img_morfo = aplicar_operaciones_morfologicas(img_mediana) # Aplicar operaciones morfológicas (dilatación y erosión)

        # Morfología cierre morfológico directo
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))  # Ajusta el tamaño del kernel según sea necesario
        # img_closing = cv2.morphologyEx(img_binarizada_mediana, cv2.MORPH_CLOSE, kernel)

        # Rellenar las células
        img_rellena = rellenar_celulas(img_morfo)

        # Aplicar ecualizado
        # canal_ecualizado = aplicar_ecualizado(canal_seleccionado, nombre)

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
        #img_binarizada = binarizar(canal_ecualizado, 60)
        #img_binarizada = binarizar_auto(canal_ecualizado) #checkinggggggggggg
        #img_binarizada = img_binarizada.astype(np.uint8)

        #img_dilatada, img_cerrada = aplicar_dilatacion_y_erosion(img_binarizada)

        # Aplicar la transformada de la distancia y Watershed a la imagen completa binaria (img_binarizada)
        # img_ws, resultados_intermedios = aplicar_watershed(img_cerrada, level=40) # No achicar mas xq se caga
        img_ws, resultados_intermedios = aplicar_watershed(img_rellena, level=40) # No achicar mas xq se caga

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
        img_bounding_boxes = dibujar_bounding_boxes(canal_seleccionado, contornos, color=(0, 255, 0), grosor=1, umbral_area_min=5000)

        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(img_bounding_boxes, cv2.COLOR_BGR2RGB))
        plt.title("Bounding Boxes" + nombre)
        plt.axis("off")
        plt.show()

        # # Llamar al pipeline de segmentación y bounding boxes de recortes
        # img_bounding_boxes_final = segmentar_recortes(canal_seleccionado, img_ws, contornos, level=30, umbral_area_max=20000, umbral_area_min=5000)

        # # Mostrar la comparación entre los bounding boxes previos y los nuevos
        # fig, axs = plt.subplots(1, 2, figsize=(15, 7))

        # # Mostrar imagen con bounding boxes previos
        # axs[0].imshow(cv2.cvtColor(cv2.cvtColor(img_bounding_boxes, cv2.COLOR_BGR2RGB), cv2.COLOR_BGR2RGB))
        # axs[0].set_title("Bounding Boxes Previos" + nombre)
        # axs[0].axis('off')

        # # Mostrar imagen con nuevos bounding boxes
        # axs[1].imshow(cv2.cvtColor(img_bounding_boxes_final, cv2.COLOR_BGR2RGB))
        # axs[1].set_title("Nuevos Bounding Boxes" + nombre)
        # axs[1].axis('off')

        # # Ajustar el layout y mostrar
        # plt.tight_layout()
        # plt.show()

        # Construir base de datos completa para cada imagen
        df = construir_base_datos(canal_seleccionado, contornos)

        # Mostrar DataFrame
        print("Base de datos de características:", nombre)
        print(df)

        # # Detenerse después de procesar la primera imagen
        # if nombre == "5.png":
        #     break

        # Definir los umbrales
        umbrales = {
            "Área": 12800,
            "Perímetro": 400,
            "Circularidad": 0.88,
            "Contraste": 30,
            "Energía": 0.12,
            "Homogeneidad": 0.6,
            "Cluster Shade": 3e+11,
            "Cluster Prominencia": 3e+13,
            "Correlación Haralick": 0.996,
            "Entropía": 9.5
        }

        # Obtener el DataFrame final balanceado
        df_infectada_sana = clasificacion_final(df, umbrales)
        
        # Agregar la columna "Imagen" con el nombre de la imagen
        df_infectada_sana["Imagen"] = nombre

        # Reorganizar las columnas para que "Imagen" sea la primera
        columnas = ["Imagen"] + [col for col in df_infectada_sana.columns if col != "Imagen"]
        df_infectada_sana = df_infectada_sana[columnas]
        print(df_infectada_sana)

        pd.set_option('display.max_columns', None)

        # Dibujar los bounding boxes con los textos en la imagen
        img_con_bboxes = dibujar_bounding_boxes_en_identificadas(img_rgb, df_infectada_sana)
        
        # Mostrar la imagen con los bounding boxes finales
        cv2.imshow("Bounding Boxes clasificados para imagen: {nombre}", img_con_bboxes)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Concatenar el DataFrame actual al DataFrame final
        df_final = pd.concat([df_final, df_infectada_sana], ignore_index=True)
    
    # Imprimir el DataFrame final
    print("DataFrame Final:")
    print(df_final)

if __name__ == "__main__":
    main()