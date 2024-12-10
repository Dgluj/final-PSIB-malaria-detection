import cv2
import numpy as np
from src.carga_imagenes import cargar_imagenes
from src.carga_imagenes import cargar_imagenes
from src.preprocesamiento import reducir_ruido, separar_canales, seleccionar_canal_menor_contraste, aplicar_wavelet
import matplotlib.pyplot as plt

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



if __name__ == "__main__":
    main()
