# main.py
import cv2
import numpy as np
print("¡Entorno configurado correctamente!")

from src.carga_imagenes import cargar_imagenes

def main():
    imagenes = cargar_imagenes()
    print(f"Se cargaron {len(imagenes)} imágenes:")
    for nombre in imagenes.keys():
        print(f"- {nombre}")

if __name__ == "__main__":
    main()
