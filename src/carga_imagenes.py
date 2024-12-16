import os
import cv2
# hola
def cargar_imagenes():
    """
    Carga todas las imágenes de la carpeta 'data/' relativa al repositorio.

    Returns:
        dict: Diccionario con nombres de archivos como claves y matrices de imágenes como valores.
    """
    directorio = os.path.join(os.path.dirname(__file__), "../data")
    extensiones_validas = ('.png', '.jpg', '.jpeg')
    imagenes = {}
    for archivo in os.listdir(directorio):
        if archivo.lower().endswith(extensiones_validas):
            ruta = os.path.join(directorio, archivo)
            img = cv2.imread(ruta)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imagenes[archivo] = img_rgb
    return imagenes