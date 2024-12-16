import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.ttk import Progressbar
from PIL import Image, ImageTk
import joblib
import cv2
import pandas as pd
import numpy as np
import threading

# Importar las funciones necesarias desde los módulos dentro de src
from src.extraccion_de_caracteristicas import contar_celulas
from src.preprocesamiento import (
    reducir_ruido, 
    separar_canales, 
    seleccionar_canal_mayor_contraste, 
    binarizar_con_kmeans, 
    aplicar_filtro_mediana, 
    aplicar_operaciones_morfologicas, 
    rellenar_celulas
)
from src.segmentacion import aplicar_watershed 
from src.utils import dibujar_bounding_boxes_en_identificadas
from src.extraccion_de_caracteristicas import construir_base_datos

def procesar_imagen_con_modelo(img, modelo):
    """
    Procesa una imagen, aplica el modelo entrenado y clasifica células como infectadas o no.

    Args:
        img (numpy.ndarray): Imagen de entrada.
        modelo (sklearn model): Modelo entrenado para clasificación.

    Returns:
        tuple: Imagen con bounding boxes y etiquetas, DataFrame con características y predicciones.
    """
    # Validación inicial: la imagen debe ser un array de numpy
    if not isinstance(img, np.ndarray):
        raise ValueError("La imagen proporcionada no es un array de NumPy válido.")

    # Paso 1: Preprocesamiento
    _, img_filtered_rgb = reducir_ruido(img)
    canal_rojo, canal_verde, canal_azul = separar_canales(img_filtered_rgb)
    canal_seleccionado = seleccionar_canal_mayor_contraste(canal_rojo, canal_verde, canal_azul)
    
    # Binarización y procesamiento morfológico
    img_binarizada = binarizar_con_kmeans(canal_seleccionado)
    img_mediana = aplicar_filtro_mediana(img_binarizada)
    img_morfo = aplicar_operaciones_morfologicas(img_mediana)
    img_rellena = rellenar_celulas(img_morfo)

    # Segmentación con Watershed
    img_ws, _ = aplicar_watershed(img_rellena, level=40)

    # Paso 2: Detección de contornos y extracción de características
    contornos, _ = cv2.findContours(img_ws, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    df = construir_base_datos(canal_seleccionado, contornos,4500)

    # Paso 3: Predicción usando el modelo entrenado
    X = df.drop(columns=["ID"]).copy()  # Eliminar columnas innecesarias para la predicción
    X = X.values  # Convertir a formato de NumPy
    df["Infectada"] = modelo.predict(X)  # Predecir y agregar las etiquetas al DataFrame

    # Paso 4: Dibujar los bounding boxes en la imagen
    img_con_bboxes = dibujar_bounding_boxes_en_identificadas(img_filtered_rgb, df)

    return img_con_bboxes, df

# Clase de la interfaz
class InterfazAnalisisCélulas:
    def __init__(self, root, modelo, accuracy):
        self.root = root
        self.mejor_modelo = modelo
        self.accuracy = accuracy
        
        # Atributos para imágenes y resultados
        self.img_original = None
        self.img_procesada = None
        self.predicciones = None

        self.root.title("Análisis de Imágenes de Células")
        self.crear_interfaz()

    def crear_interfaz(self):
        # Botones y componentes
        tk.Button(self.root, text="Cargar Imagen", command=self.cargar_imagen).pack()
        self.canvas = tk.Canvas(self.root, width=700, height=500, bg="white")
        self.canvas.pack()
        # tk.Button(self.root, text="Analizar Imagen", command=self.analizar_imagen).pack()
        self.boton_analizar = tk.Button(self.root, text="Analizar Imagen", command=self.analizar_imagen)
        self.boton_analizar.pack()
        self.progreso = Progressbar(self.root, orient=tk.HORIZONTAL, length=300, mode="indeterminate")
        self.etiqueta_accuracy = tk.Label(self.root, text=f"Exactitud del Modelo: {self.accuracy*100:.2f}%")
        self.boton_guardar_imagen = tk.Button(self.root, text="Guardar Imagen Procesada", command=self.guardar_imagen, state=tk.DISABLED)
        self.boton_guardar_imagen.pack()
        self.boton_guardar_csv = tk.Button(self.root, text="Guardar Resultados (.csv)", command=self.guardar_csv, state=tk.DISABLED)
        self.boton_guardar_csv.pack()

    def cargar_imagen(self):
        ruta_imagen = filedialog.askopenfilename(filetypes=[("Archivos de imagen", "*.jpg;*.png;*.jpeg")])
        if not ruta_imagen:
            return

        # Leer y validar la imagen
        self.img_original = cv2.imread(ruta_imagen)
        if self.img_original is None:
            messagebox.showerror("Error", "No se pudo cargar la imagen. Verifique el formato o la ruta.")
            return

        # Convertir explícitamente a NumPy array si es necesario
        self.img_original = np.asarray(self.img_original, dtype=np.uint8)
        print("Tipo de imagen después de cargar:", type(self.img_original))

        # Convertir a RGB para visualización
        self.img_original = cv2.cvtColor(self.img_original, cv2.COLOR_BGR2RGB)
        self.mostrar_imagen(self.img_original)

        # Desbloquear el botón de analizar y reiniciar el estado
        self.boton_analizar.config(state=tk.NORMAL)
        self.imagen_procesada = None  # Reiniciar la imagen procesada
        self.predicciones = None      # Reiniciar predicciones

    def mostrar_imagen(self, img, texto=None):
        """
        Muestra una imagen en el canvas con texto opcional.
        """
        img_copy = img.copy()  # Copia para evitar modificaciones accidentales
        if texto:
            cv2.putText(img_copy, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        img_pil = Image.fromarray(img_copy).resize((700, 500))
        self.imagen_tk = ImageTk.PhotoImage(img_pil)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.imagen_tk)

    def analizar_imagen(self):
        if self.img_original is None:
            messagebox.showerror("Error", "Primero debe cargar una imagen.")
            return

        self.progreso.pack()
        self.progreso.start()
        threading.Thread(target=self.procesar_imagen).start()

    def procesar_imagen(self):
        try:
            # Procesar imagen con modelo
            self.img_procesada, df = procesar_imagen_con_modelo(self.img_original, self.mejor_modelo)

            # Contar células y mostrar resultados
            num_infectadas, num_sanas = contar_celulas(df)
            texto = f"Infectadas: {num_infectadas} | Sanas: {num_sanas}"

            # Mostrar imagen procesada
            self.mostrar_imagen(self.img_procesada, texto)

            # Guardar resultados
            self.predicciones = df
            self.boton_guardar_imagen.config(state=tk.NORMAL)
            self.boton_guardar_csv.config(state=tk.NORMAL)

            self.boton_analizar.config(state=tk.DISABLED)  # Aquí deshabilitas el botón

        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un error durante el procesamiento: {e}")
        finally:
            self.progreso.stop()
            self.progreso.pack_forget()
            self.etiqueta_accuracy.config(text=f"Exactitud del Modelo: {self.accuracy*100:.2f}%") #?
            self.etiqueta_accuracy.pack() #?

    def guardar_imagen(self):
        if self.img_procesada is None:
            messagebox.showerror("Error", "No hay imagen procesada para guardar.")
            return

        ruta_guardado = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", ".png"), ("JPEG", ".jpg")])
        if not ruta_guardado:
            return

        Image.fromarray(self.img_procesada).save(ruta_guardado)
        messagebox.showinfo("Éxito", "Imagen guardada exitosamente.")

    def guardar_csv(self):
        if self.predicciones is None:
            messagebox.showerror("Error", "No hay resultados para guardar.")
            return

        ruta_guardado = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not ruta_guardado:
            return

        self.predicciones.to_csv(ruta_guardado, index=False)
        messagebox.showinfo("Éxito", "Resultados guardados exitosamente.")

# Función principal
if __name__ == "__main__":
    modelo = joblib.load("mejor_modelo.pkl")
    root = tk.Tk()
    app = InterfazAnalisisCélulas(root, modelo, 0.91)
    root.mainloop()