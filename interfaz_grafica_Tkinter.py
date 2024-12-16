import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.ttk import Progressbar
from PIL import Image, ImageTk
import joblib
import cv2
import pandas as pd
import numpy as np
import threading

# Importar funciones desde `src`
from src.procesamiento_interfaz import procesar_imagen_con_modelo
from src.extraccion_de_caracteristicas import contar_celulas

# Clase de la interfaz
class InterfazAnalisisCélulas:
    def __init__(self, root, modelo, accuracy):
        self.root = root
        self.mejor_modelo = modelo
        self.accuracy = accuracy
        self.imagen_cargada = None
        self.imagen_procesada = None
        self.predicciones = None

        self.root.title("Análisis de Imágenes de Células")
        self.crear_interfaz()

    def crear_interfaz(self):
        # Botones y componentes
        tk.Button(self.root, text="Cargar Imagen", command=self.cargar_imagen).pack()
        self.canvas = tk.Canvas(self.root, width=700, height=500, bg="white")
        self.canvas.pack()
        tk.Button(self.root, text="Analizar Imagen", command=self.analizar_imagen).pack()
        self.progreso = Progressbar(self.root, orient=tk.HORIZONTAL, length=300, mode="indeterminate")
        self.etiqueta_accuracy = tk.Label(self.root, text=f"Exactitud del Modelo: {self.accuracy*100:.2f}%")
        self.etiqueta_accuracy.pack()

        # self.boton_guardar_imagen = tk.Button(self.root, text="Guardar Imagen Procesada", command=self.guardar_imagen, state=tk.DISABLED)
        # self.boton_guardar_imagen.pack()

        # self.boton_guardar_csv = tk.Button(self.root, text="Guardar Resultados (.csv)", command=self.guardar_csv, state=tk.DISABLED)
        # self.boton_guardar_csv.pack()


    # def cargar_imagen(self):
    #     ruta_imagen = filedialog.askopenfilename(filetypes=[("Archivos de imagen", "*.jpg;*.png;*.jpeg")])
    #     if not ruta_imagen:
    #         return
    #     self.imagen_cargada = cv2.imread(ruta_imagen)
    #     if self.imagen_cargada is None:
    #         messagebox.showerror("Error", "Error al cargar la imagen.")
    #     else:
    #         self.mostrar_imagen(self.imagen_cargada)
    def cargar_imagen(self):
        ruta_imagen = filedialog.askopenfilename(filetypes=[("Archivos de imagen", "*.jpg;*.png;*.jpeg")])
        if not ruta_imagen:
            return

        # Leer la imagen y validar
        self.imagen_cargada = cv2.imread(ruta_imagen)
        if self.imagen_cargada is None:
            messagebox.showerror("Error", "No se pudo cargar la imagen. Verifique el formato o la ruta.")
            return

        # Verificar tipo de dato
        if not isinstance(self.imagen_cargada, np.ndarray):
            messagebox.showerror("Error", "La imagen cargada no es un array válido.")
            return

        # Convertir a RGB para visualización
        self.imagen_cargada = cv2.cvtColor(self.imagen_cargada, cv2.COLOR_BGR2RGB)
        self.mostrar_imagen(self.imagen_cargada)

    def mostrar_imagen(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb).resize((700, 500))
        self.imagen_tk = ImageTk.PhotoImage(img_pil)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.imagen_tk)

    def analizar_imagen(self):
        if self.imagen_cargada is None:
            messagebox.showerror("Error", "Primero debe cargar una imagen.")
            return
        self.progreso.pack()
        self.progreso.start()
        threading.Thread(target=self.procesar_imagen).start()

    def procesar_imagen(self):
        try:
            img_bboxes, _, df = procesar_imagen_con_modelo(self.imagen_cargada, self.mejor_modelo)
            num_infectadas, num_sanas = contar_celulas(df)
            texto = f"Infectadas: {num_infectadas} | Sanas: {num_sanas}"
            self.mostrar_imagen(img_bboxes)
            messagebox.showinfo("Conteo de Células", texto)
        except Exception as e:
            messagebox.showerror("Error", f"Error durante el procesamiento: {e}")
        finally:
            self.progreso.stop()
            self.progreso.pack_forget()

# Función principal
if __name__ == "__main__":
    modelo = joblib.load("mejor_modelo.pkl")
    root = tk.Tk()
    app = InterfazAnalisisCélulas(root, modelo, 0.96)
    root.mainloop()