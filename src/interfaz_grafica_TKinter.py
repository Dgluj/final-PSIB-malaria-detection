import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.ttk import Progressbar
from PIL import Image, ImageTk
import cv2
import pandas as pd
import numpy as np
import threading
from src.procesamiento_interfaz import procesar_imagen_con_modelo


# Clase:
class InterfazAnalisisCélulas:
    def __init__(self, root, modelo, accuracy):
        self.root = root
        self.mejor_modelo = modelo  # Modelo previamente cargado
        self.accuracy = accuracy   # Accuracy del modelo
        self.imagen_cargada = None
        self.imagen_procesada = None
        self.predicciones = None

        # Crear la interfaz
        self.root.title("Análisis de Imágenes de Células")
    
    # Componentes de la interfaz
        # Botón para cargar la imagen
        self.boton_cargar = tk.Button(root, text="Cargar Imagen", command=self.cargar_imagen)
        self.boton_cargar.pack()

        # Canvas para mostrar la imagen
        self.canvas = tk.Canvas(root, width=500, height=500, bg="white")
        self.canvas.pack()

        # Botón para procesar la imagen
        self.boton_procesar = tk.Button(root, text="Analizar Imagen", command=self.analizar_imagen)
        self.boton_procesar.pack()

        # Barra de progreso
        self.progreso = Progressbar(root, orient=tk.HORIZONTAL, length=300, mode="indeterminate")

        # Etiqueta para mostrar la accuracy
        self.etiqueta_accuracy = tk.Label(root, text=f"Exactitud del Modelo: {self.accuracy*100:.2f}%")
        self.etiqueta_accuracy.pack()

        # Botones para guardar resultados
        self.boton_guardar_imagen = tk.Button(root, text="Guardar Imagen Procesada", command=self.guardar_imagen, state=tk.DISABLED)
        self.boton_guardar_imagen.pack()

        self.boton_guardar_csv = tk.Button(root, text="Guardar Resultados (.csv)", command=self.guardar_csv, state=tk.DISABLED)
        self.boton_guardar_csv.pack()

    # Métodos de la clase 
    def cargar_imagen(self):
        # Diálogo para seleccionar una imagen
        ruta_imagen = filedialog.askopenfilename(filetypes=[("Archivos de imagen", "*.jpg;*.png;*.jpeg")])
        if not ruta_imagen:
            return

        # Leer y mostrar la imagen
        self.imagen_cargada = cv2.imread(ruta_imagen)
        self.mostrar_imagen(self.imagen_cargada)

    def mostrar_imagen(self, img):
        # Convertir la imagen a formato compatible con Tkinter
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_pil = img_pil.resize((500, 500))
        self.imagen_tk = ImageTk.PhotoImage(img_pil)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.imagen_tk)

    def analizar_imagen(self):
        if self.imagen_cargada is None:
            messagebox.showerror("Error", "Primero debe cargar una imagen.")
            return

        # Deshabilitar botones y mostrar barra de progreso
        self.boton_cargar.config(state=tk.DISABLED)
        self.boton_procesar.config(state=tk.DISABLED)
        self.progreso.pack()
        self.progreso.start()

        # Procesar la imagen en un hilo separado para evitar bloquear la interfaz
        threading.Thread(target=self.procesar_imagen).start()

    def procesar_imagen(self):
        try:
            # Llamar a la función de procesamiento
            img_bounding_boxes, predicciones = procesar_imagen_con_modelo(self.imagen_cargada, self.mejor_modelo)

            # Actualizar resultados
            self.imagen_procesada = img_bounding_boxes
            self.predicciones = predicciones

            # Mostrar resultados
            self.mostrar_imagen(self.imagen_procesada)

            # Habilitar botones para guardar
            self.boton_guardar_imagen.config(state=tk.NORMAL)
            self.boton_guardar_csv.config(state=tk.NORMAL)

        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un error durante el procesamiento: {e}")

        finally:
            # Detener barra de progreso y habilitar botones
            self.progreso.stop()
            self.progreso.pack_forget()
            self.boton_cargar.config(state=tk.NORMAL)
            self.boton_procesar.config(state=tk.NORMAL)

    def guardar_imagen(self):
        if self.imagen_procesada is None:
            messagebox.showerror("Error", "No hay imagen procesada para guardar.")
            return

        # Diálogo para guardar la imagen
        ruta_guardado = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if not ruta_guardado:
            return

        # Guardar la imagen procesada
        img_rgb = cv2.cvtColor(self.imagen_procesada, cv2.COLOR_BGR2RGB)
        Image.fromarray(img_rgb).save(ruta_guardado)
        messagebox.showinfo("Éxito", "Imagen guardada exitosamente.")

    def guardar_csv(self):
        if self.predicciones is None:
            messagebox.showerror("Error", "No hay resultados para guardar.")
            return

        # Diálogo para guardar el archivo CSV
        ruta_guardado = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not ruta_guardado:
            return

        # Crear el DataFrame y guardar
        df = pd.DataFrame(self.predicciones, columns=["X", "Y", "Width", "Height", "Clasificación"])
        df.to_csv(ruta_guardado, index=False)
        messagebox.showinfo("Éxito", "Resultados guardados exitosamente.")

# Función principal para ejecutar la interfaz
if __name__ == "__main__":
    # Cargar el modelo previamente guardado
    modelo = np.load("mejor_modelo.npy", allow_pickle=True) # El parámetro allow_pickle=True es necesario si tu objeto (el modelo) es un objeto complejo, como un objeto de un modelo de aprendizaje automático que no es un simple array de números. 
    accuracy = 0.96 # Ejemplo de precisión del modelo
    root = tk.Tk()
    app = InterfazAnalisisCélulas(root, modelo, accuracy)
    root.mainloop()