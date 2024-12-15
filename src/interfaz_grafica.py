import sys
import cv2
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from src.procesamiento_interfaz import procesar_imagen_con_modelo

"""
Widgets: Para elementos gráficos (botones, etiquetas, diálogos, etc.).
Gui: Para manejar gráficos como imágenes (QPixmap).
Core: Incluye constantes como Qt.AlignCenter.
"""

# Importaciones necesarias
from src.preprocesamiento import (
    reducir_ruido, separar_canales, seleccionar_canal_mayor_contraste, binarizar_con_kmeans,
    aplicar_filtro_mediana, aplicar_operaciones_morfologicas, rellenar_celulas
)
from src.segmentacion import aplicar_watershed, dibujar_bounding_boxes
from src.extraccion_de_caracteristicas import construir_base_datos

class CellDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Detección de Células")
        self.setGeometry(100, 100, 800, 600) # Tamaño y posición: 800x600 píxeles, iniciando en (100, 100) en la pantalla.

        # Widgets principales
        self.image_label = QLabel("Carga una imagen para empezar", self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(600, 400)

        self.load_button = QPushButton("Cargar Imagen")
        self.process_button = QPushButton("Procesar Imagen")
        self.save_button = QPushButton("Guardar Resultados")
        self.process_button.setEnabled(False) # Los botones de procesar y guardar están deshabilitados inicialmente.
        self.save_button.setEnabled(False)

        # Layout
        layout = QVBoxLayout() # Layout vertical, un widget abajo del otro
        layout.addWidget(self.image_label)
        layout.addWidget(self.load_button)
        layout.addWidget(self.process_button)
        layout.addWidget(self.save_button)

        # Container
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Conexiones: Cada botón está conectado a una función (o slot) que se ejecuta al hacer clic.
        self.load_button.clicked.connect(self.load_image)
        self.process_button.clicked.connect(self.process_image)
        self.save_button.clicked.connect(self.save_results)

        # Variables internas
        self.image_path = None
        self.processed_image = None
        self.df_ml = None
        self.df_all = None

        # Cargar el modelo guardado al iniciar la aplicación
        try:
            self.mejor_modelo = np.load("mejor_modelo.npy", allow_pickle=True)
            print("Modelo cargado exitosamente.")
        except FileNotFoundError:
            print("No se encontró el archivo del modelo. Asegúrate de guardarlo correctamente.")
            self.mejor_modelo = None

    # Funciones principales:
    # 0. Esta es una función que hará todo el procesamiento de la imagen
    def procesar_imagen_con_modelo(self, img):
        if self.mejor_modelo is None:
            print("No se cargó ningún modelo. Por favor, revisa la carga del modelo.")
            return None, None
        
        img_filtered = reducir_ruido(img)  
        img_rgb = cv2.cvtColor(img_filtered, cv2.COLOR_BGR2RGB)
        canal_rojo, canal_verde, canal_azul = separar_canales(img_rgb)  # Función que separa los canales de color
        canal_seleccionado = seleccionar_canal_mayor_contraste(canal_rojo, canal_verde, canal_azul)
        img_binarizada = binarizar_con_kmeans(canal_seleccionado)
        img_mediana = aplicar_filtro_mediana(img_binarizada)
        img_morfo = aplicar_operaciones_morfologicas(img_mediana)
        img_rellena = rellenar_celulas(img_morfo)
        img_ws, resultados_intermedios = aplicar_watershed(img_rellena, level=40)
        contornos, _ = cv2.findContours(img_ws, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_bounding_boxes = dibujar_bounding_boxes(canal_seleccionado, contornos, color=(0, 255, 0), grosor=1, umbral_area_min=5000)
        df = construir_base_datos(canal_seleccionado, contornos)
        X = df.drop(columns=["Imagen","ID"]).copy() 
        X = X.values
        predicciones = self.mejor_modelo.predict(X)

        return img_bounding_boxes, predicciones  # Devuelve la imagen con las cajas y el DataFrame

    # 1. Cargar Imagen
    def load_image(self):
        # Diálogo para cargar imagen
        file_path, _ = QFileDialog.getOpenFileName(self, "Seleccionar Imagen", "", "Archivos PNG (*.png);;Todos los archivos (*)")
        if file_path:
            self.image_path = file_path # Si se selecciona una imagen, se guarda su ruta en self.image_path
            pixmap = QPixmap(file_path)
            self.image_label.setPixmap(pixmap.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio)) # Carga la imagen con QPixmap y la muestra en la etiqueta (self.image_label), escalándola para mantener la proporción.
            self.process_button.setEnabled(True) # Habilita el botón de procesar imagen.
    
    # 2. Procesar Imagen + boton de analizar 
    def process_image(self):
        if self.image_path:
            # Cargar la imagen con OpenCV
            image = cv2.imread(self.image_path)
            # TODO: Aquí integra tu modelo de ML y procesa la imagen
            # Procesar la imagen con el modelo (función que agregamos)
            img_bounding_boxes,  = procesar_imagen_con_modelo(self, image)
            
            # Actualizar la imagen procesada
            self.processed_image = image
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            q_image = QPixmap.fromImage(QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888))
            self.image_label.setPixmap(q_image)
            self.save_button.setEnabled(True)

    # 3. Guardar Resultados
    def save_results(self):
        if self.processed_image is not None:
            # Guardar la imagen procesada
            save_path, _ = QFileDialog.getSaveFileName(self, "Guardar Imagen Procesada", "", "Archivos PNG (*.png)")
            if save_path:
                cv2.imwrite(save_path, self.processed_image) # Guarda la imagen con OpenCV (cv2.imwrite).
            # Guardar los dataframes en archivos CSV
            self.df_ml.to_csv("ml_results.csv", index=False)
            self.df_all.to_csv("all_cells.csv", index=False)

# Ejecucion Principal
if __name__ == "__main__":
    # Crea una instancia de la aplicación (QApplication) y de la ventana principal.
    app = QApplication(sys.argv)
    window = CellDetectionApp()
    window.show()
    # Ejecuta el bucle principal con app.exec_().
    sys.exit(app.exec_())
