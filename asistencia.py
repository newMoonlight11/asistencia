import streamlit as st
import torch
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN
import matplotlib.pyplot as plt
import warnings
from datetime import datetime
from scipy.spatial.distance import euclidean
import numpy as np
import os

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Face recognition",
    page_icon=":sauropod:"
)

st.image('ing_s.png', width=300)
st.title("Reconocimiento facial con PyTorch y Streamlit")
st.write("Este proyecto se centra en el desarrollo de una aplicación web interactiva para el reconocimiento facial utilizando PyTorch y Streamlit, permitiendo la captura de imágenes y también la carga de imágenes. Implementa un modelo preentrenado de InceptionResnetV1 para la generación de embeddings faciales y utiliza MTCNN para la detección de caras en imágenes.")

class_names = open("./clases.txt", "r").readlines()

# Inicializar dispositivo
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Función para inicializar el modelo
@st.cache_resource
def inicializar_modelo(device):
    # Inicializar el modelo con pesos preentrenados
    modelo = InceptionResnetV1(pretrained='vggface2', classify=False, device=device).eval()
    return modelo

# Inicializar el modelo
with st.spinner('Modelo está cargando'):
    _encoder = inicializar_modelo(device)

# Crear el detector MTCNN
_mtcnn = MTCNN(select_largest=True, min_face_size=20, thresholds=[0.6, 0.7, 0.7], post_process=False, image_size=160, device=device)

@st.cache_resource
def obtener_embeddings(ruta_dataset, _encoder, _mtcnn, device):
    embeddings = {}
    nombres = []
    for root, dirs, files in os.walk(ruta_dataset):
        for file in files:
            if file.endswith(('jpg', 'png', 'jfif')):
                image_path = os.path.join(root, file)
                image = Image.open(image_path)
                
                # Detección de cara
                face = _mtcnn(image)
                if face is not None:
                    # Embedding de cara
                    embedding_cara = _encoder.forward(face.reshape((1, 3, 160, 160))).detach().cpu()
                    # Obtener el nombre de la persona (carpeta)
                    nombre = os.path.basename(root)
                    if nombre not in embeddings:
                        embeddings[nombre] = []
                        nombres.append(nombre)
                    embeddings[nombre].append(embedding_cara)
                else:
                    st.write("")
    return embeddings, nombres

ruta_dataset = r"data"

# Obtener embeddings de todas las imágenes en el dataset
with st.spinner('Procesando el dataset...'):
    embeddings, nombres = obtener_embeddings(ruta_dataset, _encoder, _mtcnn, device)
    # st.write("Embeddings generados para todas las imágenes en el dataset.")

def detectar_y_mostrar_caras(image, embeddings, nombres):
    # Detección de bounding box y landmarks
    boxes, probs, landmarks = _mtcnn.detect(image, landmarks=True)

    if boxes is not None:
        fig, ax = plt.subplots()
        ax.imshow(image)
        for box, landmark in zip(boxes, landmarks):
            ax.scatter(landmark[:, 0], landmark[:, 1], s=8, c='red')
            rect = plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, color='red')
            ax.add_patch(rect)
        st.pyplot(fig)

        # Detección de cara
        face = _mtcnn(image)

        # Embedding de cara
        if face is not None:
            embedding_cara = _encoder.forward(face.reshape((1, 3, 160, 160))).detach().cpu()

            # Comparar con embeddings del dataset
            comparaciones = {}
            for nombre, lista_embeddings in embeddings.items():
                min_dist = min(euclidean(embedding_cara.flatten(), emb.flatten()) for emb in lista_embeddings)
                comparaciones[nombre] = min_dist
            nombre_reconocido = min(comparaciones, key=comparaciones.get)
            st.write(f'Persona reconocida: {nombre_reconocido}')
        else:
            st.write("No se detectó ninguna cara en la imagen.")
    else:
        st.write("No se detectó ninguna cara en la imagen.")


# Subir imagen desde archivo
uploaded_file = st.file_uploader("Elija una imagen...", type=["jpg", "png", "jfif"])
if uploaded_file is not None:
    # Cargar la imagen
    image = Image.open(uploaded_file)

    # Mostrar la imagen
    st.image(image, caption='Imagen cargada', use_column_width=True)
    st.write("Detectando caras...")
    
    detectar_y_mostrar_caras(image, embeddings, nombres)

# Capturar imagen desde la cámara
img_file_buffer = st.camera_input("O capture una foto para identificar una cara")
if img_file_buffer is not None:
    # Cargar la imagen
    image = Image.open(img_file_buffer)

    # Mostrar la imagen
    st.image(image, caption='Imagen capturada', use_column_width=True)
    st.write("Detectando caras...")
    
    detectar_y_mostrar_caras(image, embeddings, nombres)

def registrar_asistencia(participantes):
    fecha_actual = datetime.now().strftime("%Y-%m-%d")
    archivo_registro = f"registro_asistencia_{fecha_actual}.txt"

    try:
        with open(archivo_registro, 'r') as archivo:
            asistencia_previa = archivo.read().splitlines()
    except FileNotFoundError:
        asistencia_previa = []
    
    nuevos_participantes = [nombre for nombre in participantes if nombre not in asistencia_previa]
    if nuevos_participantes:
        with open(archivo_registro, 'a') as archivo:
            for nombre in nuevos_participantes:
                archivo.write(nombre + "\n")

st.markdown("***")
st.markdown("<h6 style='text-align: center;'>© 2024 Proyecto de Reconocimiento Facial | Desarrollado por María Camila Villamizar & Carlos Fernando Escobar Silva</h6>", unsafe_allow_html=True)
