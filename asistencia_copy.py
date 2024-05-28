# # streamlit_audio_recorder y whisper by Alfredo Diaz - version Mayo 2024

# # En VsC seleccione la version de Python (recomiendo 3.9) 
# #CTRL SHIFT P  para crear el enviroment (Escriba Python Create Enviroment) y luego venv 

# #o puede usar el siguiente comando en el shell
# #Vaya a "view" en el menú y luego a terminal y lance un terminal.
# #python -m venv env

# #Verifique que el terminal inicio con el enviroment o en la carpeta del proyecto active el env.
# #cd D:\flores\env\Scripts\
# #.\activate 

# #Debe quedar asi: (.venv) D:\proyectos_ia\Flores>

# #Puedes verificar que no tenga ningun libreria preinstalada con
# #pip freeze
# #Actualicie pip con pip install --upgrade pip

# #pip install tensorflow==2.15 La que tiene instalada Google Colab o con la versión qu fué entrenado el modelo
# #Verifique se se instaló numpy, no trate de instalar numpy con pip install numpy, que puede instalar una version diferente
# #pip install streamlit
# #Verifique se se instaló no trante de instalar con pip install pillow
# #Esta instalacion se hace si la requiere pip install opencv-python

# #Descargue una foto de una flor que le sirva de ícono 

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

st.title("Reconocimiento facial con PyTorch y Streamlit")
st.write("Somos un equipo apasionado de profesionales dedicados a hacer la diferencia")
st.write("Este proyecto fue desarrollado por María Camila Villamizar & Carlos Fernando Escobar Silva")

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
    encoder = inicializar_modelo(device)

# Crear el detector MTCNN
mtcnn = MTCNN(select_largest=True, min_face_size=20, thresholds=[0.6, 0.7, 0.7], post_process=False, image_size=160, device=device)

def cargar_imagen(path):
    imagen = Image.open(path)
    imagen = imagen.convert('RGB')
    return imagen

def procesamiento_imagen(image):
     boxes, probs, landmarks = mtcnn.detect(image, landmarks=True)
    # Detección de cara
     faces = mtcnn(image)

     embeddings = []
     for face in faces:
        face = face.unsqueeze(0)
        embedding = encoder(face).detach().cpu()
        embeddings.append(embedding)
    
     return embeddings, boxes

def identificar(embedding_cara, embeddings):
    comparaciones = {}
    for nombre, emb_list in embeddings.items():
        min_dist = min(euclidean(embedding_cara.flatten(), emb.flatten()) for emb in emb_list)
        comparaciones[nombre] = min_dist
    
    nombre_reconocido = min(comparaciones, key=comparaciones.get)
    return nombre_reconocido

def identificar_multiples_rostros(imagen, embeddings):
    embeddings_imagen, boxes = procesamiento_imagen(imagen)
    nombres_caras = []

    for i, embedding in enumerate(embeddings_imagen):
        nombre = identificar(embedding, embeddings)
        nombres_caras.append((nombre, boxes[i]))

    return nombres_caras

ruta_dataset = r"data"

@st.cache_resource
def cargar_embeddings():
    embeddings = {}

    for root, dirs, files in os.walk(ruta_dataset):
        for name in dirs:
            nombre_persona = name
            embeddings[nombre_persona] = []
            ruta_persona = os.path.join(root, name)
            for filename in os.listdir(ruta_persona):
                if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
                    ruta_imagen = os.path.join(ruta_persona, filename)
                    embeddings[nombre_persona].append(procesamiento_imagen(cargar_imagen(ruta_imagen)))
    
    return embeddings

embeddings = cargar_embeddings()

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

# Subir imagen desde archivo
uploaded_file = st.file_uploader("Elija una imagen...", type=["jpg", "png", "jfif"])
img_file_buffer = st.camera_input("O capture una foto para identificar una cara")

if img_file_buffer is not None:
    # Cargar la imagen
    image = Image.open(img_file_buffer)
elif uploaded_file is not None:
    image = Image.open(uploaded_file)
else:
    st.write("Por favor tome una foto o suba una imagen")
    image = None
    
if image is not None:
    st.image(image, use_column_width=True)

    #Procesar la imagen
    nombres_caras = identificar_multiples_rostros(image, embeddings)
    if nombres_caras:
        nombres_detectados = [nombre for nombre, _ in nombres_caras]
        registrar_asistencia(nombres_detectados)

        for nombre, box in nombres_caras:
            x1, y1, x2, y2 = box.astype(int)
            recorte_cara = np.array(image)[y1:y2, x1:x2]
            st.image(recorte_cara, caption=nombre)
        st.success(f"Asistieron a clase: {', '.join(nombres_detectados)}")
    else:
        st.text("No se detectó ningún rostro en la imagen")