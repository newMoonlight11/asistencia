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

# # importing the libraries and dependencies needed for creating the UI and supporting the deep learning models used in the project
# import os
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# import streamlit as st  
# import tensorflow as tf # TensorFlow is required for Keras to work
# from PIL import Image
# import numpy as np

# # hide deprication warnings which directly don't affect the working of the application
# import warnings
# warnings.filterwarnings("ignore")

# # set some pre-defined configurations for the page, such as the page title, logo-icon, page loading state (whether the page is loaded automatically or you need to perform some action for loading)
# st.set_page_config(
#     page_title="Reconocimiento de Caras",
#     page_icon = ":angry:",
#     initial_sidebar_state = 'auto'
# )

# # hide the part of the code, as this is just for adding some custom CSS styling but not a part of the main idea 
# hide_streamlit_style = """
# 	<style>
#   #MainMenu {visibility: hidden;}
# 	footer {visibility: hidden;}
#   </style>
# """

# st.markdown(hide_streamlit_style, unsafe_allow_html=True) # Oculta el código CSS de la pantalla, ya que están incrustados en el texto de rebajas. Además, permita que Streamlit se procese de forma insegura como HTML

# #st.set_option('deprecation.showfileUploaderEncoding', False)
# @st.cache_resource
# def load_model():
#     model=tf.keras.models.load_model('./caras_model.h5')
#     return model
# with st.spinner('Modelo está cargando..'):
#     model=load_model()
    


# with st.sidebar:
#         #st.image('rosa.jpeg')
#         st.title("Reconocimiento de imagen")
#         st.header("By  Camila Villamizar & Carlos Escobar")
#         st.subheader("Reconocimiento de imagen para caras")
#         st.write("UNAB")
#         confianza = st.slider("Seleccione el nivel de confianza %", 0, 100, 50)/100

# col1, col2, col3 = st.columns(3)

# """ with col2:
#      st.image('rosa.jpeg') """

# #st.image('logo.png')
# st.title("Modelo de reconocimiento de Caras")
# st.write("Somos un equipo apasionado de profesionales dedicados a hacer la diferencia")
# st.write("""
#          # Detección de Caras
#          """
#          )

# def predict(image_data, model, class_names):
    
#     image_data = image_data.resize((180, 180))
    
#     image = tf.keras.utils.img_to_array(image_data)
#     image = tf.expand_dims(image, 0) # Create a batch

    
#     # Predecir con el modelo
#     prediction = model.predict(image)
#     index = np.argmax(prediction)
#     score = tf.nn.softmax(prediction[0])
#     class_name = class_names[index]
    
#     return class_name, score


# class_names = open("./clases.txt", "r").readlines()

# img_file_buffer = st.camera_input("Capture una foto para tomar asistencia")    
# if img_file_buffer is None:
#     st.text("Por favor tome una foto")
# else:
#     image = Image.open(img_file_buffer)
#     st.image(image, use_column_width=True)
    
#     # Realizar la predicción
#     class_name, score = predict(image, model, class_names)
    
#     # Mostrar el resultado

#     if np.max(score)>0.5:
#         st.subheader(f"Estudiante: {class_name}")
#         st.text(f"Puntuación de confianza: {100 * np.max(score):.2f}%")
#     else:
#         st.text(f"No se pudo determinar el estudiante")


# PARTE 2 (USANDO CV2)
# import os
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# import streamlit as st  
# import tensorflow as tf
# from PIL import Image
# import numpy as np
# import cv2
# import dlib
# import face_recognition
# from datetime import datetime

# # Configuración de la página de Streamlit
# st.set_page_config(
#     page_title="Reconocimiento de Rostros",
#     page_icon=":smiley:",
#     initial_sidebar_state='auto'
# )

# # Estilos CSS para ocultar el menú y el pie de página
# hide_streamlit_style = """
#     <style>
#     #MainMenu {visibility: hidden;}
#     footer {visibility: hidden;}
#     </style>
# """
# st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# # Función para cargar el modelo
# @st.cache_resource
# def load_model():
#     model = tf.keras.models.load_model('./caras_model.h5')
#     return model

# with st.spinner('Modelo está cargando...'):
#     model = load_model()

# # Configuración de la barra lateral
# with st.sidebar:
#     st.image('face_icon.png')
#     st.title("Reconocimiento de Rostros")
#     st.subheader("Identifica rostros en imágenes y videos")
#     confianza = st.slider("Seleccione la confianza %", 0, 100, 50) / 100

# # Título de la página
# st.title("Modelo de Reconocimiento de Rostros")
# st.write("Aplicación para la toma de asistencia mediante el reconocimiento de rostros.")

# # Función para predecir el rostro
# def import_and_predict(image_data, model):
#     image_data = image_data.resize((180, 180))
#     image = tf.keras.utils.img_to_array(image_data)
#     image = tf.expand_dims(image, 0)  # Crear un batch
#     prediction = model.predict(image)
#     return prediction

# # Función para registrar la asistencia
# def register_attendance(name):
#     date_str = datetime.now().strftime('%Y-%m-%d')
#     with open('attendance.txt', 'a+') as file:
#         file.seek(0)
#         lines = file.readlines()
#         entries = [line.strip() for line in lines]
#         if f"{name} - {date_str}" not in entries:
#             file.write(f"{name} - {date_str}\n")

# # Cargar imágenes o videos
# upload_option = st.selectbox("Seleccione la fuente de entrada:", ["Cargar Imagen", "Cargar Video"])

# if upload_option == "Cargar Imagen":
#     img_file_buffer = st.file_uploader("Cargar una imagen", type=["jpg", "jpeg", "png"])
#     if img_file_buffer is not None:
#         image = Image.open(img_file_buffer)
#         st.image(image, caption='Imagen cargada', use_column_width=True)
#         # Convertir la imagen a formato compatible con face_recognition
#         image_np = np.array(image)
#         face_locations = face_recognition.face_locations(image_np)
#         face_encodings = face_recognition.face_encodings(image_np, face_locations)
        
#         for face_encoding in face_encodings:
#             predictions = model.predict([face_encoding])
#             name = "Desconocido"
#             # Verificar la predicción más confiable
#             if max(predictions) > confianza:
#                 name = class_names[np.argmax(predictions)]
#                 register_attendance(name)
#             st.write(f"Rostro reconocido: {name}")

# elif upload_option == "Cargar Video":
#     video_file_buffer = st.file_uploader("Cargar un video", type=["mp4", "mov", "avi"])
#     if video_file_buffer is not None:
#         video_bytes = video_file_buffer.read()
#         st.video(video_bytes)
#         temp_video_file = 'temp_video.mp4'
#         with open(temp_video_file, 'wb') as f:
#             f.write(video_bytes)
        
#         video_capture = cv2.VideoCapture(temp_video_file)
#         while video_capture.isOpened():
#             ret, frame = video_capture.read()
#             if not ret:
#                 break
            
#             face_locations = face_recognition.face_locations(frame)
#             face_encodings = face_recognition.face_encodings(frame, face_locations)
            
#             for face_encoding in face_encodings:
#                 predictions = model.predict([face_encoding])
#                 name = "Desconocido"
#                 if max(predictions) > confianza:
#                     name = class_names[np.argmax(predictions)]
#                     register_attendance(name)
#                 cv2.putText(frame, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
#             cv2.imshow('Video', frame)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
        
#         video_capture.release()
#         cv2.destroyAllWindows()

# # Leer los nombres de las clases
# class_names = open("./clases.txt", "r").readlines()
