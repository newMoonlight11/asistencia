# import os
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# import streamlit as st  
# import tensorflow as tf
# from PIL import Image
# import numpy as np
# import cv2
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
#     model = tf.keras.models.load_model('./face_recognition_model.h5')
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
