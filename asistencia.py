import streamlit as st
import torch
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Face recognition",
    page_icon = "	:sauropod:"
)

st.title("Reconocimiento facial con PyTorch y Streamlit")
st.write("Somos un equipo apasionado de profesionales dedicados a hacer la diferencia")
st.write("Este proyecto fue desarrollado por María Camila Villamizar & Carlos Fernando Escobar Silva")

uploaded_file = st.file_uploader("Elija una imagen...", type=["jpg", "png", "jfif"])

# Inicializar dispositivo
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Ruta al modelo guardado
ruta_modelo = 'modelo_inception_resnet_v1.pth'

# Función para cargar el modelo
def cargar_modelo(ruta_modelo, device):
    # Inicializar el modelo
    modelo = InceptionResnetV1(pretrained='vggface2', classify=False, device=device).eval()
    # Cargar el estado del modelo guardado
    modelo.load_state_dict(torch.load(ruta_modelo, map_location=device))
    return modelo

# Cargar el modelo
encoder = cargar_modelo(ruta_modelo, device)

# Crear el detector MTCNN
mtcnn = MTCNN(select_largest=True, min_face_size=20, thresholds=[0.6, 0.7, 0.7], post_process=False, image_size=160, device=device)


if uploaded_file is not None:
    # Cargar la imagen
    image = Image.open(uploaded_file)

    # Mostrar la imagen
    st.image(image, caption='Imagen cargada', use_column_width=True)
    st.write("")
    st.write("Detectando caras...")

    # Detección de bounding box y landmarks
    boxes, probs, landmarks = mtcnn.detect(image, landmarks=True)

    if boxes is not None:
        # Dibujar las bounding boxes y landmarks en la imagen
        fig, ax = plt.subplots()
        ax.imshow(image)
        for box, landmark in zip(boxes, landmarks):
            ax.scatter(landmark[:, 0], landmark[:, 1], s=8, c='red')
            rect = plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, color='red')
            ax.add_patch(rect)
        st.pyplot(fig)

        # Detección de cara
        face = mtcnn(image)

        # Embedding de cara
        embedding_cara = encoder.forward(face.unsqueeze(0)).detach().cpu().numpy()
        st.write(f'Embedding de la cara: {embedding_cara}')
    else:
        st.write("No se detectó ninguna cara en la imagen.")
