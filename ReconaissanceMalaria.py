import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os
import random
import gdown

st.set_page_config(
    page_title="Détection de la Malaria",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Agrandir la sidebar
st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {
            min-width: 350px;
            max-width: 350px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

ms = st.session_state

if "themes" not in ms:
    ms.themes = {
        "current_theme": "light",
        "refreshed": True,
        "light": {
            "theme.base": "light",
            "theme.backgroundColor": "#ffffff",
            "theme.primaryColor": "#000000",
            "theme.secondaryBackgroundColor": "#f5f5f5",
            "theme.textColor": "#0a1464",
            "button_face": "🌙",
            "button_text": "Appuyer pour passer en mode sombre",
        },
        "dark": {
            "theme.base": "dark",
            "theme.backgroundColor": "#0e0e0e",
            "theme.primaryColor": "#c98bdb",
            "theme.secondaryBackgroundColor": "#2b2b2b",
            "theme.textColor": "white",
            "button_face": "☀️",
            "button_text": "Appuyer pour passer en mode clair",
        },
    }


def change_theme():
    current_theme = ms.themes["current_theme"]
    new_theme = "dark" if current_theme == "light" else "light"
    ms.themes["current_theme"] = new_theme

    for key, value in ms.themes[new_theme].items():
        if key.startswith("theme"):
            st._config.set_option(key, value)

    ms.themes["refreshed"] = False


current_theme_config = ms.themes[ms.themes["current_theme"]]
btn_face = current_theme_config["button_face"]
btn_text = current_theme_config["button_text"]

# Sidebar : Nom + Thème + Options
st.sidebar.markdown("Réalisé par **Yanis Zedira et Aymen Djerad**")
st.sidebar.title("Détection de la Malaria")
st.sidebar.button(f"{btn_face} {btn_text}", on_click=change_theme)

if ms.themes["refreshed"] == False:
    ms.themes["refreshed"] = True
    st.rerun()

st.sidebar.write(f"Mode actuel : **{ms.themes['current_theme'].capitalize()}**")


# Téléchargement du modèle depuis Google Drive
@st.cache_resource
def load_cnn_model():
    model_path = "CNN_best.keras"

    if not os.path.exists(model_path):
        st.write("📥 Téléchargement du modèle depuis Google Drive...")
        url = 'https://drive.google.com/uc?id=1S-_M7q4LfaOpqPaHnNfCxCI7ypnYX8Nd'
        gdown.download(url, model_path, quiet=False)

    model = load_model(model_path)
    return model


model = load_cnn_model()


def predict(image_path, model, threshold=0.5):
    img = image.load_img(image_path, target_size=(64, 64))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0][0]
    return prediction <= threshold


@st.cache_resource
def load_test_images():
    test_images = []
    test_images_dir = "images/"
    for img_name in os.listdir(test_images_dir):
        if img_name.endswith(('.png', '.jpg', '.jpeg')):
            test_images.append(os.path.join(test_images_dir, img_name))

    random.shuffle(test_images)

    return test_images


test_images = load_test_images()

st.sidebar.title("Options")
app_mode = st.sidebar.selectbox(
    "Que souhaitez-vous faire ?",
    ["Tester une Image", "En savoir plus sur le projet"]
)

if app_mode == "Tester une Image":
    st.title("Tester une Image")

    uploaded_file = st.file_uploader("Téléchargez une image de cellule sanguine", type=["png", "jpg", "jpeg"])
    selected_test_image = st.selectbox("Ou sélectionnez une image dans la base de donnée", test_images)

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Image téléchargée", width=200)
        img_path = uploaded_file
    else:
        img = Image.open(selected_test_image)
        st.image(img, caption="", width=200)
        img_path = selected_test_image

    if st.button("Analyser l'Image"):
        with st.spinner("🧪 Analyse en cours..."):
            is_infected = predict(img_path, model)

        if is_infected:
            st.error("**Résultat : Infecté**")
        else:
            st.success("**Résultat : Non Infecté**")

elif app_mode == "En savoir plus sur le projet":
    st.title("En savoir plus sur le projet")
    st.markdown("""
    Ce projet est né d’un constat simple : détecter la malaria à partir d’images microscopiques de frottis sanguins est une tâche complexe qui demande une grande expertise. 
    Face à cette difficulté, j’ai choisi de concevoir de A à Z un réseau de neurones convolutionnel (CNN), sans utiliser de modèle préentraîné, dans le but d’automatiser cette détection.

    Pourquoi avoir choisi un CNN précisément ?  
    Parce que les images microscopiques de cellules sont pleines de détails subtils. Il faut souvent l’œil aiguisé d’un spécialiste pour repérer les anomalies liées à la présence d’un parasite. 
    Le CNN est l’outil idéal pour cette tâche, car il est capable d’apprendre à reconnaître ces motifs complexes tout seul, en analysant des images. 
    Il n’a pas besoin qu’on lui explique quoi chercher : il va découvrir, au fil de l’entraînement, quelles formes et quelles textures sont caractéristiques d’une cellule saine ou infectée.

    Le modèle que j’ai conçu repose sur une architecture classique de CNN avec des couches de convolution, des opérations de pooling, de la normalisation, et des couches denses pour la classification finale. 
    Il a été entraîné sur un ensemble de données d’images de cellules saines et infectées, avec des techniques d’augmentation pour éviter le surapprentissage.

    Ce système pourrait servir d’outil d’assistance au diagnostic dans des zones où l’accès à des experts est limité.
    """)

# Footer sur chaque page
st.markdown(
    """
    <hr style="border:1px solid #f0f0f0; margin-top: 40px; margin-bottom: 10px;">
    <div style="text-align: center; font-size: 14px;">
        Réalisé par <strong>Yanis Zedira et Aymen Djerad</strong>
    </div>
    """,
    unsafe_allow_html=True,
)
