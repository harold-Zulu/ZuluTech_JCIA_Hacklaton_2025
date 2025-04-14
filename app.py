import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import time
import torch.nn as nn

# Configuration de la page
st.set_page_config(
    page_title="ZuluTech - Classificateur de Prunes",
    page_icon="🤖",
    layout="wide"
)

# Style personnalisé
st.markdown("""
    <style>
    .main {
        background-color: #283338;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stFileUploader>div>div>div>div {
        background-color: #ffffff;
        border-radius: 5px;
        padding: 20px;
    }
    </style>
    """, unsafe_allow_html=True)
#Il s'agit d'une application qui permet de classer les prunes en fonction de leur état.
# Titre et description
st.title("ZuluTech - Classificateur de Prunes Africaine")
st.markdown("""
    ### Bienvenue dans notre system de classification intelligent de prunes
    Devellopper dans le cadre des journees Camerounaises de l'intelligence artificielle.
    
    
    Téléchargez une image de prune et notre modèle vous indiquera sa catégorie :
    -  unaffected (Prune saine)
    -  unripe (Prune non mûre)
    -  spotted (Prune tachetée)
    -  cracked (Prune fissurée)
    -  bruised (Prune meurtrie)
    -  rotten (Prune pourrie)
""")

# Définition du modèle CNN
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Configuration du modèle
@st.cache_resource
def load_model():
    try:
        st.write("Chargement du modèle...")
        # Créer une instance du modèle CNN
        model = SimpleCNN(num_classes=6)
        
        # Charger les poids sauvegardés
        state_dict = torch.load('plum_classifier.pth', map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        
        st.write("Modèle chargé avec succès")
        model.eval()
        return model
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle: {str(e)}")
        return None

# Fonction de prédiction
def predict(image, model):
    if model is None:
        st.error("Le modèle n'a pas pu être chargé")
        return None, None
        
    try:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image = transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            _, predicted = torch.max(outputs, 1)
        
        return predicted[0], probabilities
    except Exception as e:
        st.error(f"Erreur lors de la prédiction: {str(e)}")
        return None, None

# Interface principale
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Téléchargement de l'image")
    uploaded_file = st.file_uploader("Choisissez une image de prune", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Image téléchargée", use_column_width=True)

with col2:
    if uploaded_file is not None:
        st.subheader("📊 Résultats de l'analyse")
        
        # Barre de progression
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)
        
        # Chargement du modèle et prédiction
        model = load_model()
        prediction, probabilities = predict(image, model)
        
        # Définir les classes
        classes = ['unaffected', 'unripe', 'spotted', 'cracked', 'bruised', 'rotten']
        prediction_text = classes[prediction]
        
        # Style conditionnel pour la prédiction
        if prediction_text == 'unaffected':
            st.success(f"✅ Prune saine")
        elif prediction_text == 'unripe':
            st.warning(f"⚠️ Prune non mûre")
        elif prediction_text == 'spotted':
            st.warning(f"⚠️ Prune tachetée")
        elif prediction_text == 'cracked':
            st.warning(f"⚠️ Prune fissurée")
        elif prediction_text == 'bruised':
            st.warning(f"⚠️ Prune meurtrie")
        else:
            st.error(f"❌ Prune pourrie")
        
        # Détails des probabilités
        st.subheader("📈 Détails des probabilités")
        for i, (class_name, prob) in enumerate(zip(classes, probabilities)):
            st.metric(
                label=class_name,
                value=f"{prob:.2%}",
                delta=None
            )

# Pied de page
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p> Développé par ZuluTech - © 2024</p>
    </div>
""", unsafe_allow_html=True) 
