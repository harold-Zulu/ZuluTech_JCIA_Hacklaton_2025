import os
import random
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# Configuration
IMG_SIZE = (224, 224)
NUM_CLASSES = 6
CLASS_NAMES = ['unaffected', 'unripe', 'spotted', 'cracked', 'bruised', 'rotten']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Charger le modèle entraîné
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
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

def load_model(model_path):
    model = SimpleCNN(NUM_CLASSES)
    model.load_state_dict(torch.load(model_path))
    model.to(DEVICE)
    model.eval()
    return model

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor, image

def predict_image(model, image_tensor):
    with torch.no_grad():
        image_tensor = image_tensor.to(DEVICE)
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    return predicted_class, confidence

def get_random_images(data_dir, num_images=10):
    all_images = []
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(data_dir, class_name)
        for img_name in os.listdir(class_dir):
            if img_name.endswith(('.jpg', '.jpeg', '.png')):
                all_images.append((os.path.join(class_dir, img_name), class_name))
    
    return random.sample(all_images, min(num_images, len(all_images)))

def main():
    # Charger le modèle
    model = load_model('plum_classifier.pth')
    
    # Obtenir 10 images aléatoires
    random_images = get_random_images('dataset')
    
    # Créer une figure pour afficher les résultats
    plt.figure(figsize=(20, 10))
    
    for i, (image_path, true_class) in enumerate(random_images):
        # Prétraiter l'image
        image_tensor, original_image = preprocess_image(image_path)
        
        # Faire la prédiction
        predicted_class, confidence = predict_image(model, image_tensor)
        
        # Afficher l'image et les résultats
        plt.subplot(2, 5, i + 1)
        plt.imshow(original_image)
        plt.title(f'Vrai: {true_class}\nPrédit: {CLASS_NAMES[predicted_class]}\nConfiance: {confidence:.2f}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('predictions.png')
    plt.close()

if __name__ == "__main__":
    main() 