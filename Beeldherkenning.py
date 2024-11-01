# beeldherkenning.py
import torch
from torchvision import models, transforms
from PIL import Image

class BeeldHerkenning:
    def __init__(self):
        # Gebruik een ResNet-model dat vooraf is getraind op ImageNet
        self.model = models.resnet50(weights='DEFAULT')  # Hernoem naar 'weights' voor compatibiliteit
        self.model.eval()
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_features(self, image_path):
        # Laad en verwerk de afbeelding
        image = Image.open(image_path).convert("RGB")  # Zorg ervoor dat de afbeelding RGB is
        image = self.transforms(image).unsqueeze(0)
        
        with torch.no_grad():
            features = self.model(image)
        
        # Converteer naar een lijst van floats
        return features.numpy().flatten().tolist()  # Flatten om een lijst van floats te krijgen

