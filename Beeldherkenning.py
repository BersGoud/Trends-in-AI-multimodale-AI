import torch
from PIL import Image
from sklearn.decomposition import PCA
from transformers import ViTForImageClassification, ViTFeatureExtractor

class BeeldHerkenning:
    def __init__(self, n_components=3):  # Adjust as necessary
        self.model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k")
        self.feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
        self.model.eval()
        self.pca = PCA(n_components=n_components)
        self.fitted = False

    def fit_pca(self, feature_list):
        """ Train de PCA op een lijst van feature-vectors """
        if len(feature_list) < 2:
            raise ValueError("Er zijn niet genoeg features om PCA op toe te passen.")
        
        n_samples = len(feature_list)
        n_features = len(feature_list[0])  # Assuming all features are of the same length
        self.pca = PCA(n_components=min(3, n_samples - 1, n_features))  # Ensure components are within bounds
        self.pca.fit(feature_list)
        self.fitted = True


    def extract_features(self, image_path):
        # Laad en verwerk de afbeelding
        image = Image.open(image_path).convert("RGB")  # Zorg ervoor dat de afbeelding RGB is
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits  # Directe output van de model
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            
            # Get the number of classes from the model's config
            num_classes = self.model.config.num_labels
            k = min(num_classes, 5)  # Use the minimum of available classes and 5
            
            top_probabilities, top_indices = torch.topk(probabilities, k)
        
        labels = [self.model.config.id2label[idx.item()] for idx in top_indices[0]]
        features = logits.numpy().flatten()

        # Pas PCA toe als deze is getraind
        if self.fitted:
            reduced_features = self.pca.transform(features.reshape(1, -1))  # Reshape voor PCA
        else:
            reduced_features = features.reshape(1, -1)  # Geen PCA toegepast als deze niet is getraind

        return reduced_features.flatten().tolist(), labels  # Return both features and labels