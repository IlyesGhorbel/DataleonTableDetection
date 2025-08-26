from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import torch
import os

class TableDetector:
    def __init__(self, model_name="TahaDouaji/detr-doc-table-detection"):
        # Initialisation du processor et du modèle DETR
        self.processor = DetrImageProcessor.from_pretrained(model_name)
        self.model = DetrForObjectDetection.from_pretrained(model_name)

    def predict(self, image_path: str, threshold: float = 0.9):
        # Vérification que le fichier existe
        if not os.path.exists(image_path):
            raise ValueError(f"Fichier inexistant : {image_path}")

        try:
            image = Image.open(image_path)
            # Conversion en RGB si nécessaire
            if image.mode != "RGB":
                image = image.convert("RGB")
        except Exception as e:
            raise ValueError(f"Impossible d'ouvrir l'image {image_path}: {e}")

        # Préparer l'image pour le modèle
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)

        # Taille de sortie pour post-traitement
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=threshold
        )[0]

        # Transformer les résultats en liste de dictionnaires
        return [
            {
                "score": score.item(),
                "label": self.model.config.id2label[label.item()],
                "box": [round(i, 2) for i in box.tolist()],
            }
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"])
        ]
