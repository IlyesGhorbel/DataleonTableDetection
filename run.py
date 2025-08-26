from src.TableDetector import TableDetector
import argparse

# Créer le parser pour récupérer les arguments
parser = argparse.ArgumentParser(description="Run table detection on an image")
parser.add_argument(
    "--image", type=str, required=True, help="Path to the image you want to analyze"
)
parser.add_argument(
    "--threshold", type=float, default=0.9, help="Confidence threshold (default=0.9)"
)
args = parser.parse_args()

# Initialisation du détecteur
detector = TableDetector(model_name="TahaDouaji/detr-doc-table-detection")

# Lancer la prédiction sur l’image passée en argument
results = detector.predict(args.image, threshold=args.threshold)

# Afficher les résultats
for r in results:
    print(r)




