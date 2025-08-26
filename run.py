from src.TableDetector import TableDetector

# Initialisation du détecteur
detector = TableDetector(model_name="TahaDouaji/detr-doc-table-detection")

# Lancer la prédiction sur une image d’exemple
results = detector.predict("Tests/Data/Invoice.png", threshold=0.9)

# Afficher les résultats
for r in results:
    print(r)
