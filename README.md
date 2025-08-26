# 📊 Table Detection Project

A Python project for **automatic table detection** in documents (invoices, bank statements, reports, etc.) using the [DETR model](https://huggingface.co/TahaDouaji/detr-doc-table-detection).

---

## 🚀 Features
- Detects tables in scanned images.
- Returns bounding boxes, confidence scores, and labels.
- Built with [Transformers](https://huggingface.co/docs/transformers/index) and [PyTorch](https://pytorch.org/).
- Includes unit tests with `pytest`.

---

## 📂 Project Structure
```
TableDetectionProject/
│
├── src/
│ └── TableDetector.py # Core TableDetector class
│
├── Tests/
│ ├── Data/ # Sample test images
│ └── TestTableDetector.py # Unit tests
│
├── run.py # Example script to run detection
├── requirements.txt # Python dependencies
└── README.md # Project documentation
```


## ⚙️ Installation
## 1. Clone the repository:
```bash
git clone https://github.com/IlyesGhorbel/DataleonTableDetection.git
cd TableDetectionProject
```
## 2 .Create and activate a virtual environment:
Linux / macOS
```bash
python3 -m venv venv
source venv/bin/activate
```
Windows (cmd)
```bash
python -m venv venv
venv\Scripts\activate.bat
```
## 3. Install dependencies:
```bash
pip install -r requirements.txt
```

## ▶️ Usage
Run table detection on any image. Replace the path with your desired image:
```bash
python run.py --image Tests/Data/Invoice.png --threshold 0.9
```
Example output:
```bash
{'score': 0.9869465827941895, 'label': 'table', 'box': [53.92, 290.43, 637.3, 701.62]}
```
## 🧪 Running Tests
```bash
pytest -v
```
## 🛠️ Prérequis
- Python >= 3.8
  



