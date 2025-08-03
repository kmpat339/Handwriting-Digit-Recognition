# Handwriting Digit Recognition

An end-to-end image recognition pipeline using Scikit-learn’s neural network on the MNIST dataset.

- **Preprocessing**  
  • Load 5000+ handwritten digits  
  • Normalize pixel values, reshape for model input, handle missing values  
- **Model**  
  • Multi-layer perceptron (MLPClassifier)  
- **Evaluation**  
  • Precision, recall, F1-score, ROC-AUC  
  • Confusion matrix & ROC curves  

---

## 📦 Installation

```bash
git clone https://github.com/<your-username>/Handwriting-Digit-Recognition.git
cd Handwriting-Digit-Recognition
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## 🚀 Usage

1. **Train the model**  
   ```bash
   python src/train.py
   ```  
   Saves trained model to `models/mlp_mnist.pkl`.

2. **Evaluate performance**  
   ```bash
   python src/evaluate.py
   ```  
   Prints classification report, shows confusion matrix and ROC curves.

3. **Explore in Jupyter**  
   ```bash
   jupyter lab notebooks/exploratory.ipynb
   ```

---

## 📂 File overview

- `src/data_preprocessing.py`  
  Data‐loading & preprocessing functions.  
- `src/model.py`  
  Encapsulates MLP model creation.  
- `src/train.py`  
  Script to train and serialize the model.  
- `src/evaluate.py`  
  Script to load model, evaluate on test set, and plot metrics.  
- `notebooks/exploratory.ipynb`  
  Walkthrough of data analysis & visualizations.
