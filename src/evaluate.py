import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve
)
from src.data_preprocessing import load_and_preprocess

def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()

def plot_roc_auc(model, X_test, y_test, n_classes=10):
    # One-vs-rest ROC AUC
    y_score = model.predict_proba(X_test)
    plt.figure()
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve((y_test == i).astype(int), y_score[:, i])
        plt.plot(fpr, tpr, label=f'Class {i} (AUC={roc_auc_score((y_test==i).astype(int), y_score[:,i]):.2f})')
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves by Class')
    plt.legend(loc='lower right')
    plt.show()

def main():
    print("\nReloading data & model...")
    X_train, X_test, y_train, y_test = load_and_preprocess()
    model = joblib.load('models/mlp_mnist.pkl')

    print("\nClassification Report:")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Plot
    plot_confusion_matrix(y_test, y_pred, labels=list(range(10)))
    plot_roc_auc(model, X_test, y_test)

if __name__ == "__main__":
