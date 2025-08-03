import os
import joblib
from src.data_preprocessing import load_and_preprocess
from src.model import build_model

def main():
    X_train, X_test, y_train, y_test = load_and_preprocess()
    mlp = build_model(hidden_layer_sizes=(128, 64), max_iter=300)

    print("Training MLP on MNIST subset...")
    mlp.fit(X_train, y_train)

    os.makedirs('models', exist_ok=True)
    model_path = os.path.join('models', 'mlp_mnist.pkl')
    joblib.dump(mlp, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
