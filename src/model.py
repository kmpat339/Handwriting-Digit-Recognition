from sklearn.neural_network import MLPClassifier

def build_model(hidden_layer_sizes=(100,), max_iter=200, random_state=42):
    """
    Create and return a Scikit-learn MLPClassifier.
    """
    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation='relu',
        solver='adam',
        max_iter=max_iter,
        random_state=random_state,
        verbose=True
    )
    return mlp
