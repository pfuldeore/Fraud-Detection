import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

def train_models(X, y, test_size=0.2, random_state=42):
    """
    Splits the data, applies SMOTE, and trains Logistic Regression, 
    Random Forest, and LightGBM models.
    
    Returns:
        models (dict): Trained model objects.
        X_test, y_test: Held-out test set for evaluation.
    """
    # 1. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print("Train/Test split complete.")
    print("Training shape:", X_train.shape, "Test shape:", X_test.shape)
    print("Class balance in training:\n", y_train.value_counts())

    # 2. Apply SMOTE for class imbalance
    smote = SMOTE(random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print("\nSMOTE applied:")
    print("Resampled training shape:", X_train_resampled.shape)
    print("Resampled class balance:\n", y_train_resampled.value_counts())

    # 3. Train models
    print("\nTraining models...")

    lr_model = LogisticRegression(max_iter=1000, random_state=random_state)
    rf_model = RandomForestClassifier(random_state=random_state)

    lr_model.fit(X_train_resampled, y_train_resampled)
    rf_model.fit(X_train_resampled, y_train_resampled)

    models = {
        "Logistic Regression": lr_model,
        "Random Forest": rf_model
    }

    print("Model training complete.")

    return models, X_test, y_test
