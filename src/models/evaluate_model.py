from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def evaluate_models(models, X_test, y_test):
    """
    Evaluates a dictionary of models on test data and prints metrics.
    Returns the best model based on ROC AUC score.
    """
    print("\nEvaluating models on the test set:")

    best_model = None
    best_score = 0
    best_model_name = ""

    for name, model in models.items():
        print(f"\n--- {name} ---")

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        roc_auc = roc_auc_score(y_test, y_pred_proba)
        print(f"ROC AUC: {roc_auc:.4f}")

        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        if roc_auc > best_score:
            best_score = roc_auc
            best_model = model
            best_model_name = name

    print(f"\nBest Model: {best_model_name} with ROC AUC: {best_score:.4f}")
    return best_model
