import os
import pickle
from src.data.preprocessing import prepare_data_for_modeling
from src.models.train_model import train_models
from src.models.evaluate_model import evaluate_models
from src.features.build_features import create_feature_engineered_dataset
from src.utils.helper import load_csv_data

# Load and prepare data
loan_applications_df = load_csv_data('data/loan_applications.csv')
transactions_df = load_csv_data('data/transactions.csv')
loan_df = create_feature_engineered_dataset(loan_applications_df, transactions_df)
X_processed, y_fraud, _, preprocessor = prepare_data_for_modeling(loan_df)

# Train models
models, X_test, y_test = train_models(X_processed, y_fraud)

# Evaluate models and get best model
best_model = evaluate_models(models, X_test, y_test)

# Save best model
# Save best model, preprocessor, and feature columns
os.makedirs("outputs/models", exist_ok=True)

with open("outputs/models/new_best_model_bundle.pkl", "wb") as f:
    pickle.dump({
        "model": best_model,
        "preprocessor": preprocessor,
        "feature_columns": preprocessor.get_feature_names_out().tolist()
        
    }, f)

print("Best model, preprocessor, and feature columns saved to outputs/models/best_model_bundle.pkl")
