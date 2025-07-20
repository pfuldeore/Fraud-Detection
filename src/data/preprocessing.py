import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def define_features_and_targets(df: pd.DataFrame):
    """
    Separates features and target columns.
    """
    y_fraud = df['fraud_flag']
    y_loan_status = df['loan_status']

    X = df.drop(columns=[
        'fraud_flag',
        'loan_status',
        'fraud_type',
        'application_id',
        'customer_id',
        'application_date',
        'residential_address'
    ])

    return X, y_fraud, y_loan_status


def identify_feature_types(X: pd.DataFrame):
    """
    Identifies numerical and categorical features.
    """
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    return numerical_features, categorical_features


def preprocess_features(X: pd.DataFrame):
    """
    Applies preprocessing transformations to numerical and categorical features.
    Returns transformed features and fitted preprocessor.
    """
    numerical_features, categorical_features = identify_feature_types(X)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    X_processed = preprocessor.fit_transform(X)

    print(f"\nNumerical features: {numerical_features}")
    print(f"Categorical features: {categorical_features}")
    print("Shape of original features X:", X.shape)
    print("Shape of processed features X_processed:", X_processed.shape)

    return X_processed, preprocessor


def prepare_data_for_modeling(df: pd.DataFrame):
    """
    Full pipeline: defines targets, separates features, and preprocesses them.
    Returns processed features, targets, and preprocessor.
    """
    X, y_fraud, y_loan_status = define_features_and_targets(df)
    X_processed, preprocessor = preprocess_features(X)
    return X_processed, y_fraud, y_loan_status, preprocessor
