import pandas as pd
import os

def load_csv_data(file_path: str) -> pd.DataFrame:
    """
    Loads a CSV file into a pandas DataFrame.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
    
        pd.DataFrame: Loaded data.

    Raises:
        FileNotFoundError: If file is not found at the path.
        pd.errors.EmptyDataError: If file is empty.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded '{file_path}' successfully. Shape: {df.shape}")
        return df
    except pd.errors.EmptyDataError:
        raise pd.errors.EmptyDataError(f"File is empty: {file_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading file {file_path}: {e}")

def calculate_credit_score(df):
    score = 0

    # CIBIL
    score += min(max((df["cibil_score"] - 300) / 6, 0), 40)

    # Income
    if df["monthly_income"] >= 100000:
        score += 20
    elif df["monthly_income"] >= 50000:
        score += 15
    elif df["monthly_income"] >= 30000:
        score += 10
    else:
        score += 5

    # DTI
    if df["debt_to_income_ratio"] < 20:
        score += 15
    elif df["debt_to_income_ratio"] < 35:
        score += 10
    else:
        score += 5

    # EMI to Income Ratio
    emi_ratio = (df["existing_emis_monthly"] / (df["monthly_income"] + 1e-6)) * 100
    if emi_ratio < 20:
        score += 10
    elif emi_ratio < 40:
        score += 5

    # Loan to Income Ratio
    loan_ratio = (df["loan_amount_requested"] / (df["monthly_income"] + 1e-6)) * 100
    if loan_ratio < 100:
        score += 10
    elif loan_ratio < 200:
        score += 5

    # Age
    if df["applicant_age"] >= 30:
        score += 5

    return round(score, 1)

def get_credit_label(score):
    if score >= 80:
        return "Excellent"
    elif score >= 60:
        return "Good"
    elif score >= 40:
        return "Average"
    else:
        return "Poor"

