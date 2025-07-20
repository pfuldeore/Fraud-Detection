import pandas as pd
import numpy as np
from datetime import timedelta
import re


def add_application_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts year, month, and day of week from application_date.
    """
    df = df.copy()
    df['application_date'] = pd.to_datetime(df['application_date'], errors='coerce') 
    df['application_year'] = df['application_date'].dt.year
    df['application_month'] = df['application_date'].dt.month
    df['application_day_of_week'] = df['application_date'].dt.dayofweek  # Monday=0
    return df


def add_income_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates income-related ratios.
    """
    df = df.copy()
    epsilon = 1e-6
    df['existing_emi_to_income_ratio'] = (
        df['existing_emis_monthly'] / (df['monthly_income'] + epsilon)
    ) * 100

    df['loan_amount_to_income_ratio'] = (
        df['loan_amount_requested'] / (df['monthly_income'] + epsilon)
    ) * 100
    return df


def generate_transaction_aggregation(
    loan_df: pd.DataFrame,
    transaction_df: pd.DataFrame,
    time_windows: list = [30, 90, 180, 365]
) -> pd.DataFrame:
    """
    Aggregates transaction-level features for each loan application
    over different time windows.
    """
    loan_df = loan_df.copy()
    transaction_df = transaction_df.copy()

    merged_df = pd.merge(loan_df, transaction_df, on='customer_id', how='left')
    merged_df.sort_values(by=['customer_id', 'transaction_date'], inplace=True)
    merged_df['application_date'] = pd.to_datetime(merged_df['application_date'], errors='coerce') 
    merged_df['transaction_date'] = pd.to_datetime(merged_df['transaction_date'], errors='coerce') 
    aggregated_features = []

    for customer_id, group in merged_df.groupby('customer_id'):
        for _, loan_row in group.drop_duplicates(subset='application_id').iterrows():
            application_date = loan_row['application_date']
            application_id = loan_row['application_id']

            transactions_before_app = group[group['transaction_date'] < application_date]
            feature_row = {'application_id': application_id}

            for window in time_windows:
                window_start = application_date - timedelta(days=window)
                transactions_in_window = transactions_before_app[
                    transactions_before_app['transaction_date'] >= window_start
                ]

                num_txns = transactions_in_window.shape[0]
                total_amt = transactions_in_window['transaction_amount'].sum()
                avg_amt = transactions_in_window['transaction_amount'].mean()
                unique_merchants = transactions_in_window['merchant_category'].nunique()

                feature_row[f'num_transactions_{window}d'] = num_txns
                feature_row[f'total_transaction_amount_{window}d'] = total_amt
                feature_row[f'average_transaction_amount_{window}d'] = avg_amt if num_txns > 0 else 0
                feature_row[f'unique_merchant_categories_{window}d'] = unique_merchants

            aggregated_features.append(feature_row)

    return pd.DataFrame(aggregated_features)

def add_location_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts city, state, and zip code from 'residential_address' column in the given DataFrame
    and returns the updated DataFrame with new columns: 'city', 'state', 'zip_code'.
    """

    def extract_city_state_zip(address):
        if pd.isnull(address):
            return pd.Series([None, None, None])

        parts = [p.strip() for p in address.split(',')]

        # Zip code extraction
        zip_code = None
        for part in reversed(parts):
            match = re.search(r'\b\d{6}\b', part)
            if match:
                zip_code = match.group()
                break

        # Guess city and state from last parts
        state = parts[-2] if len(parts) >= 2 else None
        city = parts[-3] if len(parts) >= 3 else None

        return pd.Series([city, state, zip_code])

    # Apply extraction
    df[['city', 'state', 'zip_code']] = df['residential_address'].apply(extract_city_state_zip)

    return df

def create_feature_engineered_dataset(
    loan_df: pd.DataFrame,
    transaction_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Complete pipeline for feature engineering on loan + transaction data.
    """
    loan_df = add_application_date_features(loan_df)
    loan_df = add_income_ratios(loan_df)
    loan_df = add_location_features(loan_df)
    
    txn_features_df = generate_transaction_aggregation(loan_df, transaction_df)
    full_df = pd.merge(loan_df, txn_features_df, on='application_id', how='left')

    return full_df
