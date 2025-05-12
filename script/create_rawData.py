import os
import pandas as pd
import numpy as np
import logging

# === Directory Structure Setup ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
LOG_DIR = os.path.join(BASE_DIR, 'logs')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# === Logger Setup ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'data_creation.log')),
        logging.StreamHandler()  # This will display logs in the console
    ]
)

logging.info("=== Starting Sample Data Creation ===")

# === Step 1: Data Ingestion ===
def create_sample_data():
    sample_data = pd.DataFrame({
        'originallyScheduledPaymentAmount': np.random.rand(1000) * 1000,
        'paymentAmount': np.random.rand(1000) * 1000,
        'state': np.random.choice(['CA', 'NJ', 'MO', 'WI', 'IL', 'MI', 'FL', 'SC', 'OH', 'NV', 'IN',
        'VA', 'OK', 'NC', 'MN', 'TX', 'TN', 'HI', 'MS', 'PA', 'KY', 'AL',
        'GA', 'NM', 'ID', 'ND', 'AZ', 'CO', 'NE', 'SD', 'LA', 'CT', 'KS',
        'DE', 'WY', 'UT', 'IA', 'AK', 'WA', 'RI', 'Others'], 1000),
        'apr': np.random.rand(1000) * 20,
        'loanAmount': np.random.rand(1000) * 5000,
        'identity_verification_reason_code': np.random.randint(0, 150, 1000),
        'leadCost': np.random.rand(1000) * 50,
        'total_fraud_indicators': np.random.randint(0, 10, 1000),
        'loanStatus_cleaned': np.random.choice(['Paid Off', 'New Loan', 'Collection', 'Rejected', 'Withdrawn',
        'Returned Item', 'Other', 'Voided', 'Charged Off', 'Unknown'], 1000),
        'name_address_match': np.random.choice(['match', 'partial', 'mismatch', 'unavailable', 'invalid'], 1000),
        'max_ssns_per_bank_account': np.random.randint(0, 5, 1000),
        'leadType': np.random.choice(['prescreen', 'lead', 'organic', 'bvMandatory', 'california',
        'rc_returning', 'instant-offer', 'express', 'lionpay', 'repeat'], 1000),
        'payFrequency': np.random.choice(['B', 'W', 'S', 'M', 'I'], 1000),
        'work_phone_listed_as_cellphone': np.random.choice(['False', 'Unknown', 'True'], 1000),
        'fpStatus': np.random.choice(['Checked', 'Rejected', 'NotFunded', 'Skipped', 'No Payments',
        'Cancelled', 'No Schedule', 'Pending'], 1000),
        'invalid_driver_license_format': np.random.choice(['Unknown', 'False', 'True'], 1000),
        'ssn_dob_match': np.random.choice(['invalid', 'match', 'partial', 'mismatch', 'unavailable'], 1000),
        'driver_license_inconsistent_with_file': np.random.choice(['Unknown', 'False', 'True'], 1000),
        'address_conflict_on_file': np.random.choice([False,  True], 1000),
        'overall_identity_match_result':np.random.choice(['partial', 'match', 'other', 'mismatch'], 1000),
        'paymentStatus': np.random.choice(['Checked', 'Cancelled', 'Pending', 'Rejected', 'Skipped', 'Rejected Awaiting Retry']) * 1000,
        'non_residential_address': np.random.choice([False,  True], 1000),
        'work_phone_listed_as_homephone': np.random.choice(['False', 'Unknown', 'True'], 1000),
        'phone_inconsistent_with_state': np.random.choice(['False', 'Unknown', 'True'], 1000),
        'phone_match_type_description': np.random.choice(['(U) Unlisted', '(M) Mobile Phone', '(P) Pager',
        '(L) Last Name Only', '(FA) Full Name and Address','(F) Full Name Only', '(LA) Last Name and Address', '(A) Address Only'], 1000),
        'ssn_name_match': np.random.choice(['match', 'mismatch', 'unavailable', 'partial'], 1000),
        'non_residential_address_on_file': np.random.choice([False,  True], 1000),
        'current_address_new_trade_only': np.random.choice([False,  True], 1000),
        'current_address_not_on_file': np.random.choice([False,  True], 1000),
        'fees': np.random.rand(1000) * 1000,
        'phone_inconsistent_with_address': np.random.choice([False,  True], 1000),
        'inquiries_gt_3_last_30days': np.random.choice([False,  True], 1000),
        'principal': np.random.rand(1000) * 1000,
        'phone_match_result': np.random.choice(['invalid', 'unavailable', 'partial', 'match', 'mismatch'], 1000),
        'current_address_reported_lt_90days': np.random.choice([False,  True], 1000),
        'installmentIndex': np.random.randint(0, 20, 1000),
        'credit_before_age_18': np.random.choice([False,  True], 1000),
        'ssn_reported_frequently_for_another': np.random.choice([False,  True], 1000),
        'high_risk_inquiry_address': np.random.choice([False,  True], 1000),
        'high_risk_address_on_file': np.random.choice([False,  True], 1000),
        'credit_before_ssn_issue_date': np.random.choice([False,  True], 1000),
        'high_probability_ssn_belongs_to_another': np.random.choice([False,  True], 1000),
        'age_younger_than_ssn_issue_date': np.random.choice([False,  True], 1000),
        'isCollection': np.random.choice([False,  True], 1000),
        'unverified_ssn_issue_date': np.random.choice([False,  True], 1000),
        'originated': np.random.choice([True,  False], 1000),
        'paymentReturnCode': np.random.choice(['R01', 'R02', 'R03', 'R04', 'R05', 'R06', 'R07', 'R08', 'R09', 'R10',
        'R11', 'R12', 'R13', 'R14', 'R15', 'R16', 'R17', 'R20', 'R21', 'R22',
        'R23', 'R24', 'R29', 'R31', 'R33']),
        'isFunded': np.random.choice(['0',  '1'], 1000),
        'approved': np.random.choice(['0',  '1'], 1000),
        # 'loanRisk': np.random.choice([0, 1, 2], 1000)  # High, Low, Moderate
    })

    # Get the directory where your notebook is located
    current_dir = os.path.dirname(os.path.abspath('__file__'))  
    DATA_DIR = os.path.join(current_dir, 'data')

    # Create directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)

    # Save the data
    sample_path = os.path.join(DATA_DIR, 'sample_loan_data.csv')
    sample_data.to_csv(sample_path, index=False)
    logging.info(f"Sample data created at: {sample_path}")
    logging.info(f"Data shape: {sample_data.shape}")
    logging.info("Sample data columns:\n" + "\n".join(sample_data.columns))

# === Main Execution ===
if __name__ == '__main__':
    create_sample_data()
    logging.info("=== Sample Data Creation Completed ===")
