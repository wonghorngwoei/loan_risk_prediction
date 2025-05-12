import os
import joblib
import numpy as np
import pandas as pd

# === Load the Model ===
current_dir = os.path.dirname(os.path.abspath('__file__'))  
MODEL_DIR = os.path.join(current_dir, 'models')
model_path = os.path.join(MODEL_DIR, 'lightgbm_model.pkl')

if not os.path.exists(model_path):
    print("Model file not found. Please ensure you have trained the model first.")
    exit()

model = joblib.load(model_path)
print("Model loaded successfully.")

# === Feature Input ===
print("\nEnter the following details to make a prediction:")

features = [
    'originallyScheduledPaymentAmount', 'paymentAmount', 'state', 'apr',
    'loanAmount', 'identity_verification_reason_code', 'total_fraud_indicators',
    'leadCost', 'loanStatus_cleaned', 'installmentIndex', 'name_address_match',
    'max_ssns_per_bank_account', 'leadType', 'payFrequency', 'work_phone_listed_as_cellphone',
    'fpStatus', 'ssn_dob_match', 'invalid_driver_license_format',
    'driver_license_inconsistent_with_file', 'address_conflict_on_file',
    'overall_identity_match_result', 'paymentStatus', 'non_residential_address',
    'work_phone_listed_as_homephone', 'phone_inconsistent_with_state', 'phone_match_type_description',
    'ssn_name_match', 'non_residential_address_on_file', 'current_address_new_trade_only',
    'current_address_not_on_file', 'fees', 'phone_inconsistent_with_address', 'inquiries_gt_3_last_30days',
    'principal', 'phone_match_result', 'current_address_reported_lt_90days', 'credit_before_age_18',
    'ssn_reported_frequently_for_another', 'high_risk_inquiry_address', 'high_risk_address_on_file',
    'credit_before_ssn_issue_date', 'high_probability_ssn_belongs_to_another', 'age_younger_than_ssn_issue_date',
    'isCollection', 'unverified_ssn_issue_date', 'originated', 'paymentReturnCode', 'isFunded', 'approved'
]

# Categories for label encoding (same as training)
categories = {
    'state': ['CA', 'NJ', 'MO', 'WI', 'IL', 'MI', 'FL', 'SC', 'OH', 'NV', 'IN',
              'VA', 'OK', 'NC', 'MN', 'TX', 'TN', 'HI', 'MS', 'PA', 'KY', 'AL',
              'GA', 'NM', 'ID', 'ND', 'AZ', 'CO', 'NE', 'SD', 'LA', 'CT', 'KS',
              'DE', 'WY', 'UT', 'IA', 'AK', 'WA', 'RI', 'Others'],
    'loanStatus_cleaned': ['Paid Off', 'New Loan', 'Collection', 'Rejected', 'Withdrawn',
                           'Returned Item', 'Other', 'Voided', 'Charged Off', 'Unknown'],
    'name_address_match': ['match', 'partial', 'mismatch', 'unavailable', 'invalid'],
    'leadType': ['prescreen', 'lead', 'organic', 'bvMandatory', 'california',
                 'rc_returning', 'instant-offer', 'express', 'lionpay', 'repeat'],
    'payFrequency': ['B', 'W', 'S', 'M', 'I'],
    'work_phone_listed_as_cellphone': ['False', 'Unknown', 'True'],
    'fpStatus': ['Checked', 'Rejected', 'NotFunded', 'Skipped', 'No Payments',
                 'Cancelled', 'No Schedule', 'Pending'],
    'ssn_dob_match': ['invalid', 'match', 'partial', 'mismatch', 'unavailable'],
    'invalid_driver_license_format': ['Unknown', 'False', 'True'],
    'driver_license_inconsistent_with_file': ['Unknown', 'False', 'True'],
    'address_conflict_on_file': ['False', 'True'],
    'overall_identity_match_result': ['partial', 'match', 'other', 'mismatch'],
    'paymentStatus': ['Checked', 'Cancelled', 'Pending', 'Rejected', 'Skipped', 'Rejected Awaiting Retry'],
    'non_residential_address': ['False', 'True'],
    'work_phone_listed_as_homephone': ['False', 'Unknown', 'True'],
    'phone_inconsistent_with_state': ['False', 'Unknown', 'True'],
    'phone_match_type_description': ['(U) Unlisted', '(M) Mobile Phone', '(P) Pager',
                                     '(L) Last Name Only', '(FA) Full Name and Address',
                                     '(F) Full Name Only', '(LA) Last Name and Address', '(A) Address Only'],
    'ssn_name_match': ['match', 'mismatch', 'unavailable', 'partial'],
    'non_residential_address_on_file': ['False', 'True'],
    'current_address_new_trade_only': ['False', 'True'],
    'current_address_not_on_file': ['False', 'True'],
    'phone_inconsistent_with_address': ['False', 'True'],
    'inquiries_gt_3_last_30days': ['False', 'True'],
    'phone_match_result': ['invalid', 'unavailable', 'partial', 'match', 'mismatch'],
    'current_address_reported_lt_90days': ['False', 'True'],
    'credit_before_age_18': ['False', 'True'],
    'ssn_reported_frequently_for_another': ['False', 'True'],
    'high_risk_inquiry_address': ['False', 'True'],
    'high_risk_address_on_file': ['False', 'True'],
    'credit_before_ssn_issue_date': ['False', 'True'],
    'high_probability_ssn_belongs_to_another': ['False', 'True'],
    'age_younger_than_ssn_issue_date': ['False', 'True'],
    'isCollection': ['False', 'True'],
    'unverified_ssn_issue_date': ['False', 'True'],
    'originated': ['False', 'True'],
    'paymentReturnCode': ['R01', 'R02', 'R03', 'R04', 'R05', 'R06', 'R07', 'R08', 'R09', 'R10',
                        'R11', 'R12', 'R13', 'R14', 'R15', 'R16', 'R17', 'R20', 'R21', 'R22',
                        'R23', 'R24', 'R29', 'R31', 'R33'],
    'isFunded': ['False', 'True'],
    'approved': ['False', 'True']
}

input_data = []
for feature in features:
    if feature in categories:
        print(f"Options for {feature}: {categories[feature]}")
        while True:
            value = input(f"Enter value for {feature}: ").strip()
            if value in categories[feature]:
                input_data.append(categories[feature].index(value))
                break
            else:
                print("Invalid value. Please choose from the options.")
    else:
        while True:
            try:
                value = float(input(f"Enter value for {feature} (numeric): ").strip())
                input_data.append(value)
                break
            except ValueError:
                print("Invalid input. Please enter a numeric value.")

input_df = pd.DataFrame([input_data], columns=features)

# === Make Prediction ===
prediction_proba = model.predict(input_df)
prediction_class = np.argmax(prediction_proba, axis=1)[0]

# === Display Result ===
risk_classes = {0: "High Risk", 1: "Low Risk", 2: "Moderate Risk"}
print("\n=== Prediction Result ===")
print(f"Predicted Risk: {risk_classes[prediction_class]}")
