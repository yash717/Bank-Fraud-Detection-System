import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle


def load_model(file_path):
    # Load the pre-trained XGBoost model
    with open(r'C:\Users\Yash\Desktop\fraud_detection_project\django_backend\fraud_detection\bank_apis\xgboost_model.pkl', 'rb') as model_file:
        loaded_model = pickle.load(model_file)

    return loaded_model


new_input_data = {
    'Transaction_Amount': [1500],
    'User_Account_ID': [104],
    'Account_Creation_Date': ['2022-11-15'],
    'Payment_Method': ['Credit Card'],
    'Billing_Location': ['Bangalore'],
    'Shipping_Location': ['Hyderabad'],
    'Device_IP_Address': ['192.168.1.40'],
    'Session_Duration': ['500 seconds'],
    'Frequency_of_Transactions': [7],
    'Time_Between_Transactions': ['80 seconds'],
    'Unusual_Time_of_Transaction': [0],
    'Unusual_Transaction_Amounts': [0],
    'IP_Address_History': ['192.168.1.40']
}

#give file path for csv file in the same folder


df = pd.read_csv("C:\\Users\\Yash\\Desktop\\fraud_detection_project\\django_backend\\fraud_detection\\bank_apis\\transaction_detail.csv")


X = df.drop('Fraud_Label', axis=1)
y = df['Fraud_Label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


def preprocess_input(new_input_data):
    new_input = pd.DataFrame([new_input_data])
    new_input['Account_Creation_Date'] = pd.to_datetime(
        new_input['Account_Creation_Date'])
    new_input['Session_Duration'] = new_input['Session_Duration'].str.extract(
        '(\d+)').astype(int)
    new_input['Time_Between_Transactions'] = new_input['Time_Between_Transactions'].str.extract(
        '(\d+)').astype(int)
    new_input = pd.get_dummies(new_input)
    
    # Ensure the columns in new_input match X_train columns
    missing_cols = set(X_train.columns) - set(new_input.columns)
    for col in missing_cols:
        new_input[col] = 0
        
    extra_cols = set(new_input.columns) - set(X_train.columns)
    new_input = new_input[X_train.columns]  # Align columns with X_train
    
    return new_input



def predict_fraud(loaded_model, new_input):
    # Preprocess the new input data
    new_input = preprocess_input(new_input)

    # Make predictions using the loaded model
    fraud_prediction = loaded_model.predict(new_input)
    return fraud_prediction
