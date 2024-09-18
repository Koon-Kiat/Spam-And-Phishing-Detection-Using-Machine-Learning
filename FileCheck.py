import pickle
import joblib
from joblib import load

def check_pickle_file(file_path):
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            print("File loaded successfully.")
            print("Data:", data)  # Optionally print or inspect the data
            return data
    except Exception as e:
        print(f"Error loading file: {e}")

def check_model_file(file_path):
    try:
        # Attempt to load the model with joblib
        model = joblib.load(file_path)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model with joblib: {e}")

    try:
        # Attempt to load the model with pickle if joblib fails
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        print("Model loaded successfully using pickle.")
        return model
    except Exception as e:
        print(f"Error loading model with pickle: {e}")


# Replace 'filename.pkl' with your actual file path
check_pickle_file(r'C:\Users\Koon Kiat\OneDrive\Cloud\Projects\Phishing Email Detection\Models & Parameters\ensemble_model_fold_1.pkl')
check_model_file(r'C:\Users\Koon Kiat\OneDrive\Cloud\Projects\Phishing Email Detection\Models & Parameters\ensemble_model_fold_1.pkl')