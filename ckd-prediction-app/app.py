import os
import pickle
from flask import Flask, request, render_template
import numpy as np
from utils.preprocess import preprocess_input

app = Flask(__name__)

# Get the current working directory
current_directory = os.path.dirname(os.path.abspath(__file__))

# Load models with the correct path
try:
    log_reg_model_path = os.path.join(current_directory, 'logistic_regression.pkl')
    rf_clf_model_path = os.path.join(current_directory, 'random_forest.pkl')
    svc_clf_model_path = os.path.join(current_directory, 'svm.pkl')

    with open(log_reg_model_path, 'rb') as f:
        log_reg_model = pickle.load(f)

    with open(rf_clf_model_path, 'rb') as f:
        rf_clf_model = pickle.load(f)

    with open(svc_clf_model_path, 'rb') as f:
        svc_clf_model = pickle.load(f)

except (FileNotFoundError, EOFError) as e:
    print(f"Error loading model: {e}")
    log_reg_model, rf_clf_model, svc_clf_model = None, None, None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if log_reg_model is None or rf_clf_model is None or svc_clf_model is None:
        return "Models are not loaded properly. Check server logs for details."

    try:
        # Get form data
        features = [float(x) for x in request.form.values()]
        features = np.array(features).reshape(1, -1)

        # Preprocess input
        processed_features = preprocess_input(features)

        # Make predictions
        log_reg_pred = log_reg_model.predict(processed_features)
        rf_clf_pred = rf_clf_model.predict(processed_features)
        svc_clf_pred = svc_clf_model.predict(processed_features)

        # Prepare results
        results = {
            'Logistic Regression': 'Positive' if log_reg_pred[0] else 'Negative',
            'Random Forest': 'Positive' if rf_clf_pred[0] else 'Negative',
            'Support Vector Classifier': 'Positive' if svc_clf_pred[0] else 'Negative'
        }

        return render_template('index.html', results=results)
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)
