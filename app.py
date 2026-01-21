import os
import joblib
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# Load model and scaler from the /model/ folder
# Using joblib as it is the standard for .pkl files in Scikit-Learn
try:
    model_path = os.path.join('model', 'breast_cancer_model.pkl')
    # If you saved both model and scaler in one file:
    data_pack = joblib.load(model_path)
    
    # Check if the pkl contains a dictionary or just the model
    if isinstance(data_pack, dict):
        model = data_pack['model']
        scaler = data_pack['scaler']
    else:
        model = data_pack
        # If scaler is separate, load it here:
        scaler = joblib.load(os.path.join('model', 'cancer_scaler.pkl'))
except Exception as e:
    print(f"Error loading model: {e}")

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = ""
    if request.method == 'POST':
        try:
            # 1. Get inputs from the HTML form
            input_data = [
                float(request.form['radius']),
                float(request.form['texture']),
                float(request.form['perimeter']),
                float(request.form['area']),
                float(request.form['smoothness'])
            ]
            
            # 2. Convert to numpy array and scale
            features = np.array([input_data])
            features_scaled = scaler.transform(features)
            
            # 3. Predict
            prediction = model.predict(features_scaled)
            
            # 4. Handle prediction result (assuming 0=Malignant, 1=Benign)
            # If using Neural Network, prediction might be a probability (e.g., 0.8)
            res_val = prediction[0]
            if hasattr(res_val, "__len__"): res_val = res_val[0] # Handle NN output arrays
            
            if res_val < 0.5:
                prediction_text = "MALIGNANT"
            else:
                prediction_text = "BENIGN"

        except Exception as e:
            prediction_text = f"Error: {e}"

    return render_template('index.html', prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)