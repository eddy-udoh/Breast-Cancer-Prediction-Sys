import os
import joblib
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# --- GLOBAL LOADING LOGIC ---
# We define these as None first so the app doesn't crash immediately if files are missing
model = None
scaler = None

# Update filenames to match exactly what you have in your /model/ folder
model_filename = 'breast_cancer_model.pkl' 

def load_saved_objects():
    global model, scaler
    try:
        # Step 1: Locate the file inside the /model/ directory
        model_path = os.path.join(os.getcwd(), 'model', model_filename)
        
        # Step 2: Load the pickle file
        data_pack = joblib.load(model_path)
        
        # Step 3: Extract model and scaler 
        # (Assuming they were saved together in a dictionary)
        if isinstance(data_pack, dict):
            model = data_pack.get('model')
            scaler = data_pack.get('scaler')
        else:
            model = data_pack
            # If scaler is in a different file, load it here:
            scaler_path = os.path.join(os.getcwd(), 'model', 'cancer_scaler.pkl')
            scaler = joblib.load(scaler_path)
            
        print("Model and Scaler loaded successfully!")
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load model or scaler. {e}")

# Run the loader
load_saved_objects()

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = ""
    if request.method == 'POST':
        # Safety check: if scaler didn't load, we can't predict
        if model is None or scaler is None:
            return render_template('index.html', prediction_text="Error: Model files not loaded on server.")

        try:
            # Get values from your HTML form
            input_data = [
                float(request.form['radius']),
                float(request.form['texture']),
                float(request.form['perimeter']),
                float(request.form['area']),
                float(request.form['smoothness'])
            ]
            
            # Scale and Predict
            features = np.array([input_data])
            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)
            
            # Logic: 0 = Malignant, 1 = Benign
            prediction_text = "BENIGN" if prediction[0] == 1 else "MALIGNANT"

        except Exception as e:
            prediction_text = f"Input Error: {e}"

    return render_template('index.html', prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)