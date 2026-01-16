from flask import Flask, render_template, request
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

try:
    model = load_model('cancer_model.h5')
    scaler = joblib.load('cancer_scaler.pkl')
except:
    print("Run create_cancer_model.py first!")

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = ""
    if request.method == 'POST':
        try:
            # Inputs matching our Top 5 features
            radius = float(request.form['radius'])
            texture = float(request.form['texture'])
            perimeter = float(request.form['perimeter'])
            area = float(request.form['area'])
            smoothness = float(request.form['smoothness'])

            features = np.array([[radius, texture, perimeter, area, smoothness]])
            features_scaled = scaler.transform(features)

            prediction = model.predict(features_scaled)
            val = prediction[0][0]

            # In Sklearn dataset: 0 = Malignant, 1 = Benign
            # We output probability of being 1 (Benign)
            if val < 0.5:
                res = "MALIGNANT (CANCER)"
                color = "red"
            else:
                res = "BENIGN (SAFE)"
                color = "green"
            
            prediction_text = res

        except Exception as e:
            prediction_text = f"Error: {e}"

    return render_template('index.html', prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)