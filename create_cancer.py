import pandas as pd
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. Load Data
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target # 0=Malignant, 1=Benign in original sklearn

# 2. Select Top 5 Features for the Web App
# (Making it easier to type inputs in the form)
top_features = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness']
X = df[top_features]
y = df['target'] # We will keep 0=Malignant, 1=Benign for training, but flip text in app

# 3. Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 4. Neural Network
model = Sequential()
model.add(Dense(16, activation='relu', input_shape=(5,)))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 5. Train
print("Training Cancer Model...")
model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1)

# 6. Save
model.save('cancer_model.h5')
joblib.dump(scaler, 'cancer_scaler.pkl')
print("Saved cancer_model.h5 and cancer_scaler.pkl")