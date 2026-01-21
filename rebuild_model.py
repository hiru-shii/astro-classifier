import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("--- STARTING REAL MODEL TRAINING ---")

# 1. LOAD YOUR REAL DATA
# Ensure 'cleaned_star_galaxy_quasar_data.csv' is in the same folder!
try:
    print("Loading 'cleaned_star_galaxy_quasar_data.csv'...")
    df = pd.read_csv('cleaned_star_galaxy_quasar_data.csv')
    print(f"Data loaded successfully: {len(df)} rows.")
except FileNotFoundError:
    print("ERROR: File not found. Please make sure the CSV is in this folder.")
    exit()

# 2. FEATURE ENGINEERING
# Even if your CSV already has these, calculating them again ensures 
# they match the App's logic exactly.
print("Verifying features...")
# We use .copy() to avoid SettingWithCopy warnings
df = df.copy() 

# Calculate Colors (The Logic)
df['u_g'] = df['u'] - df['g']
df['g_r'] = df['g'] - df['r']
df['r_i'] = df['r'] - df['i']
df['i_z'] = df['i'] - df['z']
df['z_w1'] = df['z'] - df['w1']

# Define Inputs (X) and Target (y)
features = ['u_g', 'g_r', 'r_i', 'i_z', 'z_w1', 'w1', 'w2']
X = df[features]
y = df['class_label']

# 3. SPLIT DATA
# We split it so we can print the accuracy at the end
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. TRAIN MODEL
print("Training Random Forest model (this may take a moment)...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. VALIDATE
acc = accuracy_score(y_test, model.predict(X_test))
print(f"Model Trained! Accuracy on test set: {acc*100:.2f}%")

# 6. SAVE MODEL
print("Saving model to 'astro_classifier_model.pkl'...")
joblib.dump(model, 'astro_classifier_model.pkl')

print("--- SUCCESS! Real model saved. ---")