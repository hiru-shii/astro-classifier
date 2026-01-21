import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image
import requests
from io import BytesIO

# --- CONFIGURATION ---
st.set_page_config(page_title="Star-Galaxy-Quasar Classifier", layout="wide")

# --- 1. LOAD THE SAVED MODEL (Phase 2 output) ---
@st.cache_resource
def load_model():
    return joblib.load('astro_classifier_model.pkl')

model = load_model()

# --- HELPER: GET SDSS IMAGE (FR07) ---
def get_sdss_image(ra, dec):
    # SDSS Cutout Service URL
    url = f"http://skyserver.sdss.org/dr17/SkyServerWS/ImgCutout/getjpeg?ra={ra}&dec={dec}&scale=0.4&width=300&height=300"
    return url


# --- HELPER: FEATURE ENGINEERING ---
def preprocess_input(u, g, r, i, z, w1, w2):
    # 1. Create the dictionary of features
    data = {
        'u_g': u - g,
        'g_r': g - r,
        'r_i': r - i,
        'i_z': i - z,
        'z_w1': z - w1,
        'w1': w1,
        'w2': w2
    }
    
    # 2. Convert to DataFrame
    df = pd.DataFrame([data])
    
    # 3. CRITICAL FIX: Force column order to match training
    # The model expects exactly this sequence. Do not change it.
    expected_cols = ['u_g', 'g_r', 'r_i', 'i_z', 'z_w1', 'w1', 'w2']
    df = df[expected_cols]
    
    return df

# --- UI LAYOUT ---
st.title("🔭 Celestial Object Classifier")
st.markdown("""
This tool uses **Machine Learning (Random Forest)** to classify objects into **Stars, Galaxies, or Quasars** using photometric data from **SDSS (Optical)** and **WISE (Infrared)**.
""")

# We use Tabs to separate Single Classification (FR01) from Bulk Upload (FR05)
tab1, tab2 = st.tabs(["Single Object Search", "Bulk Classification (CSV)"])

# ==========================================
# TAB 1: SINGLE OBJECT (FR01, FR02, FR03, FR07)
# ==========================================
with tab1:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("1. Input Coordinates")
        st.info("Enter RA/DEC to visualize the object.")
        ra_input = st.number_input("Right Ascension (RA)", value=145.0)
        dec_input = st.number_input("Declination (DEC)", value=1.0)
        
        # Display Image (FR07)
        if st.button("Get Sky Image"):
            image_url = get_sdss_image(ra_input, dec_input)
            st.image(image_url, caption=f"SDSS View at RA:{ra_input}, DEC:{dec_input}")

    with col2:
        st.subheader("2. Input Photometry")
        st.warning("Note: Enter magnitudes to run the classifier. (Requires SDSS + WISE data)")
        
        # Input form for Magnitudes
        c1, c2, c3, c4 = st.columns(4)
        u = c1.number_input("u (SDSS)", value=19.0)
        g = c2.number_input("g (SDSS)", value=18.5)
        r = c3.number_input("r (SDSS)", value=18.0)
        i = c4.number_input("i (SDSS)", value=17.5)
        
        c5, c6, c7 = st.columns(3)
        z = c5.number_input("z (SDSS)", value=17.0)
        w1 = c6.number_input("W1 (WISE)", value=16.0)
        w2 = c7.number_input("W2 (WISE)", value=15.5)

        if st.button("Classify Object"):
            # 1. Preprocess
            input_df = preprocess_input(u, g, r, i, z, w1, w2)
            
            # 2. Predict (FR02)
            prediction = model.predict(input_df)[0]
            probabilities = model.predict_proba(input_df)[0]
            
            # 3. Display Results
            st.divider()
            st.markdown(f"### Prediction: **{prediction}**")
            
            # Display Probabilities (FR03)
            classes = model.classes_
            
            # Create a nice bar chart for confidence
            prob_df = pd.DataFrame({'Class': classes, 'Probability': probabilities})
            st.bar_chart(prob_df.set_index('Class'))
            
            # Metadata Badge (FR08)
            st.caption("Source: SDSS DR17 & WISE All-Sky Data")

# ==========================================
# TAB 2: BULK UPLOAD (FR05, FR06)
# ==========================================
with tab2:
    st.subheader("Upload CSV File")
    st.markdown("Upload a CSV containing columns: `u, g, r, i, z, w1, w2`")
    
    uploaded_file = st.file_uploader("Choose a file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Load Data
            batch_df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:", batch_df.head())
            
            if st.button("Classify All Rows"):
                # Feature Engineering for the whole batch
                # We need to apply the calculations (u-g, etc.) to the whole dataframe
                # Note: This assumes the CSV has columns named u, g, r, i, z, w1, w2
                
                features = pd.DataFrame()
                features['u_g'] = batch_df['u'] - batch_df['g']
                features['g_r'] = batch_df['g'] - batch_df['r']
                features['r_i'] = batch_df['r'] - batch_df['i']
                features['i_z'] = batch_df['i'] - batch_df['z']
                features['z_w1'] = batch_df['z'] - batch_df['w1']
                features['w1'] = batch_df['w1']
                features['w2'] = batch_df['w2']
                
                # Predict
                predictions = model.predict(features)
                probs = model.predict_proba(features)
                
                # Append results to the original dataframe
                batch_df['Predicted_Class'] = predictions
                batch_df['Confidence'] = [max(p) for p in probs]
                
                st.success("Classification Complete!")
                st.write(batch_df.head())
                
                # Download Button (FR06)
                csv = batch_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name='classification_results.csv',
                    mime='text/csv',
                )
                
        except Exception as e:
            st.error(f"Error processing file. Ensure columns u, g, r, i, z, w1, w2 exist. Details: {e}")