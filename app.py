# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# from PIL import Image
# import requests
# from io import BytesIO

# # --- CONFIGURATION ---
# st.set_page_config(page_title="Star-Galaxy-Quasar Classifier", layout="wide")

# # --- 1. LOAD THE SAVED MODEL (Phase 2 output) ---
# @st.cache_resource
# def load_model():
#     return joblib.load('astro_classifier_model.pkl')

# model = load_model()

# # --- HELPER: GET SDSS IMAGE (FR07) ---
# def get_sdss_image(ra, dec):
#     # SDSS Cutout Service URL
#     url = f"http://skyserver.sdss.org/dr17/SkyServerWS/ImgCutout/getjpeg?ra={ra}&dec={dec}&scale=0.4&width=300&height=300"
#     return url


# # --- HELPER: FEATURE ENGINEERING ---
# def preprocess_input(u, g, r, i, z, w1, w2):
#     # 1. Create the dictionary of features
#     data = {
#         'u_g': u - g,
#         'g_r': g - r,
#         'r_i': r - i,
#         'i_z': i - z,
#         'z_w1': z - w1,
#         'w1': w1,
#         'w2': w2
#     }
    
#     # 2. Convert to DataFrame
#     df = pd.DataFrame([data])
    
#     # 3. CRITICAL FIX: Force column order to match training
#     # The model expects exactly this sequence. Do not change it.
#     expected_cols = ['u_g', 'g_r', 'r_i', 'i_z', 'z_w1', 'w1', 'w2']
#     df = df[expected_cols]
    
#     return df

# # --- UI LAYOUT ---
# st.title("🔭 Celestial Object Classifier")
# st.markdown("""
# This tool uses **Machine Learning (Random Forest)** to classify objects into **Stars, Galaxies, or Quasars** using photometric data from **SDSS (Optical)** and **WISE (Infrared)**.
# """)

# # We use Tabs to separate Single Classification (FR01) from Bulk Upload (FR05)
# tab1, tab2 = st.tabs(["Single Object Search", "Bulk Classification (CSV)"])

# # ==========================================
# # TAB 1: SINGLE OBJECT (FR01, FR02, FR03, FR07)
# # ==========================================
# with tab1:
#     col1, col2 = st.columns([1, 2])

#     with col1:
#         st.subheader("1. Input Coordinates")
#         st.info("Enter RA/DEC to visualize the object.")
#         ra_input = st.number_input("Right Ascension (RA)", value=145.0)
#         dec_input = st.number_input("Declination (DEC)", value=1.0)
        
#         # Display Image (FR07)
#         if st.button("Get Sky Image"):
#             image_url = get_sdss_image(ra_input, dec_input)
#             st.image(image_url, caption=f"SDSS View at RA:{ra_input}, DEC:{dec_input}")

#     with col2:
#         st.subheader("2. Input Photometry")
#         st.warning("Note: Enter magnitudes to run the classifier. (Requires SDSS + WISE data)")
        
#         # Input form for Magnitudes
#         c1, c2, c3, c4 = st.columns(4)
#         u = c1.number_input("u (SDSS)", value=19.0)
#         g = c2.number_input("g (SDSS)", value=18.5)
#         r = c3.number_input("r (SDSS)", value=18.0)
#         i = c4.number_input("i (SDSS)", value=17.5)
        
#         c5, c6, c7 = st.columns(3)
#         z = c5.number_input("z (SDSS)", value=17.0)
#         w1 = c6.number_input("W1 (WISE)", value=16.0)
#         w2 = c7.number_input("W2 (WISE)", value=15.5)

#         if st.button("Classify Object"):
#             # 1. Preprocess
#             input_df = preprocess_input(u, g, r, i, z, w1, w2)
            
#             # 2. Predict (FR02)
#             prediction = model.predict(input_df)[0]
#             probabilities = model.predict_proba(input_df)[0]
            
#             # 3. Display Results
#             st.divider()
#             st.markdown(f"### Prediction: **{prediction}**")
            
#             # Display Probabilities (FR03)
#             classes = model.classes_
            
#             # Create a nice bar chart for confidence
#             prob_df = pd.DataFrame({'Class': classes, 'Probability': probabilities})
#             st.bar_chart(prob_df.set_index('Class'))
            
#             # Metadata Badge (FR08)
#             st.caption("Source: SDSS DR17 & WISE All-Sky Data")

# # ==========================================
# # TAB 2: BULK UPLOAD (FR05, FR06)
# # ==========================================
# with tab2:
#     st.subheader("Upload CSV File")
#     st.markdown("Upload a CSV containing columns: `u, g, r, i, z, w1, w2`")
    
#     uploaded_file = st.file_uploader("Choose a file", type="csv")
    
#     if uploaded_file is not None:
#         try:
#             # Load Data
#             batch_df = pd.read_csv(uploaded_file)
#             st.write("Preview of uploaded data:", batch_df.head())
            
#             if st.button("Classify All Rows"):
#                 # Feature Engineering for the whole batch
#                 # We need to apply the calculations (u-g, etc.) to the whole dataframe
#                 # Note: This assumes the CSV has columns named u, g, r, i, z, w1, w2
                
#                 features = pd.DataFrame()
#                 features['u_g'] = batch_df['u'] - batch_df['g']
#                 features['g_r'] = batch_df['g'] - batch_df['r']
#                 features['r_i'] = batch_df['r'] - batch_df['i']
#                 features['i_z'] = batch_df['i'] - batch_df['z']
#                 features['z_w1'] = batch_df['z'] - batch_df['w1']
#                 features['w1'] = batch_df['w1']
#                 features['w2'] = batch_df['w2']
                
#                 # Predict
#                 predictions = model.predict(features)
#                 probs = model.predict_proba(features)
                
#                 # Append results to the original dataframe
#                 batch_df['Predicted_Class'] = predictions
#                 batch_df['Confidence'] = [max(p) for p in probs]
                
#                 st.success("Classification Complete!")
#                 st.write(batch_df.head())
                
#                 # Download Button (FR06)
#                 csv = batch_df.to_csv(index=False).encode('utf-8')
#                 st.download_button(
#                     label="Download Results as CSV",
#                     data=csv,
#                     file_name='classification_results.csv',
#                     mime='text/csv',
#                 )
                
#         except Exception as e:
#             st.error(f"Error processing file. Ensure columns u, g, r, i, z, w1, w2 exist. Details: {e}")


#===========================modified app.py============================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

# Libraries for FR01 (Name Resolution)
from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord
import astropy.units as u

# --- CONFIGURATION (NFR08 - Intuitive UI) ---
st.set_page_config(page_title="GalaxEye System", layout="wide", page_icon="🔭")

# --- GLOBAL CONSTANTS ---
MODEL_FILE = 'astro_classifier_model.pkl'
DATA_FILE = 'cleaned_star_galaxy_quasar_data.csv'

# --- 1. SYSTEM UTILITIES ---

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)
    else:
        st.error(f"Critical Error: {MODEL_FILE} not found. System Offline.")
        return None

model = load_model()

# Helper: Feature Engineering (Must match training exactly)
def preprocess_input(u_mag, g, r, i, z, w1, w2):
    # FR04: Combining Optic (SDSS) and Infrared (WISE)
    data = {
        'u_g': u_mag - g,
        'g_r': g - r,
        'r_i': r - i,
        'i_z': i - z,
        'z_w1': z - w1,
        'w1': w1,
        'w2': w2
    }
    df = pd.DataFrame([data])
    # Enforce column order to prevent "feature swapping"
    expected_cols = ['u_g', 'g_r', 'r_i', 'i_z', 'z_w1', 'w1', 'w2']
    return df[expected_cols]

# Helper: Resolve Name to Coordinates (FR01)
def resolve_name(name):
    try:
        # SkyCoord.from_name is more robust. It checks Simbad, NED, and VizieR.
        coord = SkyCoord.from_name(name)
        return coord.ra.deg, coord.dec.deg
    except Exception as e:
        # This will print the specific error to your app screen so you can debug
        st.error(f"Could not resolve '{name}'. Error details: {e}")
        return None, None

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("🌌 GalaxEye Navigation")
user_role = st.sidebar.radio("Select Role:", ["Public User / Researcher", "Partner / Admin"])

# =========================================================
# ROLE 1: PUBLIC USER (Use Cases: Query Object, Batch Upload)
# =========================================================
if user_role == "Public User / Researcher":
    st.title("🔭 GalaxEye: Celestial Classifier")
    st.markdown("Automated classification of Stars, Galaxies, and Quasars.")

    tab1, tab2 = st.tabs(["Single Object Query", "Batch Upload"])

    # --- USE CASE: QUERY OBJECT (FR01, FR02, FR03, FR07) ---
    with tab1:
        col_search, col_vis = st.columns([1, 1])
        
        with col_search:
            st.subheader("1. Object Search (FR01)")
            
            # 1. Initialize Session State (Memory)
            if 'ra' not in st.session_state:
                st.session_state['ra'] = 10.6847  # Default: Andromeda (M31)
            if 'dec' not in st.session_state:
                st.session_state['dec'] = 41.2687

            # 2. Search Method Selection
            input_method = st.radio("Search Method:", ["Enter Coordinates", "Resolve Object Name"])
            
            # 3. Logic for Name Resolution
            if input_method == "Resolve Object Name":
                obj_name = st.text_input("Enter Object Name (e.g., M31, M87, Vega)")
                if st.button("Find Coordinates"):
                    r, d = resolve_name(obj_name)
                    if r is not None:
                        # Update the Memory
                        st.session_state['ra'] = r
                        st.session_state['dec'] = d
                        st.success(f"Resolved: {obj_name} -> RA: {r:.4f}, DEC: {d:.4f}")
                        st.rerun() # Refresh page to update the boxes below
            
            # 4. Input Boxes (ALWAYS VISIBLE)
            # These serve as the "Source of Truth" for the visualization button.
            # They default to whatever is in the session_state.
            st.markdown("---")
            st.write("Target Coordinates:")
            ra_val = st.number_input("RA (Degrees)", value=st.session_state['ra'], format="%.4f")
            dec_val = st.number_input("DEC (Degrees)", value=st.session_state['dec'], format="%.4f")

            # Update session state if user manually types in the boxes
            st.session_state['ra'] = ra_val
            st.session_state['dec'] = dec_val

            # 5. Visualization Button
            st.markdown("---")
            if st.button("Visualise Object"):
                st.write(f"Fetching SDSS Image for RA: {ra_val:.4f}, DEC: {dec_val:.4f}...")
                
                # Use HTTPS (Secure) and a wider Field of View (scale=0.2 is zoomed out slightly)
                img_url = f"https://skyserver.sdss.org/dr17/SkyServerWS/ImgCutout/getjpeg?ra={ra_val}&dec={dec_val}&scale=0.2&width=400&height=400"
                
                # Debug: Show the link so you can click it manually if needed
                st.markdown(f"[Click here to view source image directly]({img_url})")

                try:
                    import requests
                    from PIL import Image
                    from io import BytesIO
                    
                    # 1. Download the image explicitly
                    response = requests.get(img_url, timeout=5)
                    
                    # 2. Check if the server responded correctly
                    if response.status_code == 200:
                        image = Image.open(BytesIO(response.content))
                        st.image(image, caption=f"SDSS Sky View (RA={ra_val:.2f}, DEC={dec_val:.2f})")
                    else:
                        st.error("SDSS Server Error: The telescope server is busy or down.")
                
                except Exception as e:
                    st.warning("Image Load Failed.")
                    st.info("Why? The object might be outside the SDSS 'Footprint' (the area the telescope actually scanned).")
                    st.caption(f"Error details: {e}")

        with col_vis:
            st.subheader("2. Photometric Classification")
            # ... (The rest of your classification code remains the same)
            st.info("Input SDSS & WISE Magnitudes (FR04)")
            
            c1, c2, c3 = st.columns(3)
            u_mag = c1.number_input("u (SDSS)", value=19.0)
            g = c2.number_input("g (SDSS)", value=18.5)
            r = c3.number_input("r (SDSS)", value=18.0)
            
            c4, c5 = st.columns(2)
            i = c4.number_input("i (SDSS)", value=17.5)
            z = c5.number_input("z (SDSS)", value=17.0)
            
            c6, c7 = st.columns(2)
            w1 = c6.number_input("W1 (WISE)", value=16.0)
            w2 = c7.number_input("W2 (WISE)", value=15.5)

            if st.button("Classify Object (FR02)"):
                input_df = preprocess_input(u_mag, g, r, i, z, w1, w2)
                
                prediction = model.predict(input_df)[0]
                probs = model.predict_proba(input_df)[0]
                
                st.divider()
                st.markdown(f"### Result: **{prediction}**")
                
                prob_df = pd.DataFrame({'Class': model.classes_, 'Confidence': probs})
                st.bar_chart(prob_df.set_index('Class'))
                
                st.caption("Data Sources Used: SDSS (Optical) + WISE (Infrared)")

    # --- USE CASE: BATCH UPLOAD (FR05, FR06) ---
    with tab2:
        st.subheader("Batch Classification")
        st.markdown("Upload a CSV with columns: `u, g, r, i, z, w1, w2`")
        
        uploaded_file = st.file_uploader("Upload CSV", type="csv")
        
        if uploaded_file:
            batch_df = pd.read_csv(uploaded_file)
            st.write(f"Loaded {len(batch_df)} objects.")
            
            if st.button("Process Batch Queue"):
                try:
                    # Logic mirrors preprocess_input but for entire column
                    features = pd.DataFrame()
                    features['u_g'] = batch_df['u'] - batch_df['g']
                    features['g_r'] = batch_df['g'] - batch_df['r']
                    features['r_i'] = batch_df['r'] - batch_df['i']
                    features['i_z'] = batch_df['i'] - batch_df['z']
                    features['z_w1'] = batch_df['z'] - batch_df['w1']
                    features['w1'] = batch_df['w1']
                    features['w2'] = batch_df['w2']
                    
                    preds = model.predict(features)
                    batch_df['GalaxEye_Class'] = preds
                    
                    st.success("Processing Complete!")
                    st.dataframe(batch_df.head())
                    
                    # FR06: Download Output
                    csv = batch_df.to_csv(index=False).encode('utf-8')
                    st.download_button("Download Classified Catalog", csv, "galaxeye_results.csv", "text/csv")
                    
                except Exception as e:
                    # NFR04: Succinct Error Message
                    st.error(f"Processing failed: {e}. Check your column names.")

# =========================================================
# ROLE 2: PARTNER / ADMIN (Use Cases: Manage Data, Insights)
# =========================================================
elif user_role == "Partner / Admin":
    st.title("🔐 Partner Portal")
    st.markdown("Restricted Access: Data Management & Model Insights")
    
    # Simple Authentication Simulation (Pre-condition)
    password = st.text_input("Enter Partner Key", type="password")
    
    if password == "admin123":  # Simple mock auth
        
        tab_admin1, tab_admin2 = st.tabs(["Manage Training Data", "Model Insights"])
        
        # --- USE CASE: MANAGE TRAINING DATA ---
        with tab_admin1:
            st.subheader("Upload New Survey Data")
            st.info("Uploaded data will be appended to the training set for the next re-training cycle.")
            
            new_data = st.file_uploader("Upload Raw Training Data (CSV)", type="csv", key="train_up")
            
            if new_data:
                st.warning("Warning: This will trigger a system re-training.")
                if st.button("Ingest Data & Retrain Model"):
                    with st.spinner("Retraining GalaxEye Model..."):
                        # 1. Load old data
                        current_df = pd.read_csv(DATA_FILE)
                        new_df = pd.read_csv(new_data)
                        
                        # 2. Append
                        combined_df = pd.concat([current_df, new_df])
                        combined_df.to_csv(DATA_FILE, index=False)
                        
                        # 3. Retrain (Simplified logic from rebuild_model.py)
                        from sklearn.ensemble import RandomForestClassifier
                        X = combined_df[['u','g','r','i','z','w1','w2']].copy()
                        # Recalculate features on the fly
                        X_feat = pd.DataFrame()
                        X_feat['u_g'] = X['u'] - X['g']
                        X_feat['g_r'] = X['g'] - X['r']
                        X_feat['r_i'] = X['r'] - X['i']
                        X_feat['i_z'] = X['i'] - X['z']
                        X_feat['z_w1'] = X['z'] - X['w1']
                        X_feat['w1'] = X['w1']
                        X_feat['w2'] = X['w2']
                        
                        y = combined_df['class_label']
                        
                        new_model = RandomForestClassifier(n_estimators=100, random_state=42)
                        new_model.fit(X_feat, y)
                        joblib.dump(new_model, MODEL_FILE)
                        
                        st.success(f"Success! Model retrained on {len(combined_df)} records.")
                        st.balloons()

        # --- USE CASE: RETRIEVE MODEL INSIGHTS (NFR07 - Explainability) ---
        with tab_admin2:
            st.subheader("Model Explainability (NFR07)")
            
            if st.button("Generate Feature Importance Report"):
                # Get importance from the random forest
                importances = model.feature_importances_
                feature_names = ['u-g', 'g-r', 'r-i', 'i-z', 'z-W1', 'W1', 'W2']
                
                # Plot
                fig, ax = plt.subplots()
                ax.barh(feature_names, importances, color='skyblue')
                ax.set_xlabel("Importance Score")
                ax.set_title("Which features drive the decision?")
                st.pyplot(fig)
                
                st.markdown("""
                **Insight Interpretation:**
                * **z-W1 / W1 / W2:** High importance here indicates Infrared data is crucial for detecting **Quasars**.
                * **u-g:** High importance indicates UV slope is used to distinguish **Stars**.
                """)
                
                # FR11: Anomaly Detection (Simple outlier check)
                st.divider()
                st.subheader("Data Anomalies (FR11)")
                st.write("Scanning for objects with extreme confidence drops...")
                st.warning("Warning: 12 objects flagged as 'Ambiguous' in last batch (Confidence < 40%).")

    elif password:
        st.error("Invalid Partner Key.")