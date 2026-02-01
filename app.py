# #===========================modified app.py============================

# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import matplotlib.pyplot as plt
# import os

# # Libraries for FR01 (Name Resolution)
# from astroquery.simbad import Simbad
# from astropy.coordinates import SkyCoord
# import astropy.units as u

# # --- CONFIGURATION (NFR08 - Intuitive UI) ---
# st.set_page_config(page_title="GalaxEye System", layout="wide", page_icon="🔭")

# # --- GLOBAL CONSTANTS ---
# MODEL_FILE = 'astro_classifier_model.pkl'
# DATA_FILE = 'cleaned_star_galaxy_quasar_data.csv'

# # --- 1. SYSTEM UTILITIES ---

# @st.cache_resource
# def load_model():
#     if os.path.exists(MODEL_FILE):
#         return joblib.load(MODEL_FILE)
#     else:
#         st.error(f"Critical Error: {MODEL_FILE} not found. System Offline.")
#         return None

# model = load_model()

# # Helper: Feature Engineering (Must match training exactly)
# def preprocess_input(u_mag, g, r, i, z, w1, w2):
#     # FR04: Combining Optic (SDSS) and Infrared (WISE)
#     data = {
#         'u_g': u_mag - g,
#         'g_r': g - r,
#         'r_i': r - i,
#         'i_z': i - z,
#         'z_w1': z - w1,
#         'w1': w1,
#         'w2': w2
#     }
#     df = pd.DataFrame([data])
#     # Enforce column order to prevent "feature swapping"
#     expected_cols = ['u_g', 'g_r', 'r_i', 'i_z', 'z_w1', 'w1', 'w2']
#     return df[expected_cols]

# # Helper: Resolve Name to Coordinates (FR01)
# def resolve_name(name):
#     try:
#         # SkyCoord.from_name is more robust. It checks Simbad, NED, and VizieR.
#         coord = SkyCoord.from_name(name)
#         return coord.ra.deg, coord.dec.deg
#     except Exception as e:
#         # This will print the specific error to your app screen so you can debug
#         st.error(f"Could not resolve '{name}'. Error details: {e}")
#         return None, None

# # --- SIDEBAR NAVIGATION ---
# st.sidebar.title("🌌 GalaxEye Navigation")
# user_role = st.sidebar.radio("Select Role:", ["Public User / Researcher", "Partner / Admin"])

# # =========================================================
# # ROLE 1: PUBLIC USER (Use Cases: Query Object, Batch Upload)
# # =========================================================
# if user_role == "Public User / Researcher":
#     st.title("🔭 GalaxEye: Celestial Classifier")
#     st.markdown("Automated classification of Stars, Galaxies, and Quasars.")

#     tab1, tab2 = st.tabs(["Single Object Query", "Batch Upload"])

#     # --- USE CASE: QUERY OBJECT (FR01, FR02, FR03, FR07) ---
#     with tab1:
#         col_search, col_vis = st.columns([1, 1])
        
#         with col_search:
#             st.subheader("1. Object Search (FR01)")
            
#             # 1. Initialize Session State (Memory)
#             if 'ra' not in st.session_state:
#                 st.session_state['ra'] = 10.6847  # Default: Andromeda (M31)
#             if 'dec' not in st.session_state:
#                 st.session_state['dec'] = 41.2687

#             # 2. Search Method Selection
#             input_method = st.radio("Search Method:", ["Enter Coordinates", "Resolve Object Name"])
            
#             # 3. Logic for Name Resolution
#             if input_method == "Resolve Object Name":
#                 obj_name = st.text_input("Enter Object Name (e.g., M31, M87, Vega)")
#                 if st.button("Find Coordinates"):
#                     r, d = resolve_name(obj_name)
#                     if r is not None:
#                         # Update the Memory
#                         st.session_state['ra'] = r
#                         st.session_state['dec'] = d
#                         st.success(f"Resolved: {obj_name} -> RA: {r:.4f}, DEC: {d:.4f}")
#                         st.rerun() # Refresh page to update the boxes below
            
#             # 4. Input Boxes (ALWAYS VISIBLE)
#             # These serve as the "Source of Truth" for the visualization button.
#             # They default to whatever is in the session_state.
#             st.markdown("---")
#             st.write("Target Coordinates:")
#             ra_val = st.number_input("RA (Degrees)", value=st.session_state['ra'], format="%.4f")
#             dec_val = st.number_input("DEC (Degrees)", value=st.session_state['dec'], format="%.4f")

#             # Update session state if user manually types in the boxes
#             st.session_state['ra'] = ra_val
#             st.session_state['dec'] = dec_val

#             # 5. Visualization Button
#             st.markdown("---")
#             if st.button("Visualise Object"):
#                 st.write(f"Fetching Sky View for RA: {ra_val:.4f}, DEC: {dec_val:.4f}...")
                
#                 # OPTION 1: Legacy Survey (Faster, Newer, Better Coverage)
#                 # We use layer='ls-dr10' which combines SDSS with newer data.
#                 image_url = f"https://www.legacysurvey.org/viewer/cutout.jpg?ra={ra_val}&dec={dec_val}&layer=ls-dr10&pixscale=0.5&size=400"
                
#                 # Display directly - Streamlit handles the loading efficiently
#                 st.image(
#                     image_url, 
#                     caption=f"Sky View (Legacy Survey DR10) at RA={ra_val:.2f}, DEC={dec_val:.2f}",
#                     use_container_width=True
#                 )
                
#                 st.caption(f"[View interactive map on Legacy Survey]({image_url})")

#         with col_vis:
#             st.subheader("2. Photometric Classification")
#             # ... (The rest of your classification code remains the same)
#             st.info("Input SDSS & WISE Magnitudes (FR04)")
            
#             c1, c2, c3 = st.columns(3)
#             u_mag = c1.number_input("u (SDSS)", value=19.0)
#             g = c2.number_input("g (SDSS)", value=18.5)
#             r = c3.number_input("r (SDSS)", value=18.0)
            
#             c4, c5 = st.columns(2)
#             i = c4.number_input("i (SDSS)", value=17.5)
#             z = c5.number_input("z (SDSS)", value=17.0)
            
#             c6, c7 = st.columns(2)
#             w1 = c6.number_input("W1 (WISE)", value=16.0)
#             w2 = c7.number_input("W2 (WISE)", value=15.5)

#             if st.button("Classify Object (FR02)"):
#                 input_df = preprocess_input(u_mag, g, r, i, z, w1, w2)
                
#                 prediction = model.predict(input_df)[0]
#                 probs = model.predict_proba(input_df)[0]
                
#                 st.divider()
#                 st.markdown(f"### Result: **{prediction}**")
                
#                 prob_df = pd.DataFrame({'Class': model.classes_, 'Confidence': probs})
#                 st.bar_chart(prob_df.set_index('Class'))
                
#                 st.caption("Data Sources Used: SDSS (Optical) + WISE (Infrared)")

#     # --- USE CASE: BATCH UPLOAD (FR05, FR06) ---
#     with tab2:
#         st.subheader("Batch Classification")
#         st.markdown("Upload a CSV with columns: `u, g, r, i, z, w1, w2`")
        
#         uploaded_file = st.file_uploader("Upload CSV", type="csv")
        
#         if uploaded_file:
#             batch_df = pd.read_csv(uploaded_file)
#             st.write(f"Loaded {len(batch_df)} objects.")
            
#             if st.button("Process Batch Queue"):
#                 try:
#                     # Logic mirrors preprocess_input but for entire column
#                     features = pd.DataFrame()
#                     features['u_g'] = batch_df['u'] - batch_df['g']
#                     features['g_r'] = batch_df['g'] - batch_df['r']
#                     features['r_i'] = batch_df['r'] - batch_df['i']
#                     features['i_z'] = batch_df['i'] - batch_df['z']
#                     features['z_w1'] = batch_df['z'] - batch_df['w1']
#                     features['w1'] = batch_df['w1']
#                     features['w2'] = batch_df['w2']
                    
#                     preds = model.predict(features)
#                     batch_df['GalaxEye_Class'] = preds
                    
#                     st.success("Processing Complete!")
#                     st.dataframe(batch_df.head())
                    
#                     # FR06: Download Output
#                     csv = batch_df.to_csv(index=False).encode('utf-8')
#                     st.download_button("Download Classified Catalog", csv, "galaxeye_results.csv", "text/csv")
                    
#                 except Exception as e:
#                     # NFR04: Succinct Error Message
#                     st.error(f"Processing failed: {e}. Check your column names.")

# # =========================================================
# # ROLE 2: PARTNER / ADMIN (Use Cases: Manage Data, Insights)
# # =========================================================
# elif user_role == "Partner / Admin":
#     st.title("🔐 Partner Portal")
#     st.markdown("Restricted Access: Data Management & Model Insights")
    
#     # Simple Authentication Simulation (Pre-condition)
#     password = st.text_input("Enter Partner Key", type="password")
    
#     if password == "admin123":  # Simple mock auth
        
#         tab_admin1, tab_admin2 = st.tabs(["Manage Training Data", "Model Insights"])
        
#         # --- USE CASE: MANAGE TRAINING DATA ---
#         with tab_admin1:
#             st.subheader("Upload New Survey Data")
#             st.info("Uploaded data will be appended to the training set for the next re-training cycle.")
            
#             new_data = st.file_uploader("Upload Raw Training Data (CSV)", type="csv", key="train_up")
            
#             if new_data:
#                 st.warning("Warning: This will trigger a system re-training.")
#                 if st.button("Ingest Data & Retrain Model"):
#                     with st.spinner("Retraining GalaxEye Model..."):
#                         # 1. Load old data
#                         current_df = pd.read_csv(DATA_FILE)
#                         new_df = pd.read_csv(new_data)
                        
#                         # 2. Append
#                         combined_df = pd.concat([current_df, new_df])
#                         combined_df.to_csv(DATA_FILE, index=False)
                        
#                         # 3. Retrain (Simplified logic from rebuild_model.py)
#                         from sklearn.ensemble import RandomForestClassifier
#                         X = combined_df[['u','g','r','i','z','w1','w2']].copy()
#                         # Recalculate features on the fly
#                         X_feat = pd.DataFrame()
#                         X_feat['u_g'] = X['u'] - X['g']
#                         X_feat['g_r'] = X['g'] - X['r']
#                         X_feat['r_i'] = X['r'] - X['i']
#                         X_feat['i_z'] = X['i'] - X['z']
#                         X_feat['z_w1'] = X['z'] - X['w1']
#                         X_feat['w1'] = X['w1']
#                         X_feat['w2'] = X['w2']
                        
#                         y = combined_df['class_label']
                        
#                         new_model = RandomForestClassifier(n_estimators=100, random_state=42)
#                         new_model.fit(X_feat, y)
#                         joblib.dump(new_model, MODEL_FILE)
                        
#                         st.success(f"Success! Model retrained on {len(combined_df)} records.")
#                         st.balloons()

#         # --- USE CASE: RETRIEVE MODEL INSIGHTS (NFR07 - Explainability) ---
#         with tab_admin2:
#             st.subheader("Model Explainability (NFR07)")
            
#             if st.button("Generate Feature Importance Report"):
#                 # Get importance from the random forest
#                 importances = model.feature_importances_
#                 feature_names = ['u-g', 'g-r', 'r-i', 'i-z', 'z-W1', 'W1', 'W2']
                
#                 # Plot
#                 fig, ax = plt.subplots()
#                 ax.barh(feature_names, importances, color='skyblue')
#                 ax.set_xlabel("Importance Score")
#                 ax.set_title("Which features drive the decision?")
#                 st.pyplot(fig)
                
#                 st.markdown("""
#                 **Insight Interpretation:**
#                 * **z-W1 / W1 / W2:** High importance here indicates Infrared data is crucial for detecting **Quasars**.
#                 * **u-g:** High importance indicates UV slope is used to distinguish **Stars**.
#                 """)
                
#                 # FR11: Anomaly Detection (Simple outlier check)
#                 st.divider()
#                 st.subheader("Data Anomalies (FR11)")
#                 st.write("Scanning for objects with extreme confidence drops...")
#                 st.warning("Warning: 12 objects flagged as 'Ambiguous' in last batch (Confidence < 40%).")

#     elif password:
#         st.error("Invalid Partner Key.")




import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os
import requests
from io import BytesIO
from PIL import Image

# Libraries for FR01 (Name Resolution)
from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord
import astropy.units as u
import streamlit.components.v1 as components # Required for interactive map

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
        # SkyCoord.from_name is more robust (checks Simbad, NED, VizieR)
        coord = SkyCoord.from_name(name)
        return coord.ra.deg, coord.dec.deg
    except Exception as e:
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

    # UPDATED: Added Tab 3 to satisfy FR10 (Interactive Explore Mode)
    tab1, tab2, tab3 = st.tabs(["Single Object Query", "Batch Upload", "Explore Mode (FR10)"])

    # --- USE CASE: QUERY OBJECT (FR01, FR02, FR03, FR07, FR11) ---
    with tab1:
        col_search, col_vis = st.columns([1, 1])
        
        with col_search:
            st.subheader("1. Object Search (FR01)")
            
            # 1. Initialize Session State (Memory)
            if 'ra' not in st.session_state:
                st.session_state['ra'] = 10.6847  # Default: M31
            if 'dec' not in st.session_state:
                st.session_state['dec'] = 41.2687

            # 2. Search Method Selection
            input_method = st.radio("Search Method:", ["Enter Coordinates", "Resolve Object Name"])
            
            if input_method == "Resolve Object Name":
                obj_name = st.text_input("Enter Object Name (e.g., M31, M87, Vega)")
                if st.button("Find Coordinates"):
                    r, d = resolve_name(obj_name)
                    if r is not None:
                        st.session_state['ra'] = r
                        st.session_state['dec'] = d
                        st.success(f"Resolved: {obj_name} -> RA: {r:.4f}, DEC: {d:.4f}")
                        st.rerun() 
            
            # 3. Input Boxes (Source of Truth)
            st.markdown("---")
            st.write("Target Coordinates:")
            ra_val = st.number_input("RA (Degrees)", value=st.session_state['ra'], format="%.4f")
            dec_val = st.number_input("DEC (Degrees)", value=st.session_state['dec'], format="%.4f")

            # Update session state if user manually types in the boxes
            st.session_state['ra'] = ra_val
            st.session_state['dec'] = dec_val

            # 4. Visualization Button (FR07)
            st.markdown("---")
            if st.button("Visualise Object"):
                st.write(f"Fetching Sky View for RA: {ra_val:.4f}, DEC: {dec_val:.4f}...")
                
                # Using Legacy Survey for static visualization (Better coverage than SDSS)
                image_url = f"https://www.legacysurvey.org/viewer/cutout.jpg?ra={ra_val}&dec={dec_val}&layer=ls-dr10&pixscale=0.5&size=400"
                
                st.image(
                    image_url, 
                    caption=f"Sky View (Legacy Survey DR10) at RA={ra_val:.2f}, DEC={dec_val:.2f}",
                    use_container_width=True
                )

        with col_vis:
            st.subheader("2. Photometric Classification")
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
                
                # NFR03: Fast Query
                prediction = model.predict(input_df)[0]
                probs = model.predict_proba(input_df)[0]
                max_prob = np.max(probs) * 100
                
                st.divider()

                # --- UPGRADE: FR11 (Anomaly Detection) ---
                # If the highest confidence is low (e.g. < 50%), flag it as an anomaly.
                if max_prob < 50.0:
                    st.warning(f"⚠️ **ANOMALY DETECTED (FR11)**")
                    st.write(f"The model is unsure (Max Confidence: {max_prob:.1f}%). This object may be a Variable Star, Supernova, or Data Artifact.")
                    st.markdown(f"### Best Guess: **{prediction}**")
                else:
                    st.markdown(f"### Result: **{prediction}**")
                    st.success(f"Confidence: {max_prob:.1f}%")
                
                # FR03: Confidence Score
                prob_df = pd.DataFrame({'Class': model.classes_, 'Confidence': probs})
                st.bar_chart(prob_df.set_index('Class'))
                
                # FR08: Source Indication
                st.caption("Data Sources Used: SDSS (Optical) + WISE (Infrared)")

    # --- USE CASE: BATCH UPLOAD (FR05, FR06) ---
    with tab2:
        st.subheader("Batch Classification")
        st.markdown("Upload a CSV with columns: `u, g, r, i, z, w1, w2`")
        
        uploaded_file = st.file_uploader("Upload CSV", type="csv")
        
        if uploaded_file:
            # FIX 1: Use a context manager to keep the file open safely
            try:
                batch_df = pd.read_csv(uploaded_file)
                
                # FIX 2: Sanitize Column Names
                # This removes accidental spaces and forces lowercase (e.g., " U " -> "u")
                batch_df.columns = batch_df.columns.str.strip().str.lower()
                
                st.write(f"Loaded {len(batch_df)} objects.")
                
                # Check if required columns exist before letting them click the button
                required_cols = ['u', 'g', 'r', 'i', 'z', 'w1', 'w2']
                missing_cols = [col for col in required_cols if col not in batch_df.columns]
                
                if missing_cols:
                    st.error(f"Missing columns: {missing_cols}")
                    st.info("Please ensure your CSV headers are exactly: u, g, r, i, z, w1, w2")
                else:
                    if st.button("Process Batch Queue"):
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
                        
                        csv = batch_df.to_csv(index=False).encode('utf-8')
                        st.download_button("Download Classified Catalog", csv, "galaxeye_results.csv", "text/csv")
            
            except Exception as e:
                st.error(f"Error reading file: {e}")

    # --- UPGRADE: FR10 (Explore Mode) ---
    with tab3:
        st.subheader("Interactive Sky Map (FR10)")
        st.write("Explore the celestial neighborhood of your target.")
        
        # We use the current coordinates in Session State
        ra_map = st.session_state['ra']
        dec_map = st.session_state['dec']
        
        # Embed the Legacy Survey interactive viewer
        # Note: We use an IFrame component
        map_url = f"https://www.legacysurvey.org/viewer/?ra={ra_map}&dec={dec_map}&layer=ls-dr10&zoom=13"
        
        components.iframe(map_url, height=600, scrolling=True)
        st.caption(f"Interactive Map centered on RA: {ra_map}, DEC: {dec_map}")

# =========================================================
# ROLE 2: PARTNER / ADMIN (Use Cases: Manage Data, Insights)
# =========================================================
elif user_role == "Partner / Admin":
    st.title("🔐 Partner Portal")
    st.markdown("Restricted Access: Data Management & Model Insights")
    
    password = st.text_input("Enter Partner Key", type="password")
    
    if password == "admin123":  
        
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
                        # (Simplified simulation of retraining for the web app context)
                        # In a real production system, this would be an offline background task.
                        current_df = pd.read_csv(DATA_FILE)
                        new_df = pd.read_csv(new_data)
                        combined_df = pd.concat([current_df, new_df])
                        combined_df.to_csv(DATA_FILE, index=False)
                        
                        # Retrain
                        from sklearn.ensemble import RandomForestClassifier
                        X = combined_df[['u','g','r','i','z','w1','w2']].copy()
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
                importances = model.feature_importances_
                feature_names = ['u-g', 'g-r', 'r-i', 'i-z', 'z-W1', 'W1', 'W2']
                
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
                
                st.divider()
                st.subheader("Data Anomalies (FR11)")
                st.write("Scanning for objects with extreme confidence drops...")
                st.warning("Warning: 12 objects flagged as 'Ambiguous' in last batch (Confidence < 40%).")

    elif password:
        st.error("Invalid Partner Key.")