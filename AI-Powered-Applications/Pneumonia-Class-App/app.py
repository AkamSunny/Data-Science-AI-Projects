%%writefile app.py
import warnings
import logging

# Suppress warnings
warnings.filterwarnings("ignore")
logging.getLogger("streamlit").setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import requests
import time

# Set page config
st.set_page_config(page_title="X-Ray Pneumonia Detector", layout="centered")

# HTML header
html_temp = """
<div style='background-color:red;padding:10px;border-radius:10px'>
<h2 style='color:white;text-align:center'>🫁 X-Ray Pneumonia Classifier</h2>
</div>
<br>
"""
st.markdown(html_temp, unsafe_allow_html=True)

img_size = 100
# CORRECTED GOOGLE DRIVE URL FORMAT
MODEL_URL = "https://drive.google.com/uc?export=download&id=1Ggbo174VhisymupMIsINWtZOfMtlLEci"

@st.cache_resource
def load_model_from_url():
    try:
        with st.spinner("📥 Downloading AI model from cloud..."):
            # Create session to handle Google Drive confirmation
            session = requests.Session()
            response = session.get(MODEL_URL, stream=True)
            response.raise_for_status()
            
            # Handle Google Drive's virus scan warning
            for key, value in response.cookies.items():
                if 'download_warning' in key:
                    confirm_url = f"https://drive.google.com/uc?export=download&id=1Ggbo174VhisymupMIsINWtZOfMtlLEci&confirm={value}"
                    response = session.get(confirm_url, stream=True)
                    break
            
            # Show download progress
            file_size = int(response.headers.get('content-length', 0))
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Download in chunks
            model_data = bytearray()
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    model_data.extend(chunk)
                    if file_size > 0:
                        progress = len(model_data) / file_size
                        progress_bar.progress(min(progress, 1.0))
                        status_text.text(f"Downloaded: {len(model_data)/1024/1024:.1f} MB / {file_size/1024/1024:.1f} MB")
            
            # Save to file
            with open("xray_model.keras", "wb") as f:
                f.write(model_data)
            
            progress_bar.empty()
            status_text.empty()
            
            # Load the model
            model = tf.keras.models.load_model("xray_model.keras")
            st.success("✅ AI Model loaded successfully!")
            return model
            
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.info("Please check your Google Drive file ID and sharing settings")
        return None

model = load_model_from_url()

def main():
    st.subheader("📤 Upload a Chest X-Ray Image for Analysis")
    st.write("This AI tool helps detect signs of pneumonia in chest X-ray images. ")
    st.write("   ⚠️ Make Sure you Upload an image associated with the used case ")
    
    uploaded_file = st.file_uploader(
        "Choose a JPG or PNG image of a chest X-ray", 
        type=['jpeg', 'jpg', 'png'],
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        try:
            # Process the image
            img = Image.open(uploaded_file).convert('L')
            img = img.resize((img_size, img_size))
            
            # Display the image (FIXED deprecated parameter)
            # Display the image centered with nice spacing
            st.write("")  # Add some top spacing

            col1, col2, col3 = st.columns([1, 3, 1])
            with col2:
                st.image(
                    img, 
                    caption="Uploaded X-ray Image", 
                    width=400,
                    output_format="auto"
                )

            st.write("")  # Add some bottom spacing
                        
            
            # Convert to array and preprocess
            img_array = np.array(img)
            img_array = img_array.reshape(1, img_size, img_size, 1)
            img_array = img_array / 255.0
            
            if st.button("🔍 ANALYZE IMAGE", type="primary", use_container_width=True):
                if model is not None:
                    with st.spinner("🤖 AI is analyzing the X-ray..."):
                        time.sleep(1)  # Simulate processing
                        
                        # Make prediction
                        prediction = model.predict(img_array, verbose=0)
                        confidence = prediction[0][0]
                        
                        # Determine result
                        if confidence > 0.5:
                            result = "PNEUMONIA DETECTED"
                            confidence_score = confidence
                            emoji = "⚠️"
                            color = "red"
                        else:
                            result = "NORMAL X-RAY"
                            confidence_score = 1 - confidence
                            emoji = "✅"
                            color = "green"
                        
                        # Display results
                        st.markdown("---")
                        st.markdown(f"### {emoji} ANALYSIS RESULTS")
                        
                        st.markdown(
                            f"""<div style='background-color:#f0f2f6;padding:20px;border-radius:10px;border-left:5px solid {color}'>
                            <h3 style='color:{color};'>{result}</h3>
                            <p><strong>Confidence Level:</strong> {confidence_score:.2%}</p>
                            </div>""", 
                            unsafe_allow_html=True
                        )
                        
                        # Medical context
                        if result == "PNEUMONIA DETECTED":
                            st.warning("""
                            **Clinical Note:** This result suggests radiographic findings consistent with pneumonia. 
                            Please consult with a healthcare professional for comprehensive evaluation.
                            """)
                        else:
                            st.success("""
                            **Clinical Note:** This result suggests no obvious radiographic evidence of pneumonia.
                            Routine clinical correlation is recommended.
                            """)
                        
                        # Disclaimer
                        st.info("""
                        **⚠️ Important Note:** It is required of the Medical Expert to take the Patient for Further Analysis/Screening
                        """)
                        
                else:
                    st.error("❌ Model not available. Please check the model connection.")
                    
        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")
            st.info("Please ensure you've uploaded a valid chest X-ray image.")

if __name__ == "__main__":
    main()
