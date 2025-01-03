import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import time

# Custom styling
def local_css():
    st.markdown(""" 
        <style>
        .main {
            padding: 2rem;
        }
        .stButton>button {
            width: 100%;
            background-color: #4CAF50;
            color: white;
            padding: 0.5rem;
            margin: 0.5rem 0;
        }
        .info-box {
            background-color: #f8f9fa;
            color: #2c3e50;
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
            border: 1px solid #dee2e6;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .info-box h4 {
            color: #1e3a8a;
            margin-bottom: 1rem;
            font-size: 1.2rem;
        }
        .info-box p, .info-box ul {
            color: #1f2937;
            font-size: 1rem;
            line-height: 1.6;
        }
        .info-box ul {
            margin-left: 1.5rem;
            margin-bottom: 1rem;
        }
        .info-box li {
            margin-bottom: 0.5rem;
        }
        .title {
            text-align: center;
            color: #2c3e50;
            margin-bottom: 2rem;
        }
        .prediction-box {
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)

def load_model():
    model = tf.keras.models.load_model('plant_disease_model.h5')
    return model

def preprocess_image(image):
    img = image.resize((224, 224))  # Resize image to the expected size
    img_array = tf.keras.preprocessing.image.img_to_array(img)  # Convert to array
    img_array = img_array / 255.0  # Normalize the image
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension
    return img_array

def main():
    local_css()
    
    st.markdown("<h1 class='title'>🌿 Plant Leaf Disease Detection</h1>", unsafe_allow_html=True)
    
    # Information about the application
    with st.expander("ℹ️ About This Application"):
        st.markdown("""
        <div class='info-box'>
        <h4>Why Plant Disease Detection?</h4>
        <p>Early detection of plant diseases is crucial for:</p>
        <ul>
            <li>Preventing crop losses and ensuring food security</li>
            <li>Reducing the use of chemical pesticides</li>
            <li>Optimizing agricultural resources</li>
            <li>Supporting sustainable farming practices</li>
        </ul>
        <h4>How It Works</h4>
        <p>This application uses a deep learning model trained on thousands of images of healthy and diseased plant leaves. 
        Simply upload a photo of a plant leaf, and our AI will analyze it for signs of disease.</p>
        <h4>Supported Plants</h4>
        <ul>
            <li>Apple</li>
            <li>Peach</li>
            <li>Potato</li>
            <li>Strawberry</li>
        </ul>
        <p>For each plant, we can detect whether the leaf is healthy or shows signs of disease.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # File uploader
    st.markdown("<h3>Upload a Leaf Image</h3>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Add a predict button
        if st.button('Analyze Image'):
            with st.spinner('Analyzing...'):
                # Simulate processing time with a progress bar
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                # Load model and make prediction
                model = load_model()
                processed_image = preprocess_image(image)
                prediction = model.predict(processed_image)
                
                # Debug: print raw prediction
                st.write("Raw Prediction Output:", prediction)
                
                # Class labels (update these based on your actual classes)
                # Class labels (Update to reflect your actual classes)
                classes = {
                    0: 'Apple - Healthy',
                    1: 'Peach - Healthy',
                    2: 'Potato - Healthy',
                    3: 'Strawberry - Healthy',
                    4: 'Apple - Diseased',
                    5: 'Peach - Diseased',
                    6: 'Potato - Diseased',
                    7: 'Strawberry - Diseased'
                }

                # Since it's a one-hot encoded prediction, we extract the index of the highest probability
                predicted_class = np.argmax(prediction[0])  # Index of the max value
                confidence = float(prediction[0][predicted_class])  # Confidence of the prediction
                
                # Display results with custom styling
                st.markdown(
                    f"""
                    <div class='prediction-box' style='background-color: {"#c8e6c9" if "Healthy" in classes[predicted_class] else "#ffcdd2"}; color: black;'>
                        <h2>Results</h2>
                        <h3>{classes[predicted_class]}</h3>
                        <p>Confidence: {confidence:.2%}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Additional information based on prediction
                if "Diseased" in classes[predicted_class]:
                    st.warning("⚠️ Disease detected! Please consult with an agricultural expert for treatment options.")
                else:
                    st.success("✅ Your plant appears to be healthy! Continue with regular care and monitoring.")

if __name__ == "__main__":
    main()
