import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import pandas as pd

# --- 1. SETUP & CONFIGURATION ---
st.set_page_config(page_title="AI Smart Scale", layout="wide")

# --- 2. LOAD THE BRAIN (TFLite Version) ---
@st.cache_resource
def load_model():
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path="model/model_unquant.tflite")
    interpreter.allocate_tensors()
    
    # Get details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Load labels
    with open('model/labels.txt', 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
        
    return interpreter, input_details, output_details, class_names

# --- 3. LOAD THE DATABASE (The CSV) ---
def get_product_info(predicted_label):
    try:
        df = pd.read_csv('plu_database.csv')
        # Clean label (e.g. "0 Red Apple" -> "Red Apple")
        if " " in predicted_label:
            clean_label = predicted_label.split(' ', 1)[1]
        else:
            clean_label = predicted_label
            
        product = df[df['Item'] == clean_label]
        
        if not product.empty:
            return product.iloc[0] 
        else:
            return None
    except Exception:
        return None

# --- 4. THE APP LAYOUT ---
st.title("🍎 AI Smart Scale PoC")
st.markdown("Place item on the scale to identify.")

col1, col2 = st.columns([1, 1])

with col1:
    st.header("1. Camera Input")
    camera_image = st.camera_input("Take a picture of the produce")

with col2:
    st.header("2. Identification Results")

    if camera_image is not None:
        # Load System
        interpreter, input_details, output_details, class_names = load_model()
        
        # --- A. PREPARE IMAGE ---
        image = Image.open(camera_image).convert("RGB")
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array

        # --- B. PREDICT ---
        interpreter.set_tensor(input_details[0]['index'], data)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        # --- C. DISPLAY RESULTS (STRICT MODE) ---
        
        # 1. Define strict threshold (75%)
        THRESHOLD = 0.75

        st.write(f"**Raw AI Guess:** {class_name} ({confidence_score*100:.1f}%)")

        if confidence_score < THRESHOLD:
            # STOP HERE if confidence is low
            st.error(f"❓ Low Confidence ({confidence_score*100:.1f}%). Please adjust lighting or move item closer.")
        
        else:
            # ONLY RUN THIS if confidence is HIGH
            item_data = get_product_info(class_name)

            if item_data is not None and item_data['Item'] != "Background":
                st.success(f"✅ Identified: {item_data['Item']}")
                st.metric(label="PLU Code", value=item_data['PLU'])
                price_per_kg = item_data['Price_Per_Kg']
                st.metric(label="Price / Kg", value=f"RM{price_per_kg:.2f}")

                st.markdown("---")
                st.subheader("3. Weighing")
                weight = st.number_input("Enter Weight (kg):", min_value=0.0, step=0.1)
                
                if weight > 0:
                    total_price = weight * price_per_kg
                    st.markdown(f"### 💰 Total Price: RM{total_price:.2f}")
                    if st.button("Add to Cart"):
                        st.toast(f"Added {item_data['Item']} to cart!")
            
            elif "Background" in class_name:
                st.info("Waiting for item...")
            
            else:
                st.error(f"Item '{class_name}' recognized, but not found in Database (CSV).")