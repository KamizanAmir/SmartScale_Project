import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import datetime
import os

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

# --- 3. DATABASE FUNCTIONS ---
def get_plu_data():
    """Loads the entire CSV database"""
    try:
        return pd.read_csv('plu_database.csv')
    except Exception:
        return pd.DataFrame() # Return empty if file missing

def get_product_info(item_name, df):
    """Finds a specific item in the loaded dataframe"""
    try:
        # Filter for the item
        product = df[df['Item'] == item_name]
        if not product.empty:
            return product.iloc[0] 
        else:
            return None
    except Exception:
        return None

# --- 4. LOGGING FUNCTION ---
def log_transaction(item, plu, weight, price):
    log_file = 'transaction_log.csv'
    
    # Create the data row
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_data = pd.DataFrame([{
        "Timestamp": timestamp,
        "Item": item,
        "PLU": plu,
        "Weight_kg": weight,
        "Total_Price": price
    }])

    # Append to file (or create if it doesn't exist)
    if not os.path.isfile(log_file):
        new_data.to_csv(log_file, index=False)
    else:
        new_data.to_csv(log_file, mode='a', header=False, index=False)

# --- 5. THE APP UI ---
st.title("🍎 AI Smart Scale PoC")
st.markdown("Place item on the scale to identify.")

col1, col2 = st.columns([1, 1])

# Initialize variables
detected_item = None
confidence_score = 0.0
plu_db = get_plu_data() # Load database once

with col1:
    st.header("1. Camera Input")
    camera_image = st.camera_input("Take a picture of the produce")

with col2:
    st.header("2. Identification Results")

    if camera_image is not None:
        # --- A. AI PREDICTION ---
        interpreter, input_details, output_details, class_names = load_model()
        
        image = Image.open(camera_image).convert("RGB")
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array

        interpreter.set_tensor(input_details[0]['index'], data)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        
        index = np.argmax(prediction)
        raw_class_name = class_names[index]
        confidence_score = prediction[0][index]

        # Clean class name (remove "0 ", "1 " numbers)
        if " " in raw_class_name:
            detected_item = raw_class_name.split(' ', 1)[1]
        else:
            detected_item = raw_class_name

        st.write(f"**AI Suggestion:** {detected_item} ({confidence_score*100:.1f}%)")

    # --- B. MANUAL OVERRIDE LOGIC ---
    final_item_name = None
    
    is_override = st.checkbox("⚠️ Manual Override (Wrong item / Low Confidence)")

    if is_override:
        if not plu_db.empty:
            item_list = plu_db[plu_db['Item'] != 'Background']['Item'].tolist()
            manual_selection = st.selectbox("Select Correct Item:", item_list)
            final_item_name = manual_selection
            st.info(f"Manual Mode: Selected {final_item_name}")
        else:
            st.error("Database empty.")
            
    else:
        # Use AI Result (Only if confidence is good)
        if detected_item and confidence_score >= 0.75 and "Background" not in detected_item:
            final_item_name = detected_item
        elif detected_item and confidence_score < 0.75:
             st.error(f"❓ Low Confidence ({confidence_score*100:.1f}%). Please use Manual Override.")
        elif detected_item and "Background" in detected_item:
             st.info("Waiting for item...")

    # --- C. FINAL PROCESSING ---
    if final_item_name:
        item_data = get_product_info(final_item_name, plu_db)

        if item_data is not None:
            st.success(f"✅ Active Item: {item_data['Item']}")
            
            # Display Details
            c1, c2 = st.columns(2)
            c1.metric("PLU Code", item_data['PLU'])
            c2.metric("Price / Kg", f"RM{item_data['Price_Per_Kg']:.2f}")

            st.markdown("---")
            st.subheader("3. Weighing & Transaction")
            
            # Input Weight
            weight = st.number_input("Enter Weight (kg):", min_value=0.0, step=0.1)
            
            if weight > 0:
                total_price = weight * item_data['Price_Per_Kg']
                st.markdown(f"### 💰 Total: RM{total_price:.2f}")
                
                # --- LOGGING BUTTON ---
                if st.button("Confirm & Add to Cart"):
                    log_transaction(item_data['Item'], item_data['PLU'], weight, total_price)
                    st.toast(f"Saved: {item_data['Item']} - RM{total_price:.2f}")
                    st.balloons()
        else:
            st.error("Item found in AI/Selection but NOT in Database CSV.")

# --- 6. ADMIN / HISTORY SECTION (NEW) ---
st.markdown("---")
st.header("📊 Admin Dashboard")

with st.expander("📝 View Transaction History"):
    if os.path.exists('transaction_log.csv'):
        # 1. Read the file
        df_log = pd.read_csv('transaction_log.csv')
        
        # 2. Sort by newest first
        df_log = df_log.sort_index(ascending=False)
        
        # 3. Show Summary Metrics
        total_sales = df_log['Total_Price'].sum()
        total_items = len(df_log)
        
        m1, m2 = st.columns(2)
        m1.metric("Total Items Sold", total_items)
        m2.metric("Total Revenue", f"RM{total_sales:.2f}")
        
        # 4. Show the Table
        st.dataframe(df_log, use_container_width=True)
        
        # 5. Download Button
        st.download_button(
            label="📥 Download Report (CSV)",
            data=df_log.to_csv(index=False).encode('utf-8'),
            file_name='daily_sales_report.csv',
            mime='text/csv',
        )
    else:
        st.info("No transactions recorded yet.")