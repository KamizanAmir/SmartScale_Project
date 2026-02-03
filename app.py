import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps, ImageDraw, ImageFont
import numpy as np
import pandas as pd
import datetime
import os
import barcode
from barcode.writer import ImageWriter
from io import BytesIO

# --- 1. SETUP & CONFIGURATION ---
st.set_page_config(page_title="AI Smart Scale", layout="wide")

# --- 2. LOAD THE BRAIN ---
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="model/model_unquant.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    with open('model/labels.txt', 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    return interpreter, input_details, output_details, class_names

# --- 3. DATABASE FUNCTIONS ---
def get_plu_data():
    try:
        return pd.read_csv('plu_database.csv')
    except Exception:
        return pd.DataFrame()

def get_product_info(item_name, df):
    try:
        product = df[df['Item'] == item_name]
        return product.iloc[0] if not product.empty else None
    except Exception:
        return None

# --- 4. LOGGING FUNCTION ---
def log_transaction(item, plu, weight, price):
    log_file = 'transaction_log.csv'
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_data = pd.DataFrame([{
        "Timestamp": timestamp,
        "Item": item,
        "PLU": plu,
        "Weight_kg": weight,
        "Total_Price": price
    }])
    if not os.path.isfile(log_file):
        new_data.to_csv(log_file, index=False)
    else:
        new_data.to_csv(log_file, mode='a', header=False, index=False)

# --- 5. LABEL GENERATION FUNCTION (NEW) ---
def generate_label_image(item_name, plu, weight, price_per_kg, total_price):
    # 1. Create a blank white label (400x300 pixels)
    W, H = 400, 300
    label = Image.new('RGB', (W, H), 'white')
    draw = ImageDraw.Draw(label)
    
    # 2. Add Text Info
    # (Since we don't have custom fonts installed, we use the default PIL font)
    # To make it look "bold", we draw the text multiple times slightly offset
    
    # Store Header
    draw.text((120, 10), "FRESH MARKET", fill='black')
    draw.line((10, 30, 390, 30), fill='black', width=2)
    
    # Item Name (Big)
    draw.text((20, 50), f"ITEM: {item_name}", fill='black')
    
    # Details
    draw.text((20, 80), f"Weight: {weight:.3f} kg", fill='black')
    draw.text((20, 100), f"Price/kg: RM {price_per_kg:.2f}", fill='black')
    
    # Total Price (Big & Bold)
    draw.text((20, 130), f"TOTAL: RM {total_price:.2f}", fill='black')
    
    # 3. Generate Barcode
    # We use Code128 because it fits numbers and text comfortably
    code128 = barcode.get_barcode_class('code128')
    # Create barcode for the PLU (or transaction ID)
    my_code = code128(str(plu), writer=ImageWriter())
    
    # Save barcode to memory buffer
    buffer = BytesIO()
    my_code.write(buffer)
    buffer.seek(0)
    
    # Open barcode image and resize it
    barcode_img = Image.open(buffer)
    barcode_img = barcode_img.resize((300, 100))
    
    # Paste barcode onto the label at the bottom
    label.paste(barcode_img, (50, 180))
    
    return label

# --- 6. THE APP UI ---
st.title("🍎 AI Smart Scale PoC")
st.markdown("Place item on the scale to identify.")

col1, col2 = st.columns([1, 1])

# Initialize variables
detected_item = None
confidence_score = 0.0
plu_db = get_plu_data()

with col1:
    st.header("1. Camera Input")
    camera_image = st.camera_input("Take a picture of the produce")

with col2:
    st.header("2. Identification Results")

    if camera_image is not None:
        # AI PREDICTION
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
        
        if " " in raw_class_name:
            detected_item = raw_class_name.split(' ', 1)[1]
        else:
            detected_item = raw_class_name

        st.write(f"**AI Suggestion:** {detected_item} ({confidence_score*100:.1f}%)")

    # MANUAL OVERRIDE LOGIC
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
        if detected_item and confidence_score >= 0.75 and "Background" not in detected_item:
            final_item_name = detected_item
        elif detected_item and confidence_score < 0.75:
             st.error(f"❓ Low Confidence ({confidence_score*100:.1f}%). Please use Manual Override.")
        elif detected_item and "Background" in detected_item:
             st.info("Waiting for item...")

    # FINAL PROCESSING
    if final_item_name:
        item_data = get_product_info(final_item_name, plu_db)

        if item_data is not None:
            st.success(f"✅ Active Item: {item_data['Item']}")
            
            # Display Details
            c1, c2 = st.columns(2)
            c1.metric("PLU Code", item_data['PLU'])
            c2.metric("Price / Kg", f"RM {item_data['Price_Per_Kg']:.2f}")

            st.markdown("---")
            st.subheader("3. Weighing & Transaction")
            
            weight = st.number_input("Enter Weight (kg):", min_value=0.0, step=0.1)
            
            if weight > 0:
                total_price = weight * item_data['Price_Per_Kg']
                st.markdown(f"### 💰 Total: RM {total_price:.2f}")
                
                # --- BUTTONS ROW ---
                b1, b2 = st.columns(2)
                
                with b1:
                    if st.button("Confirm & Add to Cart"):
                        log_transaction(item_data['Item'], item_data['PLU'], weight, total_price)
                        st.toast(f"Saved: {item_data['Item']} - RM {total_price:.2f}")
                        st.balloons()
                
                with b2:
                    if st.button("🖨️ Print Label"):
                        # Generate the label image
                        label_img = generate_label_image(
                            item_data['Item'], 
                            item_data['PLU'], 
                            weight, 
                            item_data['Price_Per_Kg'], 
                            total_price
                        )
                        # Show it to the user
                        st.image(label_img, caption="Generated Label", width=300)
                        st.success("Label Sent to Printer...")
        else:
            st.error("Item found in AI/Selection but NOT in Database CSV.")

# --- ADMIN SECTION ---
st.markdown("---")
st.header("📊 Admin Dashboard")
with st.expander("📝 View Transaction History"):
    if os.path.exists('transaction_log.csv'):
        df_log = pd.read_csv('transaction_log.csv')
        df_log = df_log.sort_index(ascending=False)
        total_sales = df_log['Total_Price'].sum()
        total_items = len(df_log)
        
        m1, m2 = st.columns(2)
        m1.metric("Total Items Sold", total_items)
        m2.metric("Total Revenue", f"RM {total_sales:.2f}")
        
        st.dataframe(df_log, use_container_width=True)
        st.download_button(
            label="📥 Download Report (CSV)",
            data=df_log.to_csv(index=False).encode('utf-8'),
            file_name='daily_sales_report.csv',
            mime='text/csv',
        )
    else:
        st.info("No transactions recorded yet.")