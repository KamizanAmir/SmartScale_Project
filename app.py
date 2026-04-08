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
import base64
import streamlit.components.v1 as components
import time

# --- 1. SETUP & CONFIGURATION ---
st.set_page_config(page_title="AI Smart Scale", layout="wide")

# Initialize Session State for Weight Reset
if 'weight_input' not in st.session_state:
    st.session_state['weight_input'] = 0.0

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

# --- 4. TRANSACTION FUNCTIONS ---
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

def delete_transaction(index_to_delete):
    """Deletes a specific row from the CSV"""
    log_file = 'transaction_log.csv'
    if os.path.exists(log_file):
        df = pd.read_csv(log_file)
        # Sort descending to match the UI view (newest first)
        df_sorted = df.sort_index(ascending=False)
        
        # Get the actual index in the original dataframe
        actual_index = df_sorted.index[index_to_delete]
        
        # Drop it
        df = df.drop(actual_index)
        df.to_csv(log_file, index=False)
        return True
    return False

# --- 5. IMAGE & PRINT FUNCTIONS ---
def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def generate_label_image(item_name, plu, weight, price_per_kg, total_price):
    # Canvas matching 40mm x 30mm aspect ratio (4:3) for crisp printing
    W, H = 600, 450
    label = Image.new('RGB', (W, H), 'white')
    draw = ImageDraw.Draw(label)
    
    # --- Robust Font Loading ---
    # Expanded to catch more Linux/Mac/Windows environments
    font_path = None
    system_fonts = [
        "ARIAL.TTF",                                                    # Local folder (Recommended)
        "C:\\Windows\\Fonts\\ARIAL.TTF",                                # Windows
        "C:\\Windows\\Fonts\\ARIAL.TTF",                                # Windows alternate
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",         # Linux common
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", # Linux alternate
        "/System/Library/Fonts/Supplemental/ARIAL.TTF",                 # Mac
        "/Library/Fonts/ARIAL.TTF",                                     # Mac alternate
        "/System/Library/Fonts/Helvetica.ttc"                           # Mac fallback
    ]
    
    for path in system_fonts:
        if os.path.exists(path):
            font_path = path
            break
            
    try:
        if font_path:
            # Greatly increased sizes for high-res canvas
            font_title = ImageFont.truetype(font_path, 46)
            font_item  = ImageFont.truetype(font_path, 54)
            font_label = ImageFont.truetype(font_path, 28)
            font_value = ImageFont.truetype(font_path, 36)
            font_total_label = ImageFont.truetype(font_path, 36)
            font_total = ImageFont.truetype(font_path, 75)
        else:
            # If no font is found, PIL uses a tiny default font
            font_title = font_item = font_label = font_value = font_total_label = font_total = ImageFont.load_default()
    except Exception:
        font_title = font_item = font_label = font_value = font_total_label = font_total = ImageFont.load_default()

    # Helper function to perfectly center text dynamically
    def get_text_x(text, font):
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            return (W - text_width) // 2
        except Exception:
            return (W - (len(text) * 15)) // 2 

    # --- TOP SECTION (Centered) ---
    store_text = "FRESH MARKET"
    draw.text((get_text_x(store_text, font_title), 5), store_text, fill='black', font=font_title)
    
    item_text = f"{item_name.upper()}"
    draw.text((get_text_x(item_text, font_item), 55), item_text, fill='black', font=font_item)
    
    # --- DIVIDER ---
    draw.line((20, 115, 580, 115), fill='black', width=3)
    
    # --- MIDDLE SECTION (Date, Price/KG, Weight) ---
    current_date = datetime.datetime.now().strftime("%d/%m/%y")
    
    # Date (Left)
    draw.text((30, 125), "DATE", fill='black', font=font_label)
    draw.text((30, 160), current_date, fill='black', font=font_value)
    
    # Price / KG (Center)
    price_text = f"RM {price_per_kg:.2f}"
    draw.text((250, 125), "PRICE/KG", fill='black', font=font_label)
    draw.text((250, 160), price_text, fill='black', font=font_value)
    
    # Weight (Right)
    weight_text = f"{weight:.3f} kg"
    draw.text((450, 125), "WEIGHT", fill='black', font=font_label)
    draw.text((450, 160), weight_text, fill='black', font=font_value)

    # --- TOTAL PRICE (Highlighted and Centered) ---
    draw.line((20, 205, 580, 205), fill='black', width=3)
    total_lbl = "TOTAL:"
    total_val = f"RM {total_price:.2f}"
    draw.text((80, 235), total_lbl, fill='black', font=font_total_label)
    draw.text((200, 210), total_val, fill='black', font=font_total)
    
    # --- BOTTOM SECTION (Barcode mathematically centered) ---
    code128 = barcode.get_barcode_class('code128')
    try:
        my_code = code128(str(plu), writer=ImageWriter())
        buffer = BytesIO()
        my_code.write(buffer, options={"write_text": True, "text_distance": 4, "module_height": 8.0})
        buffer.seek(0)
        
        # Stretch barcode to be clearly scannable
        barcode_width = 440
        barcode_height = 140
        barcode_img = Image.open(buffer).resize((barcode_width, barcode_height))
        
        # Calculate precise center X for the barcode
        barcode_x = (W - barcode_width) // 2
        label.paste(barcode_img, (barcode_x, 300))
    except Exception:
        err_msg = f"[BARCODE ERROR: {plu}]"
        draw.text((get_text_x(err_msg, font_value), 320), err_msg, fill='red', font=font_value)

    return label
def trigger_print_dialog(label_img):
    b64_img = image_to_base64(label_img)
    print_html = f"""
    <html>
    <head>
    <style>
        @page {{
            margin: 0;
            /* Tells the browser to format the paper sideways */
            size: landscape; 
        }}
        body {{
            margin: 0;
            padding: 0;
            background: white;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            width: 100vw;
            overflow: hidden;
        }}
        img {{
            /* Fills the label space securely without cutting off edges */
            width: 100%; 
            height: 100%; 
            object-fit: contain;
            display: block;
        }}
    </style>
    </head>
    <body>
        <img src="data:image/png;base64,{b64_img}">
        <script>
            setTimeout(function() {{ 
                window.focus(); 
                window.print(); 
            }}, 500);
        </script>
    </body>
    </html>
    """
    components.html(print_html, height=150, scrolling=False)
# --- 6. CALLBACK TO CLEAR WEIGHT ---
def clear_weight():
    st.session_state['weight_input'] = 0.0

# --- 7. THE APP UI ---
st.title("🍎 Ai Vision Scale")
st.markdown("Place item on the scale to identify.")

col1, col2 = st.columns([1, 1])

detected_item = None
confidence_score = 0.0
plu_db = get_plu_data()

with col1:
    st.header("1. Camera Input")
    camera_image = st.camera_input("Take a picture of the produce")

with col2:
    st.header("2. Identification Results")

    if camera_image is not None:
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

    # Override Logic
    final_item_name = None
    is_override = st.checkbox("⚠️ Manual Override (Wrong item / Low Confidence)")

    if is_override:
        if not plu_db.empty:
            item_list = plu_db[plu_db['Item'] != 'Background']['Item'].tolist()
            final_item_name = st.selectbox("Select Correct Item:", item_list)
        else:
            st.error("Database empty.")
    else:
        if detected_item and confidence_score >= 0.75 and "Background" not in detected_item:
            final_item_name = detected_item
        elif detected_item and confidence_score < 0.75:
             st.error(f"❓ Low Confidence ({confidence_score*100:.1f}%). Please use Manual Override.")
        elif detected_item and "Background" in detected_item:
             st.info("Waiting for item...")

    # Transaction Logic
    if final_item_name:
        item_data = get_product_info(final_item_name, plu_db)

        if item_data is not None:
            st.success(f"✅ Active Item: {item_data['Item']}")
            c1, c2 = st.columns(2)
            c1.metric("PLU Code", item_data['PLU'])
            c2.metric("Price / Kg", f"RM {item_data['Price_Per_Kg']:.2f}")

            st.markdown("---")
            st.subheader("3. Weighing & Transaction")
            
            # --- WEIGHT INPUT WITH SESSION STATE ---
            # This is the key fix: We bind the value to 'weight_input' key
            weight = st.number_input(
                "Enter Weight (kg):", 
                min_value=0.0, 
                step=0.1, 
                key='weight_input'
            )
            
            if weight > 0:
                total_price = weight * item_data['Price_Per_Kg']
                st.markdown(f"### 💰 Total: RM {total_price:.2f}")
                
                b1, b2 = st.columns(2)
                with b1:
                    # BUTTON FIX: Using on_click callback
                    def on_add_click():
                        log_transaction(item_data['Item'], item_data['PLU'], weight, total_price)
                        clear_weight() # Resets weight to 0 immediately
                    
                    if st.button("Confirm & Add to Cart", on_click=on_add_click):
                        st.toast(f"Saved: {item_data['Item']} - RM {total_price:.2f}")
                        st.balloons()
                
                with b2:
                    if st.button("🖨️ Print Label"):
                        label_img = generate_label_image(
                            item_data['Item'], 
                            item_data['PLU'], 
                            weight, 
                            item_data['Price_Per_Kg'], 
                            total_price
                        )
                        trigger_print_dialog(label_img)
                        st.success("Print Dialog Opened!")
        else:
            st.error("Item found but missing in CSV.")

# --- ADMIN SECTION ---
st.markdown("---")
st.header("📊 Admin Dashboard")

with st.expander("📝 View & Manage Transactions", expanded=True):
    if os.path.exists('transaction_log.csv'):
        df_log = pd.read_csv('transaction_log.csv')
        df_log = df_log.sort_index(ascending=False) # Newest top
        
        # Metrics
        total_sales = df_log['Total_Price'].sum()
        total_items = len(df_log)
        m1, m2 = st.columns(2)
        m1.metric("Total Items Sold", total_items)
        m2.metric("Total Revenue", f"RM {total_sales:.2f}")
        
        st.dataframe(df_log, use_container_width=True)
        
        st.markdown("---")
        
        # --- REPRINT & DELETE CONTROLS ---
        a1, a2 = st.columns(2)
        
        if not df_log.empty:
            # Create list of options
            options_labels = [
                f"{i}: {row.Timestamp} - {row.Item} (RM {row.Total_Price:.2f})"
                for i, row in enumerate(df_log.itertuples(index=False))
            ]
            
            # Use index to track selection
            selected_option_str = st.selectbox("Select Transaction to Manage:", options_labels)
            selected_index = options_labels.index(selected_option_str)
            
            with a1:
                # REPRINT
                if st.button("🖨️ Reprint Selected"):
                    # Get row from sorted dataframe
                    selected_row = df_log.iloc[selected_index]
                    derived_price = selected_row['Total_Price'] / selected_row['Weight_kg']
                    img = generate_label_image(selected_row['Item'], selected_row['PLU'], selected_row['Weight_kg'], derived_price, selected_row['Total_Price'])
                    trigger_print_dialog(img)
            
            with a2:
                # DELETE
                if st.button("🗑️ Delete Transaction", type="primary"):
                    success = delete_transaction(selected_index)
                    if success:
                        st.success("Deleted successfully! Refreshing...")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Could not delete.")
        
    else:
        st.info("No transactions recorded yet.")