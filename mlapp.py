import numpy as np
import cv2
from PIL import Image, ImageOps
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import pickle


model_path = "xgboost_digit_classifier.pkl"
with open(model_path, "rb") as file:
    model = pickle.load(file)

# Function to preprocess the image
def preprocess_image(image):
    # Convert to grayscale
    image = image.convert("L")
    
    # Invert colors if necessary
    image = ImageOps.invert(image)
    
    # Convert to NumPy array
    img_array = np.array(image)
    
    # Apply binary threshold
    _, img_array = cv2.threshold(img_array, 128, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours to detect digit area
    contours, _ = cv2.findContours(img_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get bounding box of the digit
        x, y, w, h = cv2.boundingRect(contours[0])
        
        # Crop the digit
        digit = img_array[y:y+h, x:x+w]
        
        # Resize while keeping aspect ratio
        aspect_ratio = w / h
        if aspect_ratio > 1:
            new_w = 20
            new_h = int(20 / aspect_ratio)
        else:
            new_h = 20
            new_w = int(20 * aspect_ratio)
        
        digit_resized = cv2.resize(digit, (new_w, new_h))
        
        # Create a blank 28x28 image
        img_padded = np.zeros((28, 28), dtype=np.uint8)
        
        # Compute the center position
        x_offset = (28 - new_w) // 2
        y_offset = (28 - new_h) // 2
        
        # Place the resized digit in the center
        img_padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = digit_resized
    else:
        # If no contour found, just resize normally
        img_padded = cv2.resize(img_array, (28, 28))
    
    # Normalize pixel values
    img_padded = img_padded / 255.0
    
    # Flatten the image for model input
    img_padded = img_padded.reshape(1, -1)
    
    return img_padded


st.set_page_config(page_title="Handwritten Digit Recognition", layout="wide")
st.title("Handwritten Digit Recognition ✍️")
st.markdown("""
    Welcome to the Handwritten Digit Recognition app!  
    This app uses a pre-trained model to predict handwritten digits.  
    You can upload an image, draw a digit, or use your camera to capture a digit.
""")
with st.sidebar:
    st.header("Options")
    option = st.radio("Choose input method:", ("Draw Digit","Upload Image", "Use Camera"))
    
    st.markdown("---")
    st.header("How to Use")
    st.markdown("""
        1. **Upload Image**: Upload a clear image of a handwritten digit.  
        2. **Draw Digit**: Draw a digit on the canvas.  
        3. **Use Camera**: Capture a digit using your camera.  
    """)
    
    st.markdown("---")
    if st.button("Reset App"):
        st.session_state.clear()
        st.experimental_rerun()

# Upload image
if option == "Upload Image":
    st.header("Upload an Image")
    uploaded_file = st.file_uploader("Upload a digit image", type=["png", "jpg", "jpeg"])
    
    if uploaded_file:
        st.success("File uploaded successfully!")
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        processed_img = preprocess_image(image)
        prediction = model.predict(processed_img)[0]
        st.success(f"### Predicted Digit: {prediction}")

# Drawable canvas
elif  option == "Draw Digit":
    st.header("Draw a Digit")
    st.write("Draw a digit below and see the prediction!")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=20,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas",
        )
    
    if canvas_result.image_data is not None:
        with col2:
            st.info("Processing your drawing...")
            canvas_image = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
            
            processed_img = preprocess_image(canvas_image)
            prediction = model.predict(processed_img)[0]
            st.success(f"### Predicted Digit: {prediction}")



# Camera input
elif option == "Use Camera":
    st.header("Use Camera")
    st.write("Show a handwritten digit to the camera and see the prediction!")
    
    # Use Streamlit's camera input
    picture = st.camera_input("Take a picture")
    
    if picture:
        st.success("Camera image captured!")
        image = Image.open(picture)
        st.image(image, caption="Captured Image", use_column_width=True)
        processed_img = preprocess_image(image)
        prediction = model.predict(processed_img)[0]
        st.success(f"### Predicted Digit: {prediction}")