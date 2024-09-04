import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load the Keras model
model_path = 'my_model.keras'

@st.cache_resource  # Cache the model to avoid reloading it on each run
def load_cifar_model(model_path):
    return load_model(model_path)

model = load_cifar_model(model_path)

st.title("CIFAR Model Viewer")

# Display the model summary
st.subheader("Model Summary")
model.summary(print_fn=lambda x: st.text(x))

# Allow the user to upload an image
uploaded_file = st.file_uploader("Upload an image to classify", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load the image and preprocess it
    image = load_img(uploaded_file, target_size=(32, 32))  # Adjust target size based on your model's input size
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize the image if needed
    
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Make a prediction
    prediction = model.predict(img_array)
    st.subheader("Prediction")
    
    # Assuming your model is trained on CIFAR-10, here are the class names
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    predicted_class = class_names[np.argmax(prediction)]
    st.write(f"Predicted Class: {predicted_class}")
    
    # Display prediction probabilities
    st.subheader("Prediction Probabilities")
    for i, class_name in enumerate(class_names):
        st.write(f"{class_name}: {prediction[0][i]:.4f}")

