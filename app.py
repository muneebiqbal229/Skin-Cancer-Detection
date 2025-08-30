import streamlit as st
st.set_page_config(page_title="Skin Cancer Prediction", layout="centered")

import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input

CLASS_INFO = {
    0: ("Actinic keratoses", "🔎 Suggest biopsy for potential precancerous lesion."),
    1: ("Basal cell carcinoma", "⚠️ Early-stage skin cancer. Dermatologist referral recommended."),
    2: ("Benign keratosis-like lesions", "✅ Non-cancerous. Routine observation advised."),
    3: ("Dermatofibroma", "✅ Benign skin nodule. Monitor unless changes appear."),
    4: ("Melanocytic nevi", "✅ Common mole. No concern unless asymmetric or evolving."),
    5: ("Melanoma", "🚨 High-risk cancer. Immediate specialist evaluation required."),
    6: ("Vascular lesions", "✅ Harmless blood vessel growth. No treatment typically needed.")
}

@st.cache_resource
def load_model_and_wrap():
    seq_model = load_model("../models/skin_cancer_cnn.h5")
    inputs = Input(shape=(100, 100, 3))
    outputs = seq_model(inputs)
    model = Model(inputs, outputs)
    return model

model = load_model_and_wrap()

def get_last_conv_layer_name(model):
    seq_model = model.get_layer("sequential")
    for layer in reversed(seq_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in the Sequential model.")

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    seq_model = model.get_layer("sequential")
    inputs = model.input
    x = inputs
    for layer in seq_model.layers:
        x = layer(x)
        if layer.name == last_conv_layer_name:
            conv_output = x
    predictions = x
    grad_model = tf.keras.Model(inputs=inputs, outputs=[conv_output, predictions])
    
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap)
    return heatmap.numpy(), int(pred_index), float(predictions[0][pred_index])

st.title("🧠 Skin Cancer Prediction with Grad-CAM")
st.markdown("Upload a skin lesion image")

uploaded_file = st.file_uploader("Drag and drop file here", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB").resize((100, 100))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    st.image(img, caption="Uploaded Image", width=300)

    prediction = model.predict(img_array)
    pred_index = np.argmax(prediction[0])
    confidence = prediction[0][pred_index] * 100
    label, advice = CLASS_INFO.get(pred_index, ("Unknown", "⚠️ No advice available."))

    st.success(f"🧬 **Prediction**: {label}  \n🎯 **Confidence**: {confidence:.2f}%")
    st.info(f"🩺 **Doctor's Advice:** {advice}")

    last_conv_layer_name = get_last_conv_layer_name(model)
    heatmap, _, _ = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index)

    img_cv = np.array(img)
    heatmap_resized = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_cv, 0.6, heatmap_colored, 0.4, 0)

    st.subheader("🔥 Grad-CAM Visualization")
    st.image(overlay, caption="Overlay Heatmap", width=400)