import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D

def get_last_conv_layer(model):
    """
    Finds and returns the last Conv2D layer object (handles Sequential & nested models).
    """
    for layer in reversed(model.layers):
        if isinstance(layer, Conv2D):
            return layer
        if hasattr(layer, "layers"):
            for sub_layer in reversed(layer.layers):
                if isinstance(sub_layer, Conv2D):
                    return sub_layer
    raise ValueError("No Conv2D layer found in the model.")

def make_gradcam_heatmap(img_array, model, last_conv_layer, pred_index=None):
   
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap + 1e-8)

    return heatmap.numpy()
