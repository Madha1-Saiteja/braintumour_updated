"""
classifier.py
EfficientNet-B0 brain tumor classifier + Grad-CAM explainability.
Mirrors exactly the architecture used in classification.ipynb.
"""

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

CLASS_NAMES   = ['glioma', 'meningioma', 'notumor', 'pituitary']
NO_TUMOR_IDX  = 2
IMG_SIZE      = 224
LAST_CONV     = 'top_conv'


def build_classifier(weights_path: str, num_classes: int = 4, img_size: int = IMG_SIZE) -> Model:
    """
    Rebuild EfficientNet-B0 classifier exactly as in classification.ipynb
    and load saved weights.
    """
    base_model = EfficientNetB0(
        include_top=False,
        weights=None,
        input_shape=(img_size, img_size, 3),
        pooling=None
    )
    inputs  = layers.Input(shape=(img_size, img_size, 3), name='mri_input')
    x       = base_model(inputs, training=False)
    x       = layers.GlobalAveragePooling2D(name='gap')(x)
    x       = layers.BatchNormalization(name='bn_head')(x)
    x       = layers.Dense(256, activation='relu', name='dense_256')(x)
    x       = layers.Dropout(0.4, name='dropout')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)
    model   = Model(inputs, outputs, name='EfficientNetB0_BrainTumor')
    model.load_weights(weights_path)
    print(f"[Classifier] Weights loaded from: {weights_path}")
    return model


def preprocess_image(img_rgb: np.ndarray, img_size: int = IMG_SIZE) -> np.ndarray:
    """Resize + EfficientNet preprocess → (1, H, W, 3)."""
    img = cv2.resize(img_rgb, (img_size, img_size))
    img = preprocess_input(img.astype(np.float32))
    return np.expand_dims(img, axis=0)


def classify(model: Model, img_rgb: np.ndarray) -> dict:
    """
    Run classification on an RGB numpy image.
    Returns dict with class name, confidence, and all class probabilities.
    """
    img_tensor = preprocess_image(img_rgb)
    preds      = model.predict(img_tensor, verbose=0)[0]   # (4,)
    pred_idx   = int(np.argmax(preds))
    return {
        "class_name":    CLASS_NAMES[pred_idx],
        "class_idx":     pred_idx,
        "confidence":    float(preds[pred_idx]),
        "probabilities": {c: float(p) for c, p in zip(CLASS_NAMES, preds)},
        "has_tumor":     pred_idx != NO_TUMOR_IDX,
    }


# ── Grad-CAM ──────────────────────────────────────────────────────────────────

def _make_gradcam_heatmap(img_tensor: np.ndarray, model: Model,
                           last_conv_layer_name: str,
                           pred_index: int | None = None):
    """
    Compute Grad-CAM heatmap.
    Uses nested submodel approach to handle EfficientNet inside the wrapper model
    (same pattern as classification.ipynb Section 10).
    """
    # Find EfficientNet submodel
    base_model = None
    for layer in model.layers:
        if hasattr(layer, 'layers'):
            base_model = layer
            break
    if base_model is None:
        raise ValueError("EfficientNet submodel not found inside wrapper model.")

    grad_base = Model(
        inputs=base_model.input,
        outputs=[base_model.get_layer(last_conv_layer_name).output,
                 base_model.output]
    )

    gap         = model.get_layer('gap')
    bn_head     = model.get_layer('bn_head')
    dense_256   = model.get_layer('dense_256')
    dropout     = model.get_layer('dropout')
    predictions = model.get_layer('predictions')

    img_tf = tf.cast(img_tensor, tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs, base_features = grad_base(img_tf, training=False)
        x     = gap(base_features)
        x     = bn_head(x, training=False)
        x     = dense_256(x)
        x     = dropout(x, training=False)
        preds = predictions(x)

        if pred_index is None:
            pred_index = int(tf.argmax(preds[0]))
        class_channel = preds[:, pred_index]

    grads        = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap      = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap      = tf.squeeze(tf.maximum(heatmap, 0))
    heatmap      = (heatmap / (tf.reduce_max(heatmap) + 1e-8)).numpy()
    return heatmap


def get_gradcam_overlay(model: Model, img_rgb: np.ndarray,
                         pred_index: int | None = None,
                         alpha: float = 0.4) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Grad-CAM and return:
      - heatmap_rgb : (H, W, 3) colored heatmap as uint8
      - overlay     : (H, W, 3) heatmap blended onto original image as uint8
    """
    img_resized  = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    img_tensor   = preprocess_image(img_rgb)

    heatmap = _make_gradcam_heatmap(img_tensor, model, LAST_CONV, pred_index)

    # Resize heatmap to image size
    heatmap_resized = cv2.resize(heatmap, (img_resized.shape[1], img_resized.shape[0]))

    # Colorize heatmap (JET colormap)
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_rgb     = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Blend with original image
    img_uint8 = img_resized.astype(np.uint8)
    overlay   = cv2.addWeighted(img_uint8, 1 - alpha, heatmap_rgb, alpha, 0)

    return heatmap_rgb, overlay
