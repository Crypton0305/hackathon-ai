"""
explainability.py

Provides the Explainable AI parts of the app:

1. Grad-CAM for the CNN (image model) -> shows which part of the image
   the model focused on when deciding "damaged" vs "not damaged".

2. Feature importance for the ANN (tabular model) -> shows which input
   feature (claim amount, vehicle age, etc.) influenced the risk score
   the most. We use a simple permutation-based method: shuffle one
   feature at a time and see how much the prediction changes. This is
   easy to understand and does not need any extra heavy library.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def make_gradcam_heatmap(image_array, model, last_conv_layer_name="conv2"):
    """
    Creates a Grad-CAM heatmap for a single image.

    image_array: numpy array shaped (1, height, width, 3), values 0-1
    model: the trained CNN model
    last_conv_layer_name: name of the last convolution layer in the model
    """
    # Rebuild the model as a small functional graph so we can grab the output
    # of the last conv layer AND the final output at the same time.
    # (Sequential models loaded from disk don't reliably expose `.output`
    # in Keras 3, so we re-run the same layers through a fresh Input.)
    inputs = tf.keras.Input(shape=image_array.shape[1:])
    x = inputs
    conv_layer_output = None
    for layer in model.layers:
        x = layer(x)
        if layer.name == last_conv_layer_name:
            conv_layer_output = x

    grad_model = tf.keras.models.Model(inputs=inputs, outputs=[conv_layer_output, x])

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(image_array)
        # Since this is a single sigmoid output (damage probability),
        # we track that value directly.
        target_output = predictions[:, 0]

    # Gradients of the prediction with respect to the conv layer output
    grads = tape.gradient(target_output, conv_output)

    # Average the gradients over width and height -> importance per channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize heatmap between 0 and 1
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap_on_image(original_image, heatmap, alpha=0.4):
    """
    Resizes the heatmap to match the image size and overlays it in color
    on top of the original image, so the user can see "hot" damage areas.
    """
    heatmap_resized = tf.image.resize(
        heatmap[..., tf.newaxis], (original_image.shape[0], original_image.shape[1])
    ).numpy().squeeze()

    colormap = plt.get_cmap("jet")
    colored_heatmap = colormap(heatmap_resized)[:, :, :3]  # drop alpha channel

    overlaid_image = colored_heatmap * alpha + original_image * (1 - alpha)
    overlaid_image = np.clip(overlaid_image, 0, 1)
    return overlaid_image


def get_feature_importance(model, scaler, feature_columns, input_values, num_repeats=20):
    """
    Simple permutation-based feature importance for the ANN.

    For each feature:
    1. Take the scaled input row.
    2. Replace that one feature with random noise (several times).
    3. Measure how much the model's predicted risk class probability changes.
    4. Bigger change = more important feature.

    Returns a dictionary: {feature_name: importance_score}
    """
    base_input = scaler.transform([input_values])
    base_prediction = model.predict(base_input, verbose=0)[0]
    predicted_class = np.argmax(base_prediction)
    base_confidence = base_prediction[predicted_class]

    importance_scores = {}

    for feature_index, feature_name in enumerate(feature_columns):
        confidence_changes = []

        for _ in range(num_repeats):
            noisy_input = base_input.copy()
            # Replace this one feature with random noise from a normal distribution
            noisy_input[0, feature_index] = np.random.normal(0, 1)

            noisy_prediction = model.predict(noisy_input, verbose=0)[0]
            noisy_confidence = noisy_prediction[predicted_class]

            confidence_changes.append(abs(base_confidence - noisy_confidence))

        importance_scores[feature_name] = float(np.mean(confidence_changes))

    return importance_scores, predicted_class, float(base_confidence)
