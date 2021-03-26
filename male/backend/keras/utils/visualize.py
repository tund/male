import cv2
import numpy as np
from packaging import version
import tensorflow as tf
import tensorflow.keras.backend as K


def make_gradcam_heatmap(model, imgs, proj_layer, target_ids=None):
    if isinstance(proj_layer, str):
        proj_layer = model.get_layer(proj_layer)

    if version.parse(tf.__version__) >= version.parse('2.0'):
        # Step 1: We create a model that maps the input image to the activations of the projection layer.
        inputs = tf.keras.Input(shape=(None, None, 3))
        x = tf.keras.layers.experimental.preprocessing.CenterCrop(224, 224)(inputs)
        x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
        outputs = model.layers[4](x, training=False)  # base model
        proj_layer = model.layers[4].layers[-1]
        proj_layer_model = tf.keras.Model(inputs=inputs, outputs=outputs)

        # Step 2: We create a model that maps the activations of the projection layer to the final class predictions.
        top_layers = [model.layers[5], model.layers[7]]
        inputs = tf.keras.Input(shape=proj_layer.output.shape[1:])
        # do the forward pass throught top layers
        x = inputs
        for i in top_layers:
            x = model.get_layer(i)(x) if isinstance(i, str) else i(x)
        top_layer_model = tf.keras.Model(inputs=inputs, outputs=x)

        # Step 3: We compute the gradient of the top predicted class for our input image
        # with respect to the activations of the projection layer.
        with tf.GradientTape() as tape:
            # Compute activations of the projection layer and make the tape watch it.
            proj_layer_output = proj_layer_model(imgs)
            tape.watch(proj_layer_output)
            # Predict
            preds = top_layer_model(proj_layer_output)
            if target_ids is None:
                target_ids = tf.argmax(preds, axis=1, output_type=tf.int32)
            target_nodes = tf.gather_nd(preds, tf.stack([tf.range(tf.size(target_ids)), target_ids], axis=1))

        # This is the gradient of the top predicted class with regard to
        # the output feature map of the projection layer.
        grads = tape.gradient(target_nodes, proj_layer_output)

        # This is a vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel.
        pooled_grads = tf.reduce_mean(grads, axis=(1, 2))

        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted class.
        proj_layer_output = proj_layer_output.numpy()
        pooled_grads = pooled_grads.numpy()
        proj_layer_output = proj_layer_output * pooled_grads[:, None, None, :]

        # The channel-wise mean of the resulting feature map is our heatmap of class activation.
        heatmaps = np.mean(proj_layer_output, axis=-1)

        # For visualization purpose, we will also normalize the heatmap between 0 & 1.
        heatmaps = np.maximum(heatmaps, 0) / np.max(heatmaps, axis=(1, 2))[:, None, None]

    else:
        output_nodes = model.output[:, target_ids]
        grads = K.gradients(output_nodes, proj_layer.output)[0]
        pooled_grads = K.mean(grads, axis=(0, 1, 2))
        iterate = K.function([model.input], [pooled_grads, proj_layer.output[0]])

        pooled_grads_value, proj_layer_output_value = iterate([imgs])
        for i in range(proj_layer_output_value.shape[2]):
            proj_layer_output_value[:, :, i] *= pooled_grads_value[i]

        heatmaps = np.mean(proj_layer_output_value, axis=-1)
        heatmaps = np.maximum(heatmaps, 0)
        heatmaps /= np.max(heatmaps)

    return heatmaps


def get_overlay(img, heatmap):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img
    superimposed_img /= np.max(superimposed_img)
    superimposed_img = np.uint8(255 * superimposed_img)
    return superimposed_img
