import os
import keras
import numpy as np
import tensorflow as tf
import tensorflow.image as tfi
from PIL import Image
import matplotlib.pyplot as plt

#Script found on https://www.kaggle.com/code/farahalarbeed/convert-binary-masks-to-yolo-format/notebook

def load_and_preprocess_image(image_path, SIZE):
    """Loads, preprocesses, and resizes a single image."""
    image = Image.open(image_path)
    if image.mode == "L":
        image = image.convert("RGB")
    img_array = np.array(image) / 255.0
    img_resized = tfi.resize(img_array, (SIZE, SIZE))
    return img_resized.numpy()  # Convert TensorFlow tensor to NumPy


def visualize_segmentation(original_image_path, predicted_mask):
    """Visualizes the original image and the predicted mask."""
    original_image = Image.open(original_image_path)
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_image)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Predicted Mask")
    plt.imshow(tf.squeeze(predicted_mask), cmap='gray', alpha=0.5)  # Adjust cmap and alpha as needed
    plt.axis('off')

    plt.tight_layout()
    plt.show()


# *** Define your custom layers here ***
class EncoderBlock(keras.layers.Layer):  # Make sure to inherit from keras.layers.Layer
    def __init__(self, filters, rate, pooling=True, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.filters = filters
        self.rate = rate
        self.pooling = pooling
        self.c1 = keras.layers.Conv2D(filters, kernel_size=3, strides=1, padding='same', activation='relu',
                                       kernel_initializer='he_normal')
        self.drop = keras.layers.Dropout(rate)
        self.c2 = keras.layers.Conv2D(filters, kernel_size=3, strides=1, padding='same', activation='relu',
                                       kernel_initializer='he_normal')
        self.pool = keras.layers.MaxPool2D()

    def call(self, X):
        x = self.c1(X)
        x = self.drop(x)
        x = self.c2(x)
        if self.pooling:
            y = self.pool(x)
            return y, x
        else:
            return x

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "filters": self.filters,
            'rate': self.rate,
            'pooling': self.pooling
        }


class DecoderBlock(keras.layers.Layer):  # Make sure to inherit from keras.layers.Layer
    def __init__(self, filters, rate, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.filters = filters
        self.rate = rate
        self.up = keras.layers.UpSampling2D()
        self.net = EncoderBlock(filters, rate, pooling=False)

    def call(self, X):
        X, skip_X = X
        x = self.up(X)
        c_ = keras.layers.concatenate([x, skip_X])
        x = self.net(c_)
        return x

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "filters": self.filters,
            'rate': self.rate,
        }


class AttentionGate(keras.layers.Layer):  # Make sure to inherit from keras.layers.Layer
    def __init__(self, filters, bn, **kwargs):
        super(AttentionGate, self).__init__(**kwargs)
        self.filters = filters
        self.bn = bn
        self.normal = keras.layers.Conv2D(filters, kernel_size=3, padding='same', activation='relu',
                                            kernel_initializer='he_normal')
        self.down = keras.layers.Conv2D(filters, kernel_size=3, strides=2, padding='same', activation='relu',
                                          kernel_initializer='he_normal')
        self.learn = keras.layers.Conv2D(1, kernel_size=1, padding='same', activation='sigmoid')
        self.resample = keras.layers.UpSampling2D()
        self.BN = keras.layers.BatchNormalization()

    def call(self, X):
        X, skip_X = X
        x = self.normal(X)
        skip = self.down(skip_X)
        x = keras.layers.Add()([x, skip])
        x = self.learn(x)
        x = self.resample(x)
        f = keras.layers.Multiply()([x, skip_X])
        if self.bn:
            return self.BN(f)
        else:
            return f

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "filters": self.filters,
            "bn": self.bn
        }


if __name__ == '__main__':
    # Configuration
    SIZE = 256  # Must match the size used during training
    model_path = r"C:\Users\vaioj\OneDrive\Documents\GitHub\Elec_Compo_ide\AttentionCustomUNet.h5"  # Path to your trained model file
    image_path = r"C:\Users\vaioj\OneDrive\Documents\GitHub\Elec_Compo_ide\train_fine_tuningV6\train\images\20250407_111210_webp.rf.e8a550597e43a0ae5844610adad0bb6f.jpg"  # Replace with the path to your image


    # Load the trained model
    model = keras.models.load_model(
        model_path,
        custom_objects={
            'EncoderBlock': EncoderBlock,
            'DecoderBlock': DecoderBlock,
            'AttentionGate': AttentionGate
        }
    )

    # Load and preprocess the input image
    input_image = load_and_preprocess_image(image_path, SIZE)
    input_batch = np.expand_dims(input_image, axis=0)  # Add batch dimension

    # Make the prediction
    predicted_mask = model.predict(input_batch)

    # Visualize the results
    visualize_segmentation(image_path, predicted_mask[0])