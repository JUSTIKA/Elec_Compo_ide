import os
import keras
import numpy as np
import pandas as pd
from glob import glob
import tensorflow as tf
import tensorflow.image as tfi

from tf_explain.core.grad_cam import GradCAM

# Data
from tensorflow.keras.utils import to_categorical
from PIL import Image

# Data Viz
import matplotlib.pyplot as plt

# Model
from keras.models import Model
from keras.layers import Layer
from keras.layers import Conv2D
from keras.layers import Dropout
from keras.layers import UpSampling2D
from keras.layers import concatenate
from keras.layers import Add
from keras.layers import Multiply
from keras.layers import Input
from keras.layers import MaxPool2D
from keras.layers import BatchNormalization

# Callbacks
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

# Metrics
from keras.metrics import MeanIoU


#Script found on https://www.kaggle.com/code/farahalarbeed/convert-binary-masks-to-yolo-format/notebook

def load_image(image, SIZE):
    image = Image.open(image)
    if image.mode == "L":
        image = image.convert("RGB")
    return np.round(tfi.resize(np.array((image)) / 255., (SIZE, SIZE)), 4)


def load_images(image_paths, SIZE, mask=False, trim=None):
    if trim is not None:
        image_paths = image_paths[:trim]

    if mask:
        images = np.zeros(shape=(len(image_paths), SIZE, SIZE, 1))
    else:
        images = np.zeros(shape=(len(image_paths), SIZE, SIZE, 3))

    for i, image_path in enumerate(image_paths):
        img = load_image(image_path, SIZE)
        if mask:
            if len(img.shape) == 3:  # Check if the image is RGB
                img = img.mean(axis=-1, keepdims=True)
            images[i, :, :, 0] = img[:, :, 0]
        else:
            images[i] = img

    return images


def show_image(image, title=None, cmap=None, alpha=1):
    plt.imshow(image, cmap=cmap, alpha=alpha)
    if title is not None:
        plt.title(title)
    plt.axis('off')


def show_mask(image, mask, cmap=None, alpha=0.4):
    plt.imshow(image)
    plt.imshow(tf.squeeze(mask), cmap=cmap, alpha=alpha)
    plt.axis('off')


SIZE = 256

main_directory = r'C:\Users\vaioj\OneDrive\Documents\GitHub\Elec_Compo_ide\test_segmentation_ds'

# List to store image and mask file paths
image_paths = []
mask_paths = []


# Function to traverse the directory structure
def extract_paths(root_dir):
    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        if os.path.isdir(folder_path):
            if folder == 'imgs':
                for filename in os.listdir(folder_path):
                    if filename.endswith('.jpg'):
                        image_paths.append(os.path.join(folder_path, filename))
            elif folder == 'masks':
                for filename in os.listdir(folder_path):
                    if filename.endswith('.jpg'):
                        mask_paths.append(os.path.join(folder_path, filename))
            elif folder == 'test':
                # If there's a 'test' subdirectory, recurse into it
                extract_paths(folder_path)


# Process the 'train' and 'test' directories
for dataset in ['train', 'test']:
    dataset_directory = os.path.join(main_directory, dataset)
    extract_paths(dataset_directory)


show_image(load_image(image_paths[0], SIZE))

show_mask(load_image(image_paths[0], SIZE), load_image(mask_paths[0], SIZE)[:, :, 0], alpha=0.6)

images = load_images(image_paths, SIZE)
masks = load_images(mask_paths, SIZE, mask=True)

# plt.figure(figsize=(13,8))
# for i in range(15):
#     plt.subplot(3,5,i+1)
#     id = np.random.randint(len(images))
#     show_mask(images[id], masks[id], cmap='jet')
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(13,8))
# for i in range(15):
#     plt.subplot(3,5,i+1)
#     id = np.random.randint(len(images))
#     show_mask(images[id], masks[id], cmap='binary')
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(13,8))
# for i in range(15):
#     plt.subplot(3,5,i+1)
#     id = np.random.randint(len(images))
#     show_mask(images[id], masks[id], cmap='afmhot')
# plt.tight_layout()
# plt.show()


class EncoderBlock(Layer):

    def __init__(self, filters, rate, pooling=True, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)

        self.filters = filters
        self.rate = rate
        self.pooling = pooling

        self.c1 = Conv2D(filters, kernel_size=3, strides=1, padding='same', activation='relu',
                           kernel_initializer='he_normal')
        self.drop = Dropout(rate)
        self.c2 = Conv2D(filters, kernel_size=3, strides=1, padding='same', activation='relu',
                           kernel_initializer='he_normal')
        self.pool = MaxPool2D()

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


class DecoderBlock(Layer):

    def __init__(self, filters, rate, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)

        self.filters = filters
        self.rate = rate

        self.up = UpSampling2D()
        self.net = EncoderBlock(filters, rate, pooling=False)

    def call(self, X):
        X, skip_X = X
        x = self.up(X)
        c_ = concatenate([x, skip_X])
        x = self.net(c_)
        return x

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "filters": self.filters,
            'rate': self.rate,
        }


class AttentionGate(Layer):

    def __init__(self, filters, bn, **kwargs):
        super(AttentionGate, self).__init__(**kwargs)

        self.filters = filters
        self.bn = bn

        self.normal = Conv2D(filters, kernel_size=3, padding='same', activation='relu',
                             kernel_initializer='he_normal')
        self.down = Conv2D(filters, kernel_size=3, strides=2, padding='same', activation='relu',
                           kernel_initializer='he_normal')
        self.learn = Conv2D(1, kernel_size=1, padding='same', activation='sigmoid')
        self.resample = UpSampling2D()
        self.BN = BatchNormalization()

    def call(self, X):
        X, skip_X = X

        x = self.normal(X)
        skip = self.down(skip_X)
        x = Add()([x, skip])
        x = self.learn(x)
        x = self.resample(x)
        f = Multiply()([x, skip_X])
        if self.bn:
            return self.BN(f)
        else:
            return f
        # return f

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "filters": self.filters,
            "bn": self.bn
        }


class ShowProgress(Callback):
    def on_epoch_end(self, epochs, logs=None):
        id = np.random.randint(len(images))
        exp = GradCAM()
        image = images[id]
        mask = masks[id]

        # Ensure image and mask shapes are correct
        print(f"Shape of image: {image.shape}")
        print(f"Shape of mask: {mask.shape}")

        # Predict mask
        pred_mask = self.model.predict(image[np.newaxis, ...])
        pred_mask = tf.convert_to_tensor(pred_mask)
        print(f"Shape of pred_mask: {pred_mask.shape}")

        # Ensure validation_data is compatible
        try:
            cam = exp.explain(
                validation_data=(tf.convert_to_tensor(image[np.newaxis, ...]), tf.convert_to_tensor(mask)),
                class_index=0,  # Assuming binary segmentation, class 0 or 1
                layer_name='Attention4',
                model=self.model
            )
        except Exception as e:
            print(f"GradCAM explain error: {e}")
            return  # Skip visualization if GradCAM fails

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 3, 1)
        plt.title("Original Mask")
        show_mask(image, mask, cmap='copper')

        plt.subplot(1, 3, 2)
        plt.title("Predicted Mask")
        show_mask(image, pred_mask, cmap='copper')

        plt.subplot(1, 3, 3)
        plt.title("GradCAM")
        show_image(cam, title="GradCAM")

        plt.tight_layout()
        plt.show()


input_layer = Input(shape=images.shape[-3:])

# Encoder
p1, c1 = EncoderBlock(32, 0.1, name="Encoder1")(input_layer)
p2, c2 = EncoderBlock(64, 0.1, name="Encoder2")(p1)
p3, c3 = EncoderBlock(128, 0.2, name="Encoder3")(p2)
p4, c4 = EncoderBlock(256, 0.2, name="Encoder4")(p3)

# Encoding
encoding = EncoderBlock(512, 0.3, pooling=False, name="Encoding")(p4)

# Attention + Decoder

a1 = AttentionGate(256, bn=True, name="Attention1")([encoding, c4])
d1 = DecoderBlock(256, 0.2, name="Decoder1")([encoding, a1])

a2 = AttentionGate(128, bn=True, name="Attention2")([d1, c3])
d2 = DecoderBlock(128, 0.2, name="Decoder2")([d1, a2])

a3 = AttentionGate(64, bn=True, name="Attention3")([d2, c2])
d3 = DecoderBlock(64, 0.1, name="Decoder3")([d2, a3])

a4 = AttentionGate(32, bn=True, name="Attention4")([d3, c1])
d4 = DecoderBlock(32, 0.1, name="Decoder4")([d3, a4])

# Output
output_layer = Conv2D(1, kernel_size=1, activation='sigmoid', padding='same')(d4)

# Model
model = Model(
    inputs=[input_layer],
    outputs=[output_layer]
)

# Compile
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy', MeanIoU(num_classes=2, name='IoU')]
)

# Callbacks
cb = [
    # EarlyStopping(patience=3, restore_best_weight=True), # With Segmentation I trust on eyes rather than on metrics
    ModelCheckpoint("AttentionCustomUNet.h5", save_best_only=True),
    ShowProgress()
]

BATCH_SIZE = 8
SPE = len(images) // BATCH_SIZE

# Training
results = model.fit(
    images, masks,
    validation_split=0.2,
    epochs=50,  # 15 will be enough for a good Model for better model go with 20+
    steps_per_epoch=SPE,
    batch_size=BATCH_SIZE,
    callbacks=cb
)

loss, accuracy, iou, val_loss, val_accuracy, val_iou = results.history.values()

plt.figure(figsize=(20, 8))

plt.subplot(1, 3, 1)
plt.title("Model Loss")
plt.plot(loss, label="Training")
plt.plot(val_loss, label="Validtion")
plt.legend()
plt.grid()

plt.subplot(1, 3, 2)
plt.title("Model Accuracy")
plt.plot(accuracy, label="Training")
plt.plot(val_accuracy, label="Validtion")
plt.legend()
plt.grid()

plt.subplot(1, 3, 3)
plt.title("Model IoU")
plt.plot(iou, label="Training")
plt.plot(val_iou, label="Validtion")
plt.legend()
plt.grid()

plt.show()

plt.figure(figsize=(20, 25))
n = 0
for i in range((5 * 3)):  # Adjusted range
    plt.subplot(5, 3, i + 1)
    if n == 0:
        id = np.random.randint(len(images))
        image = images[id]
        mask = masks[id]
        pred_mask = model.predict(image[np.newaxis, ...])

        plt.title("Original Mask")
        show_mask(image, mask)
        n += 1
    elif n == 1:
        plt.title("Predicted Mask")
        show_mask(image, pred_mask)
        n += 1
    elif n == 2:
        pred_mask = (pred_mask > 0.5).astype('float')
        plt.title("Processed Mask")
        show_mask(image, pred_mask)
        n = 0
plt.tight_layout()
plt.show()