# Importing necessary libraries
from keras.src.optimizers.schedules import learning_rate_schedule
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Setting up TPU if available
try:
  tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
  strategy = tf.distribute.TPUStrategy(tpu)
except:
  strategy = tf.distribute.get_strategy()

# Printing number of replicas for distributed training
print("Number of replicas: ", strategy.num_replicas_in_sync)

# Constants and data setup
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 25 * strategy.num_replicas_in_sync
IMAGE_SIZE = [180, 180]
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

# Loading training and testing data
train_images = tf.data.TFRecordDataset("gs://download.tensorflow.org/data/ChestXRay2017/train/images.tfrec")
train_paths = tf.data.TFRecordDataset("gs://download.tensorflow.org/data/ChestXRay2017/train/paths.tfrec")
ds = tf.data.Dataset.zip((train_images, train_paths))

test_images = tf.data.TFRecordDataset("gs://download.tensorflow.org/data/ChestXRay2017/test/images.tfrec")
test_paths = tf.data.TFRecordDataset("gs://download.tensorflow.org/data/ChestXRay2017/test/paths.tfrec")
test_ds = tf.data.Dataset.zip((test_images, test_paths))

# Counting normal and pneumonia cases
COUNT_NORMAL= len([filename for filename in train_paths if "NORMAL" in filename.numpy().decode("utf-8")])
COUNT_PNEUMONIA= len([filename for filename in train_paths if "PNEUMONIA" in filename.numpy().decode("utf-8")])
print("Normal: ", COUNT_NORMAL)
print("Pneumonia: ", COUNT_PNEUMONIA)

# Calculating initial bias and class weights
initial_bias = np.log([COUNT_PNEUMONIA / COUNT_NORMAL])
print("Initial bias: ", initial_bias)
TRAIN_IMG_COUNT = COUNT_PNEUMONIA + COUNT_NORMAL
weight_for_0 = (1 / COUNT_NORMAL) * TRAIN_IMG_COUNT / 2.0
weight_for_1 = (1 / COUNT_PNEUMONIA) * TRAIN_IMG_COUNT / 2.0
class_weight = {0: weight_for_0, 1: weight_for_1}
print("Weight for 0: ", weight_for_0)
print("Weight for 1: ", weight_for_1)

# Data processing functions
def process_path(image, path):
  label = get_label(path)
  img = decode_img(image)
  return img, label

def get_label(path):
  parts = tf.strings.split(path, "/")
  return parts[-2] == "PNEUMONIA"

def decode_img(image):
  img = tf.image.decode_jpeg(image, channels = 3)
  return tf.image.resize(img, IMAGE_SIZE)

# Preparing datasets
ds = ds.map(process_path, num_parallel_calls = AUTOTUNE)
test_ds = test_ds.map(process_path, num_parallel_calls = AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE)

ds = ds.shuffle(10000)
train_ds = ds.take(4200)
val_ds = ds.skip(4200)

# Exploring the datasets
for image, label in train_ds.take(1):
  print("Image shape: ", image.numpy().shape)
  print("Label: ", label.numpy())

# Function for preparing datasets for training
def prepare_for_training(ds):
  ds = ds.map(lambda image, label: (tf.cast(image, tf.float32), tf.cast(label, tf.int32)))
  ds = ds.batch(BATCH_SIZE)
  ds = ds.prefetch(buffer_size = AUTOTUNE)
  return ds

train_ds = prepare_for_training(train_ds)
val_ds = prepare_for_training(val_ds)

image_batch, label_batch = next(iter(train_ds))

# Function to display a batch of images
def show_batch(image_batch, label_batch):
  plt.figure(figsize = (10, 10))
  for n in range(25):
    ax = plt.subplot(5, 5, n + 1)
    plt.imshow(image_batch[n] / 255)
    if label_batch[n]:
      plt.title("Class 1")
    else:
      plt.title("Class 0")
    plt.axis("Off")

show_batch(image_batch, label_batch)

# Functions for building model components
def conv_block(filters, inputs):
  x = layers.SeparableConv2D(filters, 3, activation="relu", padding="same")(inputs)
  x = layers.SeparableConv2D(filters, 3, activation="relu", padding="same")(x)
  x = layers.BatchNormalization()(x)
  outputs = layers.MaxPool2D()(x)
  return outputs

def dense_block(units, dropout_rate, inputs):
  x = layers.Dense(units, activation="relu")(inputs)
  x = layers.BatchNormalization()(x)
  output = layers.Dropout(dropout_rate)(x)
  return output

# Function for building the model
def build_model():
  inputs = keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
  x = layers.Rescaling(1.0 / 255)(inputs)
  x = layers.Conv2D(16, 3, activation="relu", padding="same")(x)
  x = layers.Conv2D(16, 3, activation="relu", padding="same")(x)
  x = layers.MaxPool2D()(x)
  x = conv_block(32, x)
  x = conv_block(64, x)
  x = conv_block(128, x)
  x = layers.Dropout(0.2)(x)
  x = conv_block(256, x)
  x = layers.Dropout(0.2)(x)
  x = layers.Flatten()(x)
  x = dense_block(512, 0.7, x)
  x = dense_block(128, 0.5, x)
  x = dense_block(64, 0.2, x)
  outputs = layers.Dense(1, activation="sigmoid")(x)
  model = keras.Model(inputs=inputs, outputs=outputs)
  return model

# Setting up learning rate schedule
initial_learning_rate = 0.015
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, 
    decay_steps=100000, 
    decay_rate=0.96, 
    staircase=True
)

# Defining callbacks for training
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("xray_model.h5", save_best_only=True)
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

# Building and compiling the model
with strategy.scope():
  model = build_model()
  METRICS = [
      tf.keras.metrics.BinaryAccuracy(),
      tf.keras.metrics.Precision(name="precision"),
      tf.keras.metrics.Recall(name="recall")
  ]
  model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), 
      loss="binary_crossentropy", 
      metrics=METRICS
  )

# Training the model
history = model.fit(
    train_ds,
    epochs=100,
    validation_data=val_ds,
    class_weight=class_weight,
    callbacks=[checkpoint_cb, early_stopping_cb]
)

# Plotting training history
fig, ax = plt.subplots(1, 4, figsize=(20, 3))
ax = ax.ravel()
for i, met in enumerate(["precision", "recall", "binary_accuracy", "loss"]):
  ax[i].plot(history.history[met])
  ax[i].plot(history.history["val_" + met])
  ax[i].set_title("Model {}".format(met))
  ax[i].set_xlabel("epochs")
  ax[i].set_ylabel(met)
  ax[i].legend(["train", "val"])

# Evaluating the model on the test dataset
model.evaluate(test_ds, return_dict=True)

# Displaying a sample image and its prediction
plt.figure()
for image, label in test_ds.take(1):
  plt.imshow(image[0] / 255)
  plt.title(CLASS_NAMES[label[0]])
prediction = model.predict(test_ds.take(1))[0]
scores = [1-prediction, prediction]
for score, name in zip(scores, CLASS_NAMES):
  print("This image is %.2f percent %s" % ((100 * score), name))
