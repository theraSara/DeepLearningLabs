import pathlib
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report

lung_aden = cv2.imread("./archive/Lung_and_Colon_Cancer/Lung_Adenocarcinoma/lungaca1.jpeg")
lung_squ = cv2.imread("./archive/Lung_and_Colon_Cancer/Lung_Squamous_Cell_Carcinoma/lungscc1.jpeg")
print(lung_aden.shape)
print(lung_squ.shape)

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
    strategy = tf.distribute.TPUStrategy(tpu)
except:
    strategy = tf.distribute.get_strategy()


print("Number of replicas: ", strategy.num_replicas_in_sync)
data_directory = pathlib.Path("./archive/Lung_and_Colon_Cancer/")
AUTOTUNE = tf.data.AUTOTUNE
batch_size = 25 * strategy.num_replicas_in_sync
class_names = ["Lung-Benign_Tissue", "Lung_Adenocarcinoma", "Lung_Squamous_Cell_Carcinoma"]


def preprocess_image(image, image_size):
    # Resize the image to the desired size
    image = tf.image.resize(image, (image_size, image_size))
    # Convert the pixel values to the range [0, 1]
    image = image / 255.0
    # Normalize the pixel values (subtract mean and divide by standard deviation)
    mean = tf.constant([0.485, 0.456, 0.406])
    std = tf.constant([0.229, 0.224, 0.225])
    image = (image - mean) / std
    return image

def test_image(model, image_path, class_names):
    img = image.load_img(image_path, target_size=(tile_size, tile_size))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255
    # Predict the class
    prediction = model.predict(x)
    max_index = np.argmax(prediction)
    print("This image is predicted to be: ", class_names[max_index])
    # Display the image
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.title(class_names[max_index])
    plt.axis('off')
    plt.show()

def tile_image(image, tile_size):
    # Input image shape is (H, W, 3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # Add batch and channel dimensions
    image = tf.expand_dims(image, axis=0)
    image = tf.expand_dims(image, axis=-1)

    # Calculate the number of patches in each dimension (H x W)
    num_patches_height = tf.shape(image)[1] // tile_size
    num_patches_width = tf.shape(image)[2] // tile_size

    # Generate the grid of coordinates for patches
    grid_x, grid_y = tf.meshgrid(
        tf.range(num_patches_width), tf.range(num_patches_height), indexing='ij'
    )

    # Calculate the patch coordinates in the original image (x, y)
    boxes = tf.stack([
        tf.cast(grid_y * tile_size, dtype=tf.float32),  # ymin
        tf.cast(grid_x * tile_size, dtype=tf.float32),  # xmin
        tf.cast((grid_y + 1) * tile_size, dtype=tf.float32),  # ymax
        tf.cast((grid_x + 1) * tile_size, dtype=tf.float32)  # xmax
    ], axis=-1)

    # Reshape boxes to (num_patches, 4)
    boxes = tf.reshape(boxes, [1, -1, 4])

    # Crop and resize patches
    patches = tf.image.crop_and_resize(
        image, boxes,
        box_indices=tf.zeros(tf.shape(boxes)[1], dtype=tf.int32),
        crop_size=[tile_size, tile_size]
    )

    return patches





def create_dataset(image_size, tile_size):
    def preprocess_and_tile(image, label):
        image = preprocess_image(image, image_size)
        image = tile_image(image, tile_size)
        return image, label
    return tf.keras.preprocessing.image_dataset_from_directory(
        data_directory,
        validation_split=0.2,
        subset="training",
        seed=123,
        label_mode='categorical',
        image_size=(image_size, image_size),
        batch_size=batch_size
    ).map(lambda x, y: preprocess_and_tile(x, y), num_parallel_calls=AUTOTUNE)


tile_sizes = [100, 200, 400]
for tile_size in tile_sizes:
    train_ds = create_dataset(tile_size, tile_size)
    validation_ds = create_dataset(tile_size, tile_size)
    test_ds = create_dataset(tile_size, tile_size)

    # ResNet-50 MODEL
    for batch_images, batch_labels in train_ds:
        try:
            # CNN: ResNet-50 model architecture
            resnetmodel = Sequential()
            pretrained_model_for_demo = tf.keras.applications.ResNet50(include_top=False,
                                                                       input_shape=(None, None, 3),
                                                                       pooling='avg', classes=3,
                                                                       weights='imagenet')
            for each_layer in pretrained_model_for_demo.layers:
                each_layer.trainable = False
            resnetmodel.add(pretrained_model_for_demo)
            num_batches = tf.data.experimental.cardinality(train_ds).numpy()
            print("Number of batches in the training dataset:", num_batches)

            # fully connected output layer
            resnetmodel.add(Flatten())
            resnetmodel.add(Dense(512, activation='relu'))
            resnetmodel.add(Dense(3, activation='softmax'))
            resnetmodel.summary()

            # call backs
            checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("xray_model.h5", save_best_only=True)
            early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

            initial_learning_rate = 0.015
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate,
                decay_steps=100000,
                decay_rate=0.96,
                staircase=True
            )

            # Training ResNet-50 Model:
            with strategy.scope():
                resnetmodel = resnetmodel
                METRICS = [
                    tf.keras.metrics.BinaryAccuracy(),
                    tf.keras.metrics.Precision(name="precision"),
                    tf.keras.metrics.Recall(name="recall")
                ]
                resnetmodel.compile(loss='binary_crossentropy', optimizer='adam', metrics=METRICS)

            epochs = 15
            history = resnetmodel.fit(train_ds, validation_data=validation_ds, epochs=epochs,
                                      callbacks=[checkpoint_cb, early_stopping_cb])


            test_results = resnetmodel.evaluate(test_ds, return_dict=True)

            test_image_path = "./archive/Lung_and_Colon_Cancer/Lung_Squamous_Cell_Carcinoma/lungscc1.jpeg"
            test_image(resnetmodel, test_image_path, class_names)

            plt.figure()
            for image, label in test_ds.take(1):
                plt.imshow(image[0] / 255)
                plt.title(class_names[label[0]])

            # Get the predicted labels for the test dataset
            predictions = resnetmodel.predict(test_ds)

            # Convert one-hot encoded labels to class indices
            y_true = np.argmax(np.concatenate([y for x, y in test_ds]), axis=-1)
            y_pred = np.argmax(predictions, axis=-1)

            # Calculate the confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            print("Confusion Matrix:")
            print(cm)
            # Print classification report
            print("Classification Report:")
            print(classification_report(y_true, y_pred, target_names=class_names))

            predict1 = resnetmodel.predict(test_ds.take(1))[0]
            scores = [1 - predict1, predict1]

            for score, name in zip(scores, class_names):
                print("This image is %.2f precent %s" % ((100 * score), name))

        except tf.errors.InvalidArgumentError as e:
            print("Error decoding image:", e)
            # Add additional information about the batch or image causing the error
            print("Batch shape:", batch_images.shape)
            print("Batch labels:", batch_labels)
