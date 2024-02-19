# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import os
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report, confusion_matrix, f1_score
from imblearn.over_sampling import SMOTE
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import tensorflow as tf

# Setting up constants and directories
directory ='./THE IQ-OTHNCCD lung cancer dataset/The IQ-OTHNCCD lung cancer dataset'
categories = ['Bengin cases', 'Malignant cases', 'Normal cases']
img_size = 256
data = []
augmentation = True

# Looping through categories to load and preprocess images
for i in categories:
    cnt, samples = 0,3
    fig, ax = plt.subplots(samples, 3, figsize = (15, 15))
    fig.suptitle(i)

    path = os.path.join(directory, i)
    class_num = categories.index(i)

    for file in os.listdir(path):
        filepath = os.path.join(path, file)
        img = cv2.imread(filepath, 0)  # 1 channel: gray-scale mode
        imgResized = cv2.resize(img, (img_size, img_size))
        # Applying preprocessing steps: histogram equalization
        imgEqualized = cv2.equalizeHist(imgResized)
        data.append([imgEqualized, class_num])

        if cnt < samples:
            ax[cnt, 0].imshow(img)
            ax[cnt, 1].imshow(imgResized)
            ax[cnt, 2].imshow(imgEqualized)
            cnt += 1
plt.show()

# Shuffling data to ensure randomness
random.shuffle(data)

# Extracting features and labels from the data
X, y = [], []
for feature, label in data:
    X.append(feature)
    y.append(label)
print('X: ', len(X))
print('y: ', len(y))

# Reshaping and normalizing the feature data
X = np.array(X).reshape(-1, img_size, img_size, 1)
X = X / 255.0
y = np.array(y)

# Splitting data into train and validation sets
Xtrain, XVld, ytrain, yVld = train_test_split(X, y, random_state=10) # we will be testing on the validation data

# Reshaping the training data for oversampling
Xtrain = Xtrain.reshape(Xtrain.shape[0], img_size*img_size)

# Applying SMOTE for oversampling the minority classes
smote = SMOTE()
Xtrain, ytrain = smote.fit_resample(Xtrain, ytrain)  # balances the dataset
Xtrain = Xtrain.reshape(Xtrain.shape[0], img_size, img_size, 1)

# Building the convolutional neural network model
model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=Xtrain.shape[1:], activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(16))
model.add(Dense(3, activation='softmax'))
model.summary()

# Compiling the model
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

# Calculating class weights to handle class imbalance
new_weights = {
    0: Xtrain.shape[0]/(3*Counter(ytrain)[0]),
    1: Xtrain.shape[0]/(3*Counter(ytrain)[1]),
    2: Xtrain.shape[0]/(3*Counter(ytrain)[2])
}

# Defining callbacks for training
checkpoint = tf.keras.callbacks.ModelCheckpoint("model.h5", save_best_only=True)
early_stopping = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

# Training the model
history = model.fit(Xtrain, ytrain, batch_size=8, epochs=50, validation_data=(XVld, yVld),
                    class_weight=new_weights, callbacks=[checkpoint, early_stopping])

# Saving the trained model
model.save('model.h5')

# Making predictions on the validation set and evaluating the model
y_pred = model.predict(XVld)
y_pred_label = np.argmax(y_pred, axis=1)
print(classification_report(yVld, y_pred_label))

# Plotting training history
plt.plot(history.history['loss'], label="Train")
plt.plot(history.history['val_loss'], label="Validation")
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label="Train")
plt.plot(history.history['val_accuracy'], label="Validation")
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend()
plt.show()
