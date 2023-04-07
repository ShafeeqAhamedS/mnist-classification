# Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset
Digit classification and to verify the response for scanned handwritten images.

The MNIST dataset is a collection of handwritten digits. The task is to classify a given image of a handwritten digit into one of 10 classes representing integer values from 0 to 9, inclusively. The dataset has a collection of 60,000 handwrittend digits of size 28 X 28. Here we build a convolutional neural network model that is able to classify to it's appropriate numerical value.

![image](https://user-images.githubusercontent.com/75235293/190975763-7d3b7c0f-9458-41e9-a35c-aa063c4977da.png)
## Neural Network Model
![image](https://user-images.githubusercontent.com/93427237/230611027-babab700-8362-40f5-99da-ed0182769ae0.png)


## DESIGN STEPS

### STEP 1:
Import tensorflow and preprocessing libraries

### STEP 2:
Build a CNN model

### STEP 3:
Compile and fit the model and then predict

## PROGRAM
### Libraries
```py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report,confusion_matrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras import utils
from tensorflow.keras.preprocessing import image
```
### One Hot Encoding Outputs
```py
y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)
```
### Reshape Inputs
```py
X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)
```
### Build CNN Model
```py
model = keras.Sequential()
input = keras.Input(shape=(28,28,1))
model.add(input)

model.add(layers.Conv2D(filters=32,kernel_size=(5,5),strides=(1,1),padding='valid',activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Conv2D(filters=64,kernel_size=(5,5),strides=(1,1),padding='same',activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(20,activation='relu'))
model.add(layers.Dense(15,activation='relu'))
model.add(layers.Dense(5,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))

model.compile(optimizer='adam',
           loss='categorical_crossentropy',
           metrics=['accuracy'])

model.fit(X_train_scaled ,y_train_onehot, epochs=5,
          batch_size=64, 
          validation_data=(X_test_scaled,y_test_onehot))

```
### Metrics
```py
metrics = pd.DataFrame(model.history.history)
metrics.head()
metrics[['loss','val_loss']].plot()
metrics[['accuracy','val_accuracy']].plot()

x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)
print(confusion_matrix(y_test,x_test_predictions))
print(classification_report(y_test,x_test_predictions))
```
### Predict for own handwriting
```py
img = image.load_img('/content/drive/MyDrive/Colab Notebooks/Deep Learning/Lab/Exp 3/eight.png')
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)

img_28_gray_inverted = 255.0-img_28_gray
img_28_gray_inverted_scaled = img_28_gray_inverted.numpy()/255.0


x_single_prediction = np.argmax(
    model.predict(img_28_gray_inverted_scaled.reshape(1,28,28,1)),
     axis=1)

plt.imshow(img_28_gray_inverted_scaled.reshape(28,28),cmap='gray')

print(x_single_prediction)
```
## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot
![image](https://user-images.githubusercontent.com/93427237/230611064-952bf45f-b5bb-4521-a82c-df4675852daa.png)
![image](https://user-images.githubusercontent.com/93427237/230611103-b8d67fc0-1de5-495d-b449-ba77e77c202a.png)

### Classification Report
![image](https://user-images.githubusercontent.com/93427237/230611141-2cff4d64-a4d9-4a01-b9d5-a95dab501adb.png)

### Confusion Matrix
![image](https://user-images.githubusercontent.com/93427237/230611171-c3e67fbc-47e3-4cde-ac2c-14304599d54d.png)

### New Sample Data Prediction
![image](https://user-images.githubusercontent.com/93427237/230611209-d907cfb8-5fac-4111-9cfb-caaec5b2c468.png)
![image](https://user-images.githubusercontent.com/93427237/230611219-0032795e-da0b-45dc-9c1e-aad75476c326.png)

## RESULT
A convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is developed sucessfully.
