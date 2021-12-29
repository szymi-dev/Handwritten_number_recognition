import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import cv2

# Uploading the data / pre-processing
mnist = tf.keras.datasets.mnist
(train_x, train_y), (test_x, test_y) = mnist.load_data()

# Normalizing the data
train_x = tf.keras.utils.normalize(train_x, axis=1)
test_x = tf.keras.utils.normalize(test_x, axis=1)

# Creating a model and adding layers

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_x, train_y, epochs=10)

# Evaluating the model
loss, accuracy = model.evaluate(test_x, test_y)
print(f"The loss of the model is {loss}")
print(f"The accuracy of the model is {accuracy}")

image_number = 1
while os.path.isfile(f"numbers/number{image_number}.png"):
    try:
        img = cv2.imread(f"numbers/number{image_number}.png")[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"This number is propably {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print("Error!")
    finally:
        image_number += 1

