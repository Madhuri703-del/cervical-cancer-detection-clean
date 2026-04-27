# =========================================
# Cervical Cancer Detection using CNN
# Pap Smear Images - Cancer / Non-Cancer
# =========================================

# 1. IMPORT LIBRARIES
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np

# 2. DATA PREPROCESSING & AUGMENTATION
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

test_data = test_datagen.flow_from_directory(
    'dataset/test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# 3. BUILD CNN MODEL
model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# 4. COMPILE MODEL
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 5. TRAIN MODEL
history = model.fit(
    train_data,
    epochs=10,
    validation_data=test_data
)

# 6. EVALUATE MODEL
loss, accuracy = model.evaluate(test_data)
print("Test Accuracy:", accuracy)

# 7. PLOT ACCURACY GRAPH
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

# 8. PLOT LOSS GRAPH
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
# 9. PREDICT SINGLE IMAGE (FINAL & CORRECT)
from tensorflow.keras.preprocessing import image
import numpy as np

# CHANGE THIS PATH TO ANY IMAGE YOU WANT TO TEST
img_path = "dataset/test/normal/image3.jpg"
# img_path = "dataset/test/normal/normal1.jpg"

img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)

print("Prediction value:", prediction[0][0])

if prediction[0][0] > 0.5:
    print("Prediction: NORMAL (NO CERVICAL CANCER)")
else:
    print("Prediction: CERVICAL CANCER DETECTED")