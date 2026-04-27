import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

IMG_SIZE = 224
BATCH_SIZE = 16   # 🔥 slightly increased for better gradient stability

# 🔥 Better Data Augmentation (balanced)
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=25,   # 🔥 reduced (too high = distortion)
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train = train_datagen.flow_from_directory(
    "dataset/train",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val = val_datagen.flow_from_directory(
    "dataset/val",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# 🔥 Class Weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train.classes),
    y=train.classes
)
class_weights = dict(enumerate(class_weights))

# 🔥 Model
base = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freeze all layers initially
for layer in base.layers:
    layer.trainable = False

# 🔥 Improved Head
x = base.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(256, activation='relu')(x)   # 🔥 reduced (avoid overfitting)
x = layers.Dropout(0.4)(x)
output = layers.Dense(train.num_classes, activation='softmax')(x)

model = models.Model(inputs=base.input, outputs=output)

# 🔥 Compile Stage 1
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 🔥 Callbacks (improved)
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=2,
    min_lr=1e-6
)

checkpoint = ModelCheckpoint(
    "best_model.keras",
    monitor='val_accuracy',
    save_best_only=True
)

# 🚀 Stage 1
history1 = model.fit(
    train,
    validation_data=val,
    epochs=10,
    callbacks=[early_stop, reduce_lr, checkpoint],
    class_weight=class_weights
)

# 🔥 Fine-tuning (IMPORTANT IMPROVEMENT)
for layer in base.layers[-80:]:   # 🔥 more layers unfrozen
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 🚀 Stage 2
history2 = model.fit(
    train,
    validation_data=val,
    epochs=15,
    callbacks=[early_stop, reduce_lr, checkpoint],
    class_weight=class_weights
)

# Save
model.save("best_model.keras")

print("✅ Improved Model Trained!")

# 👇 PASTE GRAPH CODE HERE

import matplotlib.pyplot as plt

# Accuracy
acc = history1.history['accuracy'] + history2.history['accuracy']
val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']

epochs = range(1, len(acc) + 1)

plt.figure()
plt.plot(epochs, acc, label='Training Accuracy')
plt.plot(epochs, val_acc, label='Validation Accuracy')

plt.title('Training vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Loss
loss = history1.history['loss'] + history2.history['loss']
val_loss = history1.history['val_loss'] + history2.history['val_loss']

plt.figure()
plt.plot(epochs, loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')

plt.title('Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()