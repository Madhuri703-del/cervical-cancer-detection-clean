from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Load model
model = load_model("model.keras")

# Correct preprocessing (IMPORTANT 🔥)
test = ImageDataGenerator(
    preprocessing_function=preprocess_input
).flow_from_directory(
    "dataset/test",
    target_size=(224,224),
    shuffle=False
)

# Predictions
preds = model.predict(test)
y_pred = np.argmax(preds, axis=1)

# Results
print("\n📊 Classification Report:")
print(classification_report(test.classes, y_pred))

print("\n📊 Confusion Matrix:")
print(confusion_matrix(test.classes, y_pred))