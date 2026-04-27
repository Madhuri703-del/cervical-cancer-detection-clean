import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings("ignore")

from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np

app = Flask(__name__)

IMG_SIZE = 224

classes = ['Dyskeratotic','Koilocytotic','Metaplastic','Parabasal','Superficial']

# ✅ Ensure uploads folder exists
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ✅ Load model safely
try:
    model = load_model("model.keras")
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Model loading error:", e)
    model = None

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template("index.html", prediction="Model not loaded!")

    file = request.files.get('file')

    if not file or file.filename == '':
        return render_template("index.html", prediction="No file selected!")

    if not allowed_file(file.filename):
        return render_template("index.html", prediction="Invalid file type!")

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    try:
        img = image.load_img(filepath, target_size=(IMG_SIZE, IMG_SIZE))
        img = image.img_to_array(img)
        img = preprocess_input(img)
        img = np.expand_dims(img, axis=0)

        pred = model.predict(img)[0]

        result_index = np.argmax(pred)
        result = classes[result_index]
        confidence = float(pred[result_index])

        # Cancer decision
        if result == "Superficial":
            cancer_status = "Normal"
        else:
            cancer_status = "Abnormal"

        labels = classes
        values = [round(float(p)*100, 2) for p in pred]

        return render_template(
            "index.html",
            prediction=f"{result} ({confidence*100:.2f}%)",
            cancer=cancer_status,
            labels=labels,
            values=values
        )

    except Exception as e:
        print("❌ Prediction error:", e)
        return render_template("index.html", prediction="Error processing image!")

# ✅ Correct run for deployment
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)