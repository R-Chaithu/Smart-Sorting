from flask import Flask, render_template, request, url_for, redirect
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import numpy as np
import os
import tensorflow as tf

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('healthy_vs_rotten.h5')

# Class labels
CLASS_INDEX = [
    'Apple_Healthy (0)', 'Apple_Rotten (1)',
    'Banana_Healthy (2)', 'Banana_Rotten (3)',
    'Bellpepper_Healthy (4)', 'Bellpepper_Rotten (5)',
    'Carrot_Healthy (6)', 'Carrot_Rotten (7)',
    'Cucumber_Healthy (8)', 'Cucumber_Rotten (9)',
    'Grape_Healthy (10)', 'Grape_Rotten (11)',
    'Guava_Healthy (12)', 'Guava_Rotten (13)',
    'Jujube_Healthy (14)', 'Jujube_Rotten (15)',
    'Mango_Healthy (16)', 'Mango_Rotten (17)',
    'Orange_Healthy (18)', 'Orange_Rotten (19)',
    'Pomegranate_Healthy (20)', 'Pomegranate_Rotten (21)',
    'Potato_Healthy (22)', 'Potato_Rotten (23)',
    'Strawberry_Healthy (24)', 'Strawberry_Rotten (25)',
    'Tomato_Healthy (26)', 'Tomato_Rotten (27)'
]


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Check if file was uploaded
        if 'pc_image' not in request.files:
            return redirect(request.url)

        f = request.files['pc_image']

        # If no file selected
        if f.filename == '':
            return redirect(request.url)

        # Create uploads directory if not exists
        upload_dir = 'static/uploads'
        os.makedirs(upload_dir, exist_ok=True)

        # Save the uploaded file
        img_path = os.path.join(upload_dir, f.filename)
        f.save(img_path)

        # Load and preprocess the image
        try:
            img = load_img(img_path, target_size=(224, 224))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0  # Normalize

            # Make prediction
            pred = model.predict(img_array)
            pred_class = np.argmax(pred, axis=1)
            prediction = CLASS_INDEX[int(pred_class)]

            print(f"Prediction: {prediction}")

            return render_template("portfolio-details.html",
                                   predict=prediction,
                                   image_path=img_path)

        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return render_template("error.html", error=str(e))

    return redirect(url_for('index'))


if __name__ == "__main__":
    app.run(debug=True, port=2222)