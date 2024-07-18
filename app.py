import tensorflow as tf
from flask import Flask, render_template, request,url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from PIL import Image
from io import BytesIO

app = Flask(__name__)

model_path = 'C:\\Users\\aaron\\Project\\fake_logo.h5'
model = load_model(model_path)



@app.route('/', methods=['GET', 'POST'])
def predict_image():
    if request.method == 'POST':
        # Get the uploaded image from the request
        image = request.files['image']
        
        # Convert the FileStorage object to BytesIO
        image_data = image.read()
        image_stream = BytesIO(image_data)
        
        # Load and preprocess the image
        img = load_img(image_stream, target_size=(256, 256))
        img = img_to_array(img)
        img = img / 255.0  # Normalize the image
        
        # Reshape the image to match the model input shape
        img = np.expand_dims(img, axis=0)
        
        # Make predictions
        predictions = model.predict(img)
        
        # Get the predicted class label
        class_index = np.argmax(predictions)
        class_labels = ['Fake', 'Real']
        class_label = class_labels[class_index]
        
        return render_template('result.html' ,class_label=class_label)
    
    return render_template('predict.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/info')
def info():
    return render_template('info.html')



if __name__ == '__main__':
    app.run(debug=True)
