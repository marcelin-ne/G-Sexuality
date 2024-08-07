from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import model_from_json
from io import BytesIO
from flask_cors import CORS  # Importa CORS

app = Flask(__name__)
CORS(app) 

# Cargar el modelo al iniciar la aplicación
json_path = 'models/transfer_inceptionV3.json'
weights_path = 'models/transfer_inceptionV3.h5'

with open(json_path, 'r') as json_file:
    loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights(weights_path)

def load_and_preprocess_image(image_path, target_size):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def predict_class(image_path, target_size):
    img_array = load_and_preprocess_image(image_path, target_size)
    predictions = loaded_model.predict(img_array)
    return predictions

class_labels = {
    0: 'Emotivo',
    1: 'Instintivo',
    2: 'Racional'
}

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        img = image.load_img(BytesIO(file.read()), target_size=(299, 299))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        predictions = loaded_model.predict(img_array)
        predicted_class = np.argmax(predictions)
        predicted_class_name = class_labels[predicted_class]
        probabilities = {class_labels[i]: float(predictions[0][i]) for i in range(len(predictions[0]))}
        return jsonify({'predicted_class': predicted_class_name, 'probabilities': probabilities})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


#Índice: 0 | Clase: Emotivo
#Índice: 1 | Clase: Instintivo
#Índice: 2 | Clase: Racional