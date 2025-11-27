from flask import Blueprint, request, jsonify
from app.services.model_service import predict_image

classification_controller = Blueprint('classification_controller', __name__)

@classification_controller.route('/classify', methods=['POST'])
def classify_image():
    """
    Endpoint para recibir una imagen y devolver la clasificación.
    :return: La clasificación del modelo.
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    # Obtener la imagen de la solicitud
    image_file = request.files['image']

    # Leer los bytes de la imagen
    image_bytes = image_file.read()

    # Hacer la predicción
    prediction = predict_image(image_bytes)

    # Devolver la predicción
    return jsonify({'prediction': prediction})
