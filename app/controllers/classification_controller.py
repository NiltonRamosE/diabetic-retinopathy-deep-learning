from flask import Blueprint, request, jsonify
from app.services.model_service import predict_image

classification_controller = Blueprint('classification_controller', __name__)

@classification_controller.route('/classify', methods=['POST'])
def classify_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']
    image_bytes = image_file.read()

    # Si quieres también las probabilidades de todas las clases, pasa return_all_probs=True
    result = predict_image(image_bytes, return_all_probs=False)

    response = {'prediction': result}
    return jsonify(response)
