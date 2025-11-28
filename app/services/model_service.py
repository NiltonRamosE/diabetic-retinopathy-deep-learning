import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

# CLASSES de acuerdo con la descripción
CLASSES = ['Mild', 'Moderate', 'No_DR', 'Proliferate_DR', 'Severe']

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'densenet121.h5')

# Cargar el modelo (fuera de la función para cargar solo una vez)
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

def predict_image(image_bytes, return_all_probs=False):
    """
    Realiza la predicción de la imagen usando el modelo entrenado.

    :param image_bytes: Imagen en bytes
    :param return_all_probs: Si True, devuelve también las probabilidades de todas las clases
    :return: dict con 'label' y 'confidence' (float porcentaje). Si return_all_probs=True, incluye 'probs' (dict)
    """
    # Abrir la imagen en formato PIL
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    # Preprocesamiento
    image = image.resize((224, 224))
    image_array = np.array(image).astype('float32') / 255.0

    # Asegurar 3 canales
    if image_array.ndim == 2:
        image_array = np.stack([image_array] * 3, axis=-1)
    if image_array.shape[-1] != 3:
        image_array = image_array[..., :3]

    # Batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Predict (modelo puede devolver logits o probabilidades)
    preds = model.predict(image_array)  # shape (1, num_classes) o (1, )
    preds = np.asarray(preds)

    # Si salida es escalar (regresión) -> convertir a array
    if preds.ndim == 1:
        preds = np.expand_dims(preds, axis=0)

    # Convertir a probabilidades de forma segura
    try:
        probs = tf.nn.softmax(preds, axis=-1).numpy()[0]
    except Exception:
        # Si softmax falla (p. ej. ya son probabilidades), normalizamos
        probs = preds[0]
        probs = probs / (probs.sum() + 1e-12)

    top_idx = int(np.argmax(probs))
    top_label = CLASSES[top_idx]
    confidence = float(probs[top_idx] * 100.0)  # porcentaje

    result = {
        'label': top_label,
        'confidence': round(confidence, 2)  # dos decimales
    }

    if return_all_probs:
        # devolver probabilidades por clase también
        probs_dict = {CLASSES[i]: float(round(float(probs[i] * 100.0), 2)) for i in range(len(CLASSES))}
        result['probs'] = probs_dict

    return result
