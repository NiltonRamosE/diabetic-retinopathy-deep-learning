import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

# CLASSES de acuerdo con la descripción
CLASSES = ['Mild', 'Moderate', 'No_DR', 'Proliferate_DR', 'Severe']

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'EfficientNetB0_DR_model_20251215_195032')

# Cargar el modelo usando tf.saved_model.load (para modelos exportados sin los metadatos Keras)
model = tf.saved_model.load(MODEL_PATH)
print("✔️ Modelo cargado exitosamente desde SavedModel.")

def predict_image(image_bytes, return_all_probs=False):
    """
    Realiza la predicción de la imagen usando el modelo cargado.

    :param image_bytes: Imagen en bytes
    :param return_all_probs: Si True, devuelve también las probabilidades de todas las clases
    :return: dict con 'label' y 'confidence' (float porcentaje). Si return_all_probs=True, incluye 'probs' (dict)
    """
    if not model:
        return {'error': 'Modelo no cargado, por favor verifique la ruta o formato del archivo.'}

    try:
        # Abrir la imagen en formato PIL
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    except Exception as e:
        return {'error': f"Error al abrir la imagen: {e}"}

    # Preprocesamiento de la imagen
    image = image.resize((224, 224))  # Redimensionar imagen a 224x224
    image_array = np.array(image).astype('float32') / 255.0  # Normalización

    # Asegurar 3 canales (RGB)
    if image_array.ndim == 2:  # Si la imagen es en escala de grises
        image_array = np.stack([image_array] * 3, axis=-1)
    if image_array.shape[-1] != 3:  # Si no tiene 3 canales, convertir
        image_array = image_array[..., :3]

    # Añadir dimensión de batch
    image_array = np.expand_dims(image_array, axis=0)

    try:
        # Realizar predicción
        print("🚀 Realizando la predicción...")
        infer = model.signatures["serving_default"]
        preds = infer(tf.convert_to_tensor(image_array))
        
        # El modelo debería retornar predicciones con una clave que contenga los logits
        logits = preds["dense_1"]  # Asegúrate de que esta clave coincida con tu modelo
        probs = tf.nn.softmax(logits).numpy()[0]
        
        top_idx = int(np.argmax(probs))
        top_label = CLASSES[top_idx]
        confidence = float(probs[top_idx] * 100.0)  # Convertir a porcentaje

        result = {
            'label': top_label,
            'confidence': round(confidence, 2)  # Redondear la confianza a dos decimales
        }

        # Si se requiere todas las probabilidades
        if return_all_probs:
            probs_dict = {CLASSES[i]: round(float(probs[i] * 100.0), 2) for i in range(len(CLASSES))}
            result['probs'] = probs_dict

        return result

    except Exception as e:
        return {'error': f"Error en la predicción: {e}"}
