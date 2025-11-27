import tensorflow as tf
import numpy as np
from PIL import Image
import io

# CLASSES de acuerdo con la descripción
CLASSES = ['Mild', 'Moderate', 'No_DR', 'Proliferate_DR', 'Severe']

# Cargar el modelo preentrenado
model = tf.keras.models.load_model('densenet121.h5')

def predict_image(image_bytes):
    """
    Realiza la predicción de la imagen usando el modelo entrenado.
    
    :param image_bytes: Imagen en bytes
    :return: Predicción de la categoría
    """
    # Abrir la imagen en formato PIL
    image = Image.open(io.BytesIO(image_bytes))

    # Preprocesamiento de la imagen para que sea compatible con el modelo
    image = image.resize((224, 224))  # Redimensionar según las necesidades del modelo DenseNet
    image_array = np.array(image) / 255.0  # Normalizar la imagen a [0, 1]
    
    # Si la imagen tiene un solo canal (escala de grises), hacerla de 3 canales
    if len(image_array.shape) == 2:
        image_array = np.stack([image_array] * 3, axis=-1)
    
    # Asegurarse de que la imagen tiene 3 canales (RGB)
    if image_array.shape[-1] != 3:
        image_array = np.concatenate([image_array] * 3, axis=-1)
    
    # Añadir una dimensión de batch (para que el modelo reciba la entrada correctamente)
    image_array = np.expand_dims(image_array, axis=0)

    # Realizar la predicción
    predictions = model.predict(image_array)
    predicted_class_index = np.argmax(predictions, axis=-1)[0]
    
    return CLASSES[predicted_class_index]
