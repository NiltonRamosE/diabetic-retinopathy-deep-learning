# Diabetic Retinopathy Deep Learning API

Microservicio en Python para la clasificación de retinopatía diabética usando un modelo entrenado **DenseNet121** (`densenet121.h5`).  
Expone un endpoint HTTP que recibe una imagen de fondo de ojo y devuelve la clase predicha y el porcentaje de confianza.

---

## Requisitos

- **Sistema operativo**: Windows / Linux / macOS
- **Python**: 3.9.25
- **Conda** (Anaconda o Miniconda) recomendado para gestionar el entorno

---

## Estructura del proyecto

```text
diabetic-retinopathy-deep-learning/
├── app
│   ├── controllers
│   │   └── classification_controller.py
│   ├── services
│       ├── model_service.py
│       └── models
│           └── densenet121.h5
├── requirements.txt
└── run.py
```

Puntos importantes:

- El archivo del modelo entrenado **`densenet121.h5`** debe estar en:
  ```text
  app/services/models/densenet121.h5
  ```
- `run.py` es el punto de entrada de la aplicación Flask.

---

## Creación del entorno con Conda

Desde una terminal / PowerShell:

```bash
# Crear entorno con Python 3.9
conda create -n diabetic-retinopathy python=3.9

# Activar el entorno
conda activate diabetic-retinopathy
```

Verifica la versión de Python:

```bash
python --version
# Debe mostrar algo como: Python 3.9.x
```

---

## Instalación de dependencias

Dentro del entorno `diabetic-retinopathy`, desde la raíz del proyecto:

```bash
pip install -r requirements.txt
```

El archivo `requirements.txt` contiene:

```text
Flask==2.2.2
Werkzeug==2.2.2
tensorflow==2.19.0
Pillow==9.0.1
numpy>=1.26.0
```

> Nota: Las versiones están pensadas para ser compatibles con **Python 3.9** y **TensorFlow 2.19.0**.

---

## Ejecución del microservicio

Con el entorno activado y las dependencias instaladas:

```bash
python run.py
```

Por defecto, Flask levantará el servidor en:

- URL: `http://0.0.0.0:5000`
- Localmente puedes acceder como: `http://localhost:5000`

---

## Detalles del código

### 1. `run.py`

Archivo principal que inicializa la app Flask y registra el blueprint del controlador:

```python
from flask import Flask
from app.controllers.classification_controller import classification_controller

app = Flask(__name__)
app.register_blueprint(classification_controller)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
```

### 2. `app/services/model_service.py`

Responsable de:

- Cargar el modelo `densenet121.h5` una sola vez.
- Preprocesar la imagen recibida.
- Obtener la predicción del modelo.
- Calcular la clase y el porcentaje de confianza.

Puntos clave:

- Clases de salida:

  ```python
  CLASSES = ['Mild', 'Moderate', 'No_DR', 'Proliferate_DR', 'Severe']
  ```

- Carga del modelo usando una ruta relativa segura:

  ```python
  BASE_DIR = os.path.dirname(os.path.abspath(__file__))
  MODEL_PATH = os.path.join(BASE_DIR, 'models', 'densenet121.h5')
  model = tf.keras.models.load_model(MODEL_PATH, compile=False)
  ```

- Preprocesamiento y predicción:

  ```python
  image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
  image = image.resize((224, 224))
  image_array = np.array(image).astype('float32') / 255.0
  image_array = np.expand_dims(image_array, axis=0)

  preds = model.predict(image_array)
  probs = tf.nn.softmax(preds, axis=-1).numpy()[0]

  top_idx = int(np.argmax(probs))
  top_label = CLASSES[top_idx]
  confidence = float(probs[top_idx] * 100.0)
  ```

### 3. `app/controllers/classification_controller.py`

Controlador HTTP (Blueprint) que expone el endpoint `/classify`:

```python
from flask import Blueprint, request, jsonify
from app.services.model_service import predict_image

classification_controller = Blueprint('classification_controller', __name__)

@classification_controller.route('/classify', methods=['POST'])
def classify_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']
    image_bytes = image_file.read()

    # Permitir opcionalmente devolver probs completas con ?probs=1
    return_all_probs = request.args.get('probs', '0') in ('1', 'true', 'True')
    result = predict_image(image_bytes, return_all_probs=return_all_probs)

    # Respuesta final envuelta en "prediction"
    response = {'prediction': result}
    return jsonify(response), 200
```

---

## Ejemplo de consumo del endpoint

### Petición con `curl`

```bash
curl -X POST   -F "image=@ruta/a/tu/imagen.jpg"   "http://localhost:5000/classify"
```

### Respuesta (sin `probs`):

```json
{
  "prediction": {
    "label": "No_DR",
    "confidence": 40.12
  }
}
```

---

## Notas adicionales

- Si cambias la ruta o el nombre del modelo (`densenet121.h5`), recuerda actualizar `MODEL_PATH` en `model_service.py`.
- Es recomendable versionar el archivo `requirements.txt` junto con el proyecto para asegurar reproducibilidad del entorno.