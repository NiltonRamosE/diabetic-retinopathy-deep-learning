from flask import Flask
from app.controllers.classification_controller import classification_controller

# Crear la aplicación Flask
app = Flask(__name__)

# Registrar los controladores
app.register_blueprint(classification_controller)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
