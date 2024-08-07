from waitress import serve
from API import app  # Importa la aplicación Flask desde API.py

if __name__ == "__main__":
    serve(app, host='0.0.0.0', port=5000)