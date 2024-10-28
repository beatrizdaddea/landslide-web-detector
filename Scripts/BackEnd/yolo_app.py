from flask import Flask, render_template, request
import numpy as np
import base64
import cv2
import os
from ultralytics import YOLO

model = YOLO('Scripts/BackEnd/best.pt')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
UPLOAD_FOLDER = 'Scripts/Relatórios'  # Define o caminho da pasta para salvar as imagens

app = Flask(__name__, template_folder='../FrontEnd/templates', static_folder='../FrontEnd/static')

# Cria a pasta 'Relatórios' se não existir
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def resize_image(image, width=640, height=640):
    return cv2.resize(image, (width, height))

def predict_on_image(image_data):
    # Decodifica a imagem a partir dos bytes recebidos
    image = cv2.imdecode(np.frombuffer(image_data, dtype=np.uint8), cv2.IMREAD_COLOR)

    # Redimensiona a imagem original
    image = resize_image(image)

    results = model.predict(image, classes=0, conf=0.5)
    for i, r in enumerate(results):
        im_bgr = r.plot(conf=True)

    # Redimensiona a imagem de resposta
    im_bgr = resize_image(im_bgr)

    return im_bgr

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')

        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', error='No selected file')

        if file and allowed_file(file.filename):
            # Caminho para salvar a imagem original na pasta 'Relatórios'
            original_save_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(original_save_path)

            # Reinicia o ponteiro do stream e lê o conteúdo para evitar esgotamento
            file.stream.seek(0)
            image_data = file.stream.read()

            # Executa a predição na imagem
            predicted_image = predict_on_image(image_data)

            # Salva a imagem processada (detecção) na pasta 'Relatórios'
            processed_filename = f"processed_{file.filename}"
            processed_save_path = os.path.join(UPLOAD_FOLDER, processed_filename)
            cv2.imwrite(processed_save_path, predicted_image)

            # Converte a imagem processada e a original para base64 para exibir no HTML
            retval, buffer = cv2.imencode('.png', predicted_image)
            detection_img_base64 = base64.b64encode(buffer).decode('utf-8')

            original_image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
            original_image = resize_image(original_image)
            retval, buffer = cv2.imencode('.png', original_image)
            original_img_base64 = base64.b64encode(buffer).decode('utf-8')

            return render_template('result.html', original_img_data=original_img_base64, detection_img_data=detection_img_base64)

    return render_template('index.html')

@app.route('/info')
def info():
    return render_template('info.html')

@app.route('/sobre_nos')
def sobre_nos():
    return render_template('about.html')

@app.route('/duvida')
def duvida():
    return render_template('questions.html')

if __name__ == '__main__':
    os.environ.setdefault('FLASK_ENV', 'development')
    app.run(debug=False, port=5000, host='0.0.0.0')
