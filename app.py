from flask import Flask, request, jsonify
from model.predict import predict
from PIL import Image

app = Flask(__name__)


@app.route('/captcha-label', methods=['POST'])
def return_captcha_label():
    try:    
        file_storage = request.files.get('file')
        file_name = file_storage.filename
        img = Image.open(file_storage.stream)
        label =  predict(img)
        return jsonify({'status_code': 200, 'filename': file_name, 'predict': label})
    except Exception as e:
        return jsonify({'status_code': 500, 'message': str(e)})


if __name__ == '__main__':
    app.run()
