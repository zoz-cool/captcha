from flask import Flask, request, jsonify
from model.main import Model
import base64
from PIL import Image

app = Flask(__name__)

m = Model()


@app.route('/captcha-label', methods=['POST'])
def return_captcha_label():
    file_storage = request.files.get('file')
    file_name = file_storage.filename
    img = Image.open(file_storage.stream)
    label =  m.predict(img)
    return jsonify({file_name: label})


if __name__ == '__main__':
    app.run()
