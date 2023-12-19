import os

import cv2
from flask import Flask, request, flash, Response, send_file
from werkzeug.utils import secure_filename

from utils import solvePuzzle

# from utils import find_puzzle

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'webp', 'jpeg'}


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def root():
    # return Response(data, mimetype="text/json", ), 200
    # return send_file('output/output.jpg', mimetype='image/jpg'), 200
    return "Hello World"


@app.route('/uploadPuzzle', methods=['POST'])
def uploadPuzzle():
    # return "Hello Puzzle"
    if 'file' not in request.files:
        return "No file uploaded!"
    # else:
    #     return "File Uploaded!"

    file = request.files['file']

    if file.filename == '':
        flash('No selected file')
        return "No File Uploaded!"

    if file:
        filename = secure_filename(file.filename)
        filename = request.form.get('ip')+'_'+filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        print(filename)
        solvePuzzle(filename)
        image = "solvedPuzzles/solved_"+filename
        return send_file(image, mimetype="image/png", as_attachment=True), 200


if __name__ == '__main__':
    # print(keywords)
    # print("Hello World!")
    # find_puzzle(imagePath='image.webp')
    app.run(debug=True, port=48274, host="192.168.56.1")
    # find_puzzle('image.webp')


