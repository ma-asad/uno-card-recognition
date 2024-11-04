from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from flask_cors import CORS
import cv2
import numpy as np
import os

from orb import UNOCardDetector  # Import the UNOCardDetector class

# Initialize Flask app with correct template and static folder paths
app = Flask(__name__,
            template_folder='templates',
            static_folder='static')

CORS(app)

# Define upload and card images folders relative to 'app/'
UPLOAD_FOLDER = os.path.join(app.root_path, 'uploads')
# Folder containing card images
CARD_IMAGES_FOLDER = os.path.join(app.root_path, 'static', 'cards')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize the UNO card detector with correct templates path
uno_card_detector = UNOCardDetector(
    templates_path=os.path.join(app.root_path, 'static', 'img', '*.jpg'))


def detect_uno_card(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return "Unknown Card"
    detected_card = uno_card_detector.detect_card(image)
    return detected_card

# Route to serve the index.html


@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image uploads


@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    source = request.form.get('source', 'unknown')
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        message = "Received frame from camera" if source == 'camera' else "Received uploaded image"
        print(message)

        card_name = detect_uno_card(file_path)
        os.remove(file_path)

        # Generate the base card name to match the high-resolution image
        base_card_name = '_'.join(card_name.split('_')[:2])
        # Generate the card image filename
        card_image_filename = f"{base_card_name}.jpg"
        # Generate the card image path using absolute paths
        card_image_path = os.path.join(CARD_IMAGES_FOLDER, card_image_filename)

        # Print statements for debugging
        print(f"Detected card name: {card_name}")
        print(f"Base card name: {base_card_name}")
        print(f"Card image filename: {card_image_filename}")
        print(f"Card image path: {card_image_path}")

        if os.path.exists(card_image_path):
            # Note the leading slash
            card_image_url = f"/static/cards/{card_image_filename}"
        else:
            card_image_url = None  # Or a placeholder image

        # Print the card image URL
        print(f"Card image URL: {card_image_url}")

        return jsonify({
            'card': card_name,
            'message': message,
            'card_image_url': card_image_url
        })

# Remove the unnecessary route to serve static card images
# Flask serves static files from the 'static' folder by default


if __name__ == '__main__':
    print("Server starting... Access the application at http://127.0.0.1:5500/")
    app.run(debug=True, port=5500)
