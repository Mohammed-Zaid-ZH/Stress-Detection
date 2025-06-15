from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
import joblib
import traceback
import cv2
import numpy as np
import dlib
import os
from werkzeug.utils import secure_filename
import real  # Assuming 'real' is your custom module for image processing

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load pre-trained models
mlp = joblib.load('xgb_model.joblib')  # Using the most recent model mentioned
ss = joblib.load('scaler.joblib')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Helper function to validate file type
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/a')
def page_a():
    return render_template('a.html')

@app.route('/image')
def image_page():
    return render_template('image.html')

# Route to handle image processing
@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        image_file = request.files['image']

        if image_file and allowed_file(image_file.filename):
            filename = secure_filename(image_file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image_file.save(filepath)

            print(f"Image saved at: {filepath}")  # Debugging print statement

            # Process the image
            return process_image_file(filepath, filename)

        return jsonify({'error': 'Invalid file type. Only JPG, JPEG, and PNG are allowed.'}), 400

    except Exception as e:
        print(f"Exception occurred: {str(e)}")  # Log the exception
        traceback.print_exc()
        return jsonify({'error': f'Internal error: {str(e)}'}), 500

# Function to process the uploaded image
def process_image_file(image_path, filename):
    try:
        # Ensure the image file exists
        if not os.path.exists(image_path):
            return jsonify({'error': f'File does not exist at path: {image_path}'}), 400

        # Process the image using the custom `real` module
        result = real.out(image_path)
        print(f"Image processing result: {result}")  # Debugging print statement

        # Map model results to labels
        if result == 1:
            status = "Stressed"
        elif result == 0:
            status = "Not Stressed"
        elif result == 2:
            status = "Neutral or Undetermined"
        else:
            return jsonify({'error': 'Unknown result from image processing'}), 400

        # Return the processed image URL and result
        print(f"Attempting to delete image at path: {image_path}")
        delete_image(image_path)
        return jsonify({'result': status})

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Internal error: {str(e)}'}), 500
def delete_image(file_path):
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Image {file_path} deleted successfully.")
        else:
            print(f"Image {file_path} does not exist.")
    except Exception as e:
        print(f"Error deleting image {file_path}: {str(e)}")
# Serve uploaded images dynamically
@app.route('/image/<filename>', methods=['GET'])
def serve_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
