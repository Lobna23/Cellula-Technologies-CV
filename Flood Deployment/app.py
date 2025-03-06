from flask import Flask, request, render_template, jsonify, send_from_directory
import numpy as np
import tensorflow as tf
import cv2
import tifffile as tiff
import os

app = Flask(__name__)

# Ensure directories exist
UPLOAD_FOLDER = "uploads"
MASK_FOLDER = "static/masks"
ORIGINAL_FOLDER = "static/originals"  # New folder for saving original images
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MASK_FOLDER, exist_ok=True)
os.makedirs(ORIGINAL_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MASK_FOLDER"] = MASK_FOLDER
app.config["ORIGINAL_FOLDER"] = ORIGINAL_FOLDER

# Load trained U-Net model
model = tf.keras.models.load_model("model/unet_model1.keras", compile=False)

def preprocess_image(image_path):
    """ Load and preprocess the TIFF image """
    image = tiff.imread(image_path)  # Load TIFF image
    image = cv2.resize(image, (128, 128))  # Resize to match model input
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def predict_mask(image_path):
    """ Predict segmentation mask using U-Net """
    image = preprocess_image(image_path)
    pred_mask = model.predict(image)[0]  # Get first prediction
    pred_mask = (pred_mask > 0.5).astype(np.uint8)  # Threshold

    # Resize back to original image size
    original_image = tiff.imread(image_path)
    pred_mask = cv2.resize(pred_mask, (original_image.shape[1], original_image.shape[0]))

    return pred_mask

def save_tiff_as_png(tiff_path, output_path):
    """ Convert specific bands of TIFF image to PNG for browser display """
    image = tiff.imread(tiff_path)

    if len(image.shape) == 3 and image.shape[2] >= 3:
        image_rgb = image[:, :, [3, 2, 1]]  # Use bands 4, 3, 2 (zero-based index)
    else:
        image_rgb = np.stack([image] * 3, axis=-1)  # Convert grayscale to 3-channel

    image_rgb = cv2.normalize(image_rgb, None, 0, 255, cv2.NORM_MINMAX)  # Normalize
    image_rgb = np.uint8(image_rgb)  # Convert to uint8
    cv2.imwrite(output_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))  # Save PNG



@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Save uploaded file
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    # Convert TIFF to PNG and save in static/
    original_png_filename = file.filename.replace(".tiff", ".png").replace(".tif", ".png")
    original_png_path = os.path.join("static", original_png_filename)

    print(f"Saving original image to: {original_png_path}")  # 🔥 Debugging

    try:
        save_tiff_as_png(file_path, original_png_path)
    except Exception as e:
        print(f"Error saving original PNG: {e}")
        return jsonify({"error": "Failed to process the original image."}), 500

    # Predict segmentation mask
    mask = predict_mask(file_path)

    # Save mask as PNG in static/
    mask_filename = file.filename.replace(".tiff", "_mask.png").replace(".tif", "_mask.png")
    mask_path = os.path.join("static", mask_filename)
    cv2.imwrite(mask_path, mask * 255)  # Convert mask to image format

    print(f"Original Image URL: /static/{original_png_filename}")  # 🔥 Debugging
    print(f"Mask Image URL: /static/{mask_filename}")  # 🔥 Debugging

    return jsonify({
        "original_url": f"/static/{original_png_filename}",
        "mask_url": f"/static/{mask_filename}"
    })

if __name__ == "__main__":
    app.run(debug=True)
