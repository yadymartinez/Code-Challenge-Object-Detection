!pip install tensorflow keras-cv
# Import necessary libraries

import tensorflow as tf
import cv2
from PIL import Image
import keras_cv
import matplotlib.pyplot as plt
import numpy as np
from google.colab import files
from google.colab.patches import cv2_imshow  # To display images in Colab

# Load pre-trained YOLOv8 model
model = keras_cv.models.YOLOV8Detector.from_preset(
    "yolo_v8_m_pascalvoc", bounding_box_format="xywh",
)

# Define the function to detect cats
def detect_cat(image_path):
    """
    Function to detect cats in an image using YOLOv8 model.

    Args:
        image_path (str): Path to the image file.

    Returns:
        None: The function displays the image with detected cats.
    """
    # Load the image using PIL
    image = Image.open(image_path)

    # Resize the image
    image_resized = tf.image.resize(image, (640, 640))[None, ...]


     # Make predictions on the image
    predictions  = model.predict(image_resized)

    # COCO class ID for 'cat' is 7
    cat_label = 7

    # Create a plot to display the image with bounding boxes
    plt.figure(figsize=(8, 8))
    plt.imshow(image)

    # KerasCV returns a dictionary with predictions: boxes, scores, and labels
    boxes = predictions['boxes'][0]  # Bounding boxes
    scores = predictions['confidence'] [0] # Confidence scores
    labels = predictions['classes'][0] # Class labels
   
    # Create a plot to display the image with bounding boxes
    plt.figure(figsize=(8, 8))
    plt.imshow(image)

    for i in range(len(labels)):
        if labels[i] == cat_label and scores[i] > 0.5:  # Filter for cats and high-confidence scores
         # Get the coordinates of the bounding box
         x1, y1, w, h = boxes[i]         
         # Draw a rectangle around the detected cat
         plt.gca().add_patch(plt.Rectangle((int(x1), int(y1)), int(h), int(w),
                                          edgecolor='red', facecolor='none', linewidth=2))
         # Optionally, label the box with the score
         plt.text(x1, y1, f"Cat: {scores[i]:.2f}", color='white', backgroundcolor='red')

    # Remove axis and display the plot
    plt.axis("off")
    plt.show()


# To use the function, upload an image and call detect_cat()

from google.colab import files

# Upload an image
uploaded = files.upload()  # Prompt to upload the image
image_path = list(uploaded.keys())[0]

# Call the function to detect cats
detect_cat(image_path)
