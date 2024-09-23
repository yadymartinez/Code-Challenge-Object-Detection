# Code-Challenge-Object-Detection


# Install tensorflow keras-cv package for Colab
!pip install tensorflow keras-cv

# Import necessary libraries
import tensorflow as tf

from PIL import Image

import keras_cv

import matplotlib.pyplot as plt

from google.colab import files

# Function to detect cats (Cat_Detection_KerasCV_Yolov8.py)
def detect_cat(image_path):

    """
    Function to detect cats in an image using YOLOv8 model.

    Args:
        image_path (str): Path to the image file.

    Returns:
     x1, y1, w, h (int): Coordinates for the top-left (x1, y1) and weigth, heigth (w, h)
                              of the region to be detected in the cat image.


     Pre-trained model: keras_cv.models.YOLOV8Detector (yolo_v8_m_pascalvoc).
                       
    """


# Install ultralytics package for YOLOv8
!pip install ultralytics

# Import necessary libraries
from ultralytics import YOLO

import tensorflow as tf

import matplotlib.pyplot as plt

import cv2

from google.colab import files

from google.colab.patches import cv2_imshow  # For displaying images in Colab

# Function to detect cats (Cat_Detection_Ultralytics.py)----Better Results
def detect_cat(image_path):

    """
    Function to detect cats in an image using YOLOv8 model.

    Args:
        image_path (str): Path to the image file.

    Returns:
     x1, y1, x2, y2 (int): Coordinates for the top-left (x1, y1) and bottom-right (x2, y2)
                              of the region to be detected in the cat image.


     Pre-trained model: Ultralytics YOLO (yolov8s.pt).
                       
    """

# Function to replace a detected cat with a pre-defined dog image (Replace_Cat_With_Dog.py) 

def replace_image_piece(image_path_cat, image_path_dog, x1, y1, x2, y2):

    """
    Cut a region from the main image and replace it with another image.

    Args:
        image_path_cat (str): Path to the main image.
        image_path_dog (str): Path to the image to replace the cut region.
        x1, y1, x2, y2 (int): Coordinates for the top-left (x1, y1) and bottom-right (x2, y2)
                              of the region to be replaced in the main image.

    Returns:
        None: Displays the modified image.
    """
    # Function to replace a detected cat with a pre-defined dog image (Replace_Cat_With_Dog.py)   that detects a cat in an image and replaces it with a dog.

# Function to replace a detected cat with a pre-defined dog image (Replace_Cat_With_Dog.py)    

def detect_cat_replace_image_piece(image_path_cat, image_path_dog):

    """
    Cut a region from the main image and replace it with another image.

    Args:
        image_path_cat (str): Path to the main image.
        image_path_dog (str): Path to the image to replace the cut region.

    Returns:
        None: Displays the modified image.
    """