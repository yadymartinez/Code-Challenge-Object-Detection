# Code-Challenge-Object-Detection

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
# Function to detect cats (Cat_Detection_Ultralytics.py)

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
    