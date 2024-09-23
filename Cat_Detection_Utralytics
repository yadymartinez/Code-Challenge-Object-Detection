# Install ultralytics package for YOLOv8
!pip install ultralytics

# Import necessary libraries
from ultralytics import YOLO
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from google.colab import files
from google.colab.patches import cv2_imshow  # For displaying images in Colab

# Load YOLOv8 pre-trained model
model = YOLO('yolov8n.pt')  # Use smaller or larger model as needed


# Define a function to detect and locate a cat in the image, and save it
def detect_cat(image_path):
    """
    Detects cats in an image using a pre-trained YOLOv8 model (Ultralytics).
    
    Parameters:
    image_path (str): The path to the image where cats will be detected.
    
    Returns:
    None: Displays the image with detected cat(s) highlighted.
    """
    # Load the image using OpenCV
    image = cv2.imread(image_path)

    
    # Load the pre-trained YOLOv8 model (trained on COCO dataset)
    model = YOLO('yolov8s.pt')  # You can use 'yolov8s.pt' for a larger model
    
    # Perform inference on the image
    results = model(image_path)
          
    # COCO class ID for 'cat' is 15
    cat_class_id = 15
    
   
    # Iterate over the results to find cats
    for result in results:
        for box in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = box
            
            # If the detected object is a cat (class_id 15)
            if int(class_id) == cat_class_id:
                print(f"Cat detected with confidence {score:.2f} at [({x1}, {y1}), ({x2}, {y2})]")
                
                # Draw bounding box around the detected cat
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(image, f"Cat: {score:.2f}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                
    
    # Display the image with the detected cat(s) highlighted
    cv2_imshow(image)

   
# Upload and test image
from google.colab import files

# Upload an image
uploaded = files.upload()  # Prompt to upload the image
image_path = list(uploaded.keys())[0]

# Call the function to detect cats
detect_cat(image_path)
