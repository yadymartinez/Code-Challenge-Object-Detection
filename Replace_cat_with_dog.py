# Step 1: Install YOLOv8 package
!pip install ultralytics

# Step 2: Import necessary libraries
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
from google.colab.patches import cv2_imshow  # For displaying images in Colab
from google.colab import files


# Step 4: Function to cut a piece of the image and replace it with another image
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
    # Load the main image and the replacement image
    main_image = cv2.imread(image_path_cat)
    replace_image = cv2.imread(image_path_dog)

    # Get the dimensions of the region to replace (ROI)
    roi_width = x2 - x1
    roi_height = y2 - y1


    # Resize the replacement image to fit the ROI in the main image
    replace_image_resized = cv2.resize(replace_image, (roi_width, roi_height))

    # Replace the region in the main image with the resized replacement image
    main_image[y1:y2, x1:x2] = replace_image_resized

    # Display the modified image
    cv2_imshow(main_image)
