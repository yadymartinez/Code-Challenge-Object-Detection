# Step 1: Install YOLOv8 package
!pip install ultralytics

# Step 2: Import necessary libraries
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
from google.colab.patches import cv2_imshow  # For displaying images in Colab
import urllib.request
from google.colab import files

# Step 3: Define the function to detect cats using YOLOv8
def detect_cat(image_path_cat):
    """
    Detects cats in an image using a pre-trained YOLOv8 model.

    Parameters:
    image_path (str): The path to the image where cats will be detected.

    Returns:
     x1, y1, x2, y2 (int): Coordinates for the top-left (x1, y1) and bottom-right (x2, y2)
                              of the region to be detected in the cat image.
    """
    # Load the pre-trained YOLOv8 model (trained on COCO dataset)
    model = YOLO('yolov8s.pt')  # You can use 'yolov8s.pt' for a larger model

    # Perform inference on the image
    results = model(image_path_cat)

    # COCO class ID for 'cat' is 15
    cat_class_id = 15

    # Load the image using OpenCV
    img = cv2.imread(image_path_cat)

    # Iterate over the results to find cats
    for result in results:
        for box in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = box

            # If the detected object is a cat (class_id 15)
            if int(class_id) == cat_class_id:
                print(f"Cat detected with confidence {score:.2f} at [({x1}, {y1}), ({x2}, {y2})]")

                # Draw bounding box around the detected cat
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(img, f"Cat: {score:.2f}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the image with the detected cat(s) highlighted
    cv2_imshow(img)
    return int(x1), int(y1), int(x2), int(y2)



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





# Step 5: Upload or load an image
uploaded = files.upload()  # Upload the image file
image_path_cat = list(uploaded.keys())[0]


# Step 6: Upload or load an image
uploaded = files.upload()  # Upload the image file
image_path_dog = list(uploaded.keys())[0]

xc1, yc1, xc2, yc2 = detect_cat(image_path_cat)

modified_image = replace_image_piece(image_path_cat, image_path_dog, xc1, yc1, xc2, yc2)