# Code-Challenge-Object-Detection
Theoretical Question about Object Detection, Basic implementation using a pre-trained YOLOv8 model from KerasCV and  to detect and locate a cat in an image
## A first-level heading
Image Classification helps us to classify what is contained in an image. Image Localization will specify the location of a single object in an image whereas Object 
Detection specifies the location of multiple objects in the image.
There are three types of image classification:
•	Binary classification: Tags images with one of two possible labels (e.g., dog or not dog)
•	Multi-class classification: Tags images with one of many possible labels (e.g., dog, cat, bird, squirrel, rabbit, deer, coyote, fox, etc.). Each image fits into one category.
•	Multi-label classification: Can tag images with multiple labels (e.g., black, white, brown, red, orange, yellow, etc.) Each image can fit into multiple categories.

Object detection, on the other hand, identifies the location and number of specific objects in an image. Some businesses use object detection similar to how people use 
image classification: to detect if something is present in an image. While image classification is typically the preferred method for this, users sometimes prefer object
detection when the target object is relatively small in the image.

How to choose: image classification vs. object detection
Choose image classification when:
•	You want to classify or sort entire images into buckets.
•	The location or number of objects in the image isn’t important.
Choose object detection when:
•	You need to identify the location or count of an object.
•	The object you want identified is just a small part of the image and/or your images are noisy (aka, there’s a lot going on beyond just the object).

Object detection model architectures
With object detection, though, CNNs are still the better choice, as they work better for bounding box detection. 
Currently the YOLO series from Ultralytics, which uses CNNs, is considered the most advanced object detection model.

## A first-level heading
