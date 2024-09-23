## Computer Vision and Image Manipulation Interview Questions
### Explain the difference between object detection and image classification.
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

###  What is transfer learning, and how can it be useful in object detection tasks?

Transfer learning, used in machine learning, is the reuse of a pre-trained model on a new problem. In transfer learning, a machine exploits the knowledge gained from a previous task to improve generalization about another. For example, in training a classifier to predict whether an image contains food, you could use the knowledge it gained during training to recognize drinks.
Transfer learning is a powerful technique in deep learning that allows you to leverage pre-trained models to boost the performance of your object detection system. Instead of starting from scratch, you can finetune a preexisting model, often trained on a largescale dataset, to adapt it to your specific object detection task.



### Describe the architecture of YOLO (You Only Look Once) and its advantages in real-time object detection.
YOLO (You Only Look Once) is a real-time object detection system, first proposed in 2015 by Joseph Redmon and Ali Farhadi. The architecture is designed to perform bounding box detection and class prediction of objects in a single convolutional network scan, hence the name "You Only Look Once" (1).
YOLO, a new approach to object detection. Prior work on object detection repurposes classifiers to perform detection. Instead, we frame object detection as a regression problem to spatially separated bounding boxes and associated class probabilities. A single neural network predicts bounding boxes and class probabilities directly from full images in one evaluation. Since the whole detection pipeline is a single network, it can be optimized end-to-end directly on detection performance.
The YOLO algorithm revolutionized object detection by framing it as a single regression problem, predicting bounding boxes and class probabilities directly from full images in one evaluation. This streamlined approach has made YOLO synonymous with real-time detection capabilities.

Let us first understand how YOLO encodes its output,
1. Input image is divided into NxN grid cells. For each object present on image, one grid cell is responsible for predicting object.
2. Each grid predicts ‘B’ bounding box and ‘C’ class probabilities. And bounding box consist of 5 components (x,y,w,h,confidence)
(x,y) = coordinates representing center of box
(w,h) = width and height of box
Confidence = represents presence/absence of any object

YOLO (You Only Look Once) has gained popularity in the field of object detection for several reasons:
1.	Speed: YOLO is incredibly fast because it processes the entire image in a single pass through the neural network. This allows it to achieve real-time performance, processing images at up to 45 frames per second (FPS) or more1.
2.	High Detection Accuracy: YOLO is known for its high accuracy in detecting objects with minimal background errors. It achieves this by framing object detection as a regression problem, predicting bounding boxes and class probabilities directly from full images1.
3.	Simplicity: The architecture of YOLO is straightforward, using a single convolutional neural network (CNN) to predict multiple bounding boxes and class probabilities simultaneously. This simplicity makes it easier to implement and optimize2.
4.	Better Generalization: YOLO has strong generalization capabilities, meaning it performs well on new, unseen data. This is particularly true for newer versions of YOLO, which have improved in handling various object sizes and aspect ratios1.
5.	Context Awareness: Unlike traditional methods that use sliding windows or region proposals, YOLO looks at the entire image during training and testing. This helps it understand the context of objects within the image, reducing false positives2.
6.	Open Source: YOLO is open-source, which makes it accessible to a wide range of developers and researchers. This has led to a large community contributing to its development and improvement1.


### What is a Generative Adversarial Network (GAN), and how could it be used in image manipulation tasks?