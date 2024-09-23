## Computer Vision and Image Manipulation Interview Questions
### Explain the difference between object detection and image classification.
Image Classification helps us to classify what is contained in an image. Image Localization will specify the location of a single object in an image whereas Object 
Detection specifies the location of multiple objects in the image.
There are three types of image classification:
* Binary classification: Tags images with one of two possible labels (e.g., dog or not dog)
* Multi-class classification: Tags images with one of many possible labels (e.g., dog, cat, bird, squirrel, rabbit, deer, coyote, fox, etc.). Each image fits into one category.
* Multi-label classification: Can tag images with multiple labels (e.g., black, white, brown, red, orange, yellow, etc.) Each image can fit into multiple categories.

Object detection, on the other hand, identifies the location and number of specific objects in an image. Some businesses use object detection similar to how people use 
image classification: to detect if something is present in an image. While image classification is typically the preferred method for this, users sometimes prefer object
detection when the target object is relatively small in the image.

How to choose: image classification vs. object detection
Choose image classification when:
* You want to classify or sort entire images into buckets.
* The location or number of objects in the image isn’t important.
Choose object detection when:
* You need to identify the location or count of an object.
* The object you want identified is just a small part of the image and/or your images are noisy (aka, there’s a lot going on beyond just the object).

Object detection model architectures
With object detection, though, CNNs are still the better choice, as they work better for bounding box detection. 
Currently the YOLO series from Ultralytics, which uses CNNs, is considered the most advanced object detection model.

###  What is transfer learning, and how can it be useful in object detection tasks?

Transfer learning, used in machine learning, is the reuse of a pre-trained model on a new problem. In transfer learning, a machine exploits the knowledge gained from a previous task to improve generalization about another. For example, in training a classifier to predict whether an image contains food, you could use the knowledge it gained during training to recognize drinks.
Transfer learning is a powerful technique in deep learning that allows you to leverage pre-trained models to boost the performance of your object detection system. Instead of starting from scratch, you can finetune a preexisting model, often trained on a largescale dataset, to adapt it to your specific object detection task.



### Describe the architecture of YOLO (You Only Look Once) and its advantages in real-time object detection.
YOLO (You Only Look Once) is a real-time object detection system, first proposed in 2015 by Joseph Redmon and Ali Farhadi. The architecture is designed to perform bounding box detection and class prediction of objects in a single convolutional network scan, hence the name "You Only Look Once" .

YOLO, a new approach to object detection. Prior work on object detection repurposes classifiers to perform detection. Instead, we frame object detection as a regression problem to spatially separated bounding boxes and associated class probabilities. A single neural network predicts bounding boxes and class probabilities directly from full images in one evaluation. Since the whole detection pipeline is a single network, it can be optimized end-to-end directly on detection performance.
The YOLO algorithm revolutionized object detection by framing it as a single regression problem, predicting bounding boxes and class probabilities directly from full images in one evaluation. This streamlined approach has made YOLO synonymous with real-time detection capabilities.

Let us first understand how YOLO encodes its output,
1. Input image is divided into NxN grid cells. For each object present on image, one grid cell is responsible for predicting object.
2. Each grid predicts ‘B’ bounding box and ‘C’ class probabilities. And bounding box consist of 5 components (x,y,w,h,confidence)

* (x,y) = coordinates representing center of box
* (w,h) = width and height of box
* Confidence = represents presence/absence of any object

YOLO (You Only Look Once) has gained popularity in the field of object detection for several reasons:
1.	Speed: YOLO is incredibly fast because it processes the entire image in a single pass through the neural network. This allows it to achieve real-time performance, processing images at up to 45 frames per second (FPS) or more.
2.	High Detection Accuracy: YOLO is known for its high accuracy in detecting objects with minimal background errors. It achieves this by framing object detection as a regression problem, predicting bounding boxes and class probabilities directly from full images.
3.	Simplicity: The architecture of YOLO is straightforward, using a single convolutional neural network (CNN) to predict multiple bounding boxes and class probabilities simultaneously. This simplicity makes it easier to implement and optimize.
4.	Better Generalization: YOLO has strong generalization capabilities, meaning it performs well on new, unseen data. This is particularly true for newer versions of YOLO, which have improved in handling various object sizes and aspect ratios.
5.	Context Awareness: Unlike traditional methods that use sliding windows or region proposals, YOLO looks at the entire image during training and testing. This helps it understand the context of objects within the image, reducing false positives.
6.	Open Source: YOLO is open-source, which makes it accessible to a wide range of developers and researchers. This has led to a large community contributing to its development and improvement.


### What is a Generative Adversarial Network (GAN), and how could it be used in image manipulation tasks?
A Generative Adversarial Network (GAN) is a type of machine learning model that consists of two neural networks, a Generator and a Discriminator, which are trained together in a competitive process. The primary objective of a GAN is to generate data (like images) that is indistinguishable from real data.

How GANs Work:
* Generator: The generator's role is to create fake data, such as synthetic images, starting from random noise. It tries to "fool" the discriminator by generating data that resembles real images.
* Discriminator: The discriminator evaluates data (both real and generated) and predicts whether it is real or fake. The goal of the discriminator is to correctly identify real images from fake ones.
During training:

The generator attempts to improve by generating more realistic data, learning to "trick" the discriminator.
The discriminator improves by becoming better at distinguishing between real and generated data.
This adversarial training leads to the generator producing highly realistic images over time.

Use of GANs in Image Manipulation:
GANs have been widely used in various image manipulation tasks because of their ability to generate and modify visual data in sophisticated ways. Some popular applications include:

1. Image Generation and Synthesis:
GANs can generate entirely new images that look realistic. For example, StyleGAN can create highly detailed, photorealistic images of people, animals, or objects that do not exist in reality.
2. Image Super-Resolution:
GANs are used to enhance the resolution of low-quality or pixelated images, producing sharp, high-definition images. This technique is known as Super-Resolution GAN (SRGAN).
3. Image Inpainting (Filling Missing Parts):
GANs can fill in missing or damaged parts of an image. For example, if a portion of a photo is corrupted, a GAN can generate the missing pixels to restore the image, ensuring the inpainted part blends seamlessly with the rest of the image.
4. Style Transfer:
GANs can alter the style of an image, such as converting a photograph into a painting or applying the artistic style of one image onto another. This is often done using CycleGAN, which can transform the style of an image without needing paired data (e.g., converting day images to night, or horses to zebras).
5. Face Manipulation:
GANs can be used to manipulate faces in various ways, such as altering facial expressions, age, or even gender. FaceApp and DeepFake technology are based on similar concepts, where GANs help modify face characteristics while preserving photorealism.
6. Image-to-Image Translation:
This involves converting an image from one domain to another. For instance, turning a sketch into a photorealistic image or converting a grayscale image into a colorized version. GANs like pix2pix are used in this task.
7. Content Creation for Games and Movies:
GANs are used to generate assets like textures, characters, or environments in creative industries. This helps streamline the production process, reducing the time and effort needed to manually create visual content.

Why GANs Are Powerful for Image Manipulation:

* Realism: GANs are known for their ability to produce highly realistic images that are often difficult to distinguish from real ones.
* Versatility: GANs can perform a wide range of tasks, from generating entirely new images to manipulating and enhancing existing ones.
* Learning from Unlabeled Data: GANs do not require labeled data, making them highly effective in situations where annotated datasets are unavailable.

