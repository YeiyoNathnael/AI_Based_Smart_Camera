# ai_processor.py

import torch
import torchvision
from torchvision import transforms
import cv2
import pytesseract
import numpy as np

# Load a pre-trained Faster R-CNN model from torchvision (set to evaluation mode)
# This model detects objects from the COCO dataset.
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Define the COCO class labels.
# (This is a subset for demonstration. You can extend it as needed.)
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
    'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
    'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
]

# Define the transform to convert the image to tensor and normalize it.
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts a PIL image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
])

def detect_objects(image_cv, threshold=0.8):
    """
    Detect objects in an image using a pre-trained Faster R-CNN model.
    
    Args:
        image_cv (numpy.ndarray): Image in OpenCV BGR format.
        threshold (float): Confidence threshold to filter detections.
    
    Returns:
        dict: Dictionary with key "objects" containing a list of detected object names.
    """
    # Convert BGR image (OpenCV) to RGB format.
    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    # Convert image to tensor and add a batch dimension.
    image_tensor = transform(image_rgb)
    image_tensor = image_tensor.unsqueeze(0)

    # Run the model
    with torch.no_grad():
        predictions = model(image_tensor)[0]

    detected_objects = []
    # Loop through predictions and filter by confidence threshold.
    for idx, score in enumerate(predictions["scores"]):
        if score >= threshold:
            label_idx = predictions["labels"][idx].item()
            if label_idx < len(COCO_INSTANCE_CATEGORY_NAMES):
                label = COCO_INSTANCE_CATEGORY_NAMES[label_idx]
                detected_objects.append(label)
        else:
            # Since predictions are in descending order, we can break early.
            break

    return {"objects": detected_objects}

def perform_ocr(image_cv):
    """
    Perform Optical Character Recognition (OCR) on the image using pytesseract.
    
    Args:
        image_cv (numpy.ndarray): Image in OpenCV BGR format.
    
    Returns:
        str: Extracted text from the image.
    """
    # Convert image to RGB (pytesseract expects RGB)
    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    # Optionally, convert the image to grayscale and apply thresholding for better OCR results.
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    # Use pytesseract to extract text.
    text = pytesseract.image_to_string(gray)
    return text.strip()

def generate_description(detections, ocr_text):
    """
    Combine object detection and OCR results into a single description.
    
    Args:
        detections (dict): Output from detect_objects().
        ocr_text (str): Text extracted via OCR.
    
    Returns:
        str: A description of the image.
    """
    objects = detections.get("objects", [])
    objects_text = f"Detected objects: {', '.join(objects)}." if objects else "No objects detected."
    ocr_text = f"Extracted text: {ocr_text}" if ocr_text else "No text detected."
    description = f"{objects_text} {ocr_text}"
    return description

