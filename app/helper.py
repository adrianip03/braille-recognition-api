"""
Image processing utilities for Braille detection.
Handles image splitting, bounding box management, and NMS.
"""

import cv2
import numpy as np
from PIL import ImageDraw
import base64

# Sliding window parameters
WINDOW_PERCENTAGE = 0.4  # Window size as percentage of image
WINDOW_NUM = 3           # Number of windows per dimension

# Braille dot spacing standards
STD_BOX_RATIO = 47 / 79      # Standard aspect ratio for Braille characters
BOX_RATIO_ERROR_BOUND = 0.3  # Allowed deviation from standard ratio

def split_image(image):
    """
    Split image into overlapping sliding windows for better detection.
    
    Args:
        image: Input PIL Image
        
    Returns:
        image_list: List of window images plus original image
    """
    numpy_image_rgb = np.asarray(image)
    cv2_image = cv2.cvtColor(numpy_image_rgb, cv2.COLOR_RGB2BGR)
        
    image_height = cv2_image.shape[0]
    image_width = cv2_image.shape[1]
    window_height = round(image_height * WINDOW_PERCENTAGE)
    window_width = round(image_width * WINDOW_PERCENTAGE)
    
    window_shift_vertical = round(image_height / float(WINDOW_NUM))
    window_shift_horizontal = round(image_width / float(WINDOW_NUM))
    
    y_offset = 0
    x_offset = 0
    image_list = []
    
    for i in range(WINDOW_NUM):
        y_offset = i * window_shift_vertical
        for j in range(WINDOW_NUM):
            x_offset = j * window_shift_horizontal
            
            # Ensure window stays within image bounds
            y_start = min(y_offset, image_height - window_height)
            x_start = min(x_offset, image_width - window_width)
            y_end = min(y_start + window_height, image_height)
            x_end = min(x_start + window_width, image_width)
            
            window_image = cv2_image[y_start:y_end, x_start:x_end]
            image_list.append(window_image)
    
    # Add original image for full context
    image_list.append(cv2_image)            
    return image_list

def get_boxes_offset(original_size):
    """
    Calculate offset positions for each window relative to original image.
    
    Args:
        original_size: (width, height) of original image
        
    Returns:
        offset_list: List of (x, y) offsets for each window
    """
    image_width, image_height = original_size
    window_height = round(image_height * WINDOW_PERCENTAGE)
    window_width = round(image_width * WINDOW_PERCENTAGE)
    
    window_shift_vertical = round(image_height / float(WINDOW_NUM))
    window_shift_horizontal = round(image_width / float(WINDOW_NUM))
    
    offset_list = []
    
    for i in range(WINDOW_NUM):
        y_offset = i * window_shift_vertical
        for j in range(WINDOW_NUM):
            x_offset = j * window_shift_horizontal
            
            y_start = min(y_offset, image_height - window_height)
            x_start = min(x_offset, image_width - window_width)
            
            offset_list.append((x_start, y_start))
    
    # Original image has no offset
    offset_list.append((0, 0))
    return offset_list

class Box: 
    """Bounding box container with class, confidence, and coordinates."""
    
    def __init__(self, cls, conf, xyxy, offset):
        """
        Initialize box with offset adjustment.
        
        Args:
            cls: Class index
            conf: Confidence score
            xyxy: [x1, y1, x2, y2] coordinates
            offset: (x_offset, y_offset) to add to coordinates
        """
        self.cls = cls
        self.conf = conf

        x1, y1, x2, y2 = xyxy[0].tolist()
        x_offset, y_offset = offset
        
        x1 += x_offset
        x2 += x_offset
        y1 += y_offset
        y2 += y_offset
        
        self.xyxy = np.array([[x1, y1, x2, y2]], dtype=np.float32)

def get_merged_boxes(results, original_size):
    """
    Merge detection results from all windows.
    Filters boxes based on expected Braille character aspect ratio.
    
    Args:
        results: Detection results from each window
        original_size: (width, height) of original image
        
    Returns:
        merged_boxes: List of valid Box objects
    """
    offset_list = get_boxes_offset(original_size)
    merged_boxes = []
    
    for idx, result in enumerate(results):
        offset = offset_list[idx]
        boxes = [Box(box.cls, box.conf, box.xyxy, offset) for box in result.boxes]
        
        for box in boxes: 
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            boxw = x2 - x1
            boxh = y2 - y1
            box_rat = float(boxw) / boxh
            
            # Only keep boxes with aspect ratio similar to Braille characters
            if (box_rat <= STD_BOX_RATIO * (1 + BOX_RATIO_ERROR_BOUND) and 
                box_rat >= STD_BOX_RATIO * (1 - BOX_RATIO_ERROR_BOUND)):
                merged_boxes.append(box)
    
    return merged_boxes

def get_area(xyxy):
    """Calculate area of bounding box."""
    x1, y1, x2, y2 = xyxy
    return max(x2 - x1 + 1, 0) * max(y2 - y1 + 1, 0)

def get_intersection_area(xyxy1, xyxy2):
    """Calculate intersection area between two bounding boxes."""
    xyxyi = (
        max(xyxy1[0], xyxy2[0]), 
        max(xyxy1[1], xyxy2[1]), 
        min(xyxy1[2], xyxy2[2]), 
        min(xyxy1[3], xyxy2[3])
    )
    return get_area(xyxyi)

def get_IOU(xyxy1, xyxy2, a1, a2):
    """
    Calculate Intersection over Union (IoU) for two boxes.
    
    Args:
        xyxy1, xyxy2: Box coordinates
        a1, a2: Areas of the boxes
    """
    ia = get_intersection_area(xyxy1, xyxy2)
    return float(ia) / (a1 + a2 - ia)

def non_max_suppression(boxes, iou_threshold):
    """
    Apply Non-Maximum Suppression to remove duplicate detections.
    
    Args:
        boxes: List of Box objects
        iou_threshold: IoU threshold for considering boxes as duplicates
        
    Returns:
        keep: List of boxes after NMS
    """
    # Sort by confidence (highest first)
    sorted_boxes = sorted(boxes, key=lambda x: x.conf, reverse=True)
    sorted_boxes_with_areas = [(box, get_area(box.xyxy[0].tolist())) for box in sorted_boxes]
    
    keep = []
    while len(sorted_boxes_with_areas) > 0:
        highest_conf_box, highest_conf_box_area = sorted_boxes_with_areas.pop(0)
        
        # Calculate IoU with all remaining boxes
        IOUs = [
            get_IOU(highest_conf_box.xyxy[0].tolist(), box.xyxy[0].tolist(), 
                    highest_conf_box_area, boxa) 
            for (box, boxa) in sorted_boxes_with_areas
        ]
        
        # Keep only boxes with low overlap
        new_sorted_boxes_with_areas = []
        for box_with_area, iou in zip(sorted_boxes_with_areas, IOUs):
            if iou < iou_threshold:
                new_sorted_boxes_with_areas.append(box_with_area)
        
        keep.append(highest_conf_box)
        sorted_boxes_with_areas = new_sorted_boxes_with_areas       
        
    return keep

def get_normalized_bounding_box(boxes, images_size):
    """
    Calculate normalized bounding box coordinates (0-1 range).
    
    Args:
        boxes: List of Box objects
        images_size: (width, height) of original image
        
    Returns:
        [x_center, y_center, width, height] normalized
    """
    box_array = np.array([box.xyxy[0].tolist() for box in boxes])        
    x1_min = np.min(box_array[:, 0])  
    y1_min = np.min(box_array[:, 1])  
    x2_max = np.max(box_array[:, 2])
    y2_max = np.max(box_array[:, 3])
    
    img_width, img_height = images_size
    return [
        float(x1_min) / img_width, 
        float(y1_min) / img_height, 
        float(x2_max - x1_min) / img_width, 
        float(y2_max - y1_min) / img_height
    ]

def visualize(image, boxes, names):
    """Draw bounding boxes and labels on image for debugging."""
    img_with_boxes = image.copy()
    draw = ImageDraw.Draw(img_with_boxes)
    import random
    
    colors = dict()
    for box in boxes: 
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        class_name = names[int(box.cls[0])]
        
        if class_name not in colors:
            colors[class_name] = tuple(random.randint(50, 255) for _ in range(3))    
        color = colors[class_name]
        
        # Draw box outline
        for i in range(3):
            draw.rectangle([x1-i, y1-i, x2+i, y2+i], outline=color, width=1)
        
        # Create label
        label = f"{class_name}"
        if hasattr(box, 'conf') and box.conf is not None:
            conf_value = float(box.conf[0]) if hasattr(box.conf, '__len__') else float(box.conf)
            label += f" {conf_value:.2f}"
        
        # Draw label background and text
        bbox = draw.textbbox((0, 0), label)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        draw.rectangle(
            [x1, y1 - text_height - 5, x1 + text_width + 10, y1],
            fill=color
        )
        draw.text(
            (x1 + 5, y1 - text_height - 3),
            label,
            fill=(0, 0, 0),
        )

    img_with_boxes.show()

def inpaint_image(img, boxes):
    """
    Remove detected Braille characters from image using inpainting.
    
    Args:
        img: Original PIL Image
        boxes: List of Box objects to remove
        
    Returns:
        b64_inpainted_png: Base64 encoded PNG of inpainted image
    """
    numpy_image_rgb = np.asarray(img)
    cv2_image = cv2.cvtColor(numpy_image_rgb, cv2.COLOR_BGR2RGB)
    mask = cv2.cvtColor(numpy_image_rgb, cv2.COLOR_BGR2GRAY)
    mask[:, :] = 0
    
    # Create mask covering all detected boxes
    for box in boxes:         
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, -1)    
        
    # Apply inpainting
    inpainted_image = cv2.inpaint(cv2_image, mask, 5, cv2.INPAINT_TELEA)
    
    # Encode as base64 PNG
    success, buf = cv2.imencode(".png", inpainted_image)
    if not success:
        raise RuntimeError("PNG encoding failed")
    
    return base64.b64encode(buf.tobytes()).decode("utf-8")