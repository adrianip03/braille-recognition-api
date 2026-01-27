import cv2
import numpy as np

WINDOW_PERCENTAGE = 0.4
WINDOW_NUM = 3

def split_image(image):
    numpy_image_rgb = np.asarray(image)
    cv2_image = cv2.cvtColor(numpy_image_rgb, cv2.COLOR_RGB2BGR)
        
    image_height = cv2_image.shape[0]
    image_width = cv2_image.shape[1]
    window_height = round(image_height * WINDOW_PERCENTAGE)
    window_width = round(image_width * WINDOW_PERCENTAGE)
    
    window_shift_vertical = round(image_height / float(WINDOW_NUM))
    window_shift_horizontal = round(image_width / float(WINDOW_NUM))
    
    y_offest = 0
    x_offset = 0
    image_list = []
    for i in range(WINDOW_NUM):
        y_offest = i * window_shift_vertical
        for j in range(WINDOW_NUM):
            x_offset = j * window_shift_horizontal
            
            y_start = min(y_offest, image_height - window_height)
            x_start = min(x_offset, image_width - window_width)
            y_end = min(y_start + window_height, image_height)
            x_end = min(x_start + window_width, image_width)
            
            window_image = cv2_image[y_start:y_end, x_start:x_end]
            image_list.append(window_image)
            
    return image_list


def get_boxes_offset(original_size):
    image_width, image_height = original_size
    window_height = round(image_height * WINDOW_PERCENTAGE)
    window_width = round(image_width * WINDOW_PERCENTAGE)
    
    window_shift_vertical = round(image_height / float(WINDOW_NUM))
    window_shift_horizontal = round(image_width / float(WINDOW_NUM))
    
    offset_list = []
    
    for i in range(WINDOW_NUM):
        y_offest = i * window_shift_vertical
        for j in range(WINDOW_NUM):
            x_offset = j * window_shift_horizontal
            
            y_start = min(y_offest, image_height - window_height)
            x_start = min(x_offset, image_width - window_width)
            
            offset = (x_start, y_start)
            offset_list.append(offset)
            
    return offset_list

class Box: 
    def __init__(self, cls, conf, xyxy, offset):
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
    offset_list = get_boxes_offset(original_size)
    merged_boxes = []
    for idx, result in enumerate(results):
        offset = offset_list[idx]
        boxes = [Box(box.cls, box.conf, box.xyxy, offset) for box in result.boxes]
        merged_boxes.extend(boxes)
    return merged_boxes

def get_area(xyxy):
    x1, y1, x2, y2 = xyxy
    return max(x2-x1+1, 0) * max(y2-y1+1, 0)
    
def get_intersection_area(xyxy1, xyxy2):
    xyxyi = (max(xyxy1[0], xyxy2[0]), max(xyxy1[1], xyxy2[1]), min(xyxy1[2], xyxy2[2]), min(xyxy1[3], xyxy2[3]))
    intersection_area = get_area(xyxyi)
    return intersection_area

def get_IOU(xyxy1, xyxy2, a1, a2):
    ia = get_intersection_area(xyxy1,xyxy2)
    return float(ia) / (a1 + a2 - ia)
    
# N^2 time cry should be optimizable
def non_max_suppression(boxes, iou_threshold):
    sorted_boxes = sorted(boxes, key= lambda x: x.conf, reverse=True)
    sorted_boxes_with_areas = [(box, get_area(box.xyxy[0].tolist())) for box in sorted_boxes]
    
    keep = []
    while len(sorted_boxes_with_areas) > 0:
        highest_conf_box, highest_conf_box_area = sorted_boxes_with_areas.pop(0)
        keep.append(highest_conf_box)
        
        IOUs = [get_IOU(highest_conf_box.xyxy[0].tolist(), box.xyxy[0].tolist(), highest_conf_box_area, boxa) for (box, boxa) in sorted_boxes_with_areas]
    
        new_sorted_boxes_with_areas = []
        for box_with_a, iou in zip(sorted_boxes_with_areas, IOUs):
            if iou < iou_threshold:
                new_sorted_boxes_with_areas.append(box_with_a)
        sorted_boxes_with_areas = new_sorted_boxes_with_areas       
        
    return keep    

def get_normalized_bounding_box(boxes, images_size):
    box_array = np.array([box.xyxy[0].tolist() for box in boxes])        
    x1_min = np.min(box_array[:, 0])  
    y1_min = np.min(box_array[:, 1])  
    x2_max = np.max(box_array[:, 2])
    y2_max = np.max(box_array[:, 3])
    
    img_width, img_height = images_size
    return [float(x1_min) / img_width, float(y1_min) / img_height, float(x2_max - x1_min) / img_width, float(y2_max - y1_min) / img_height]
